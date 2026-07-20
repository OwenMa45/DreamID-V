"""Stage-2 trainer: causal Consistency Distillation (CD) for DreamID-V.

Initialises student / teacher / EMA from the Stage-1 AR checkpoint and trains the
student with the genuine-CD objective (only ground-truth swap latents needed).
The frozen causal teacher performs the AR one-step transition with img_ref CFG.
"""
import gc
import os
import time

import torch
import torch.distributed as dist

# wandb disabled: this server has no internet access, so we never record runs.
# `wandb` is forced to None -> every guarded `wandb is not None` block below stays
# inert. To re-enable later, restore the import and set disable_wandb: false.
# try:
#     import wandb
# except ImportError:
#     wandb = None
wandb = None

from omegaconf import OmegaConf

from model import NaiveConsistency
from utils.dataset import cycle, SwapLatentLMDBDataset
from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import set_seed, build_swap_conditioning


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_wandb = getattr(config, "disable_wandb", True)

        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()
        set_seed(config.seed + global_rank)

        # wandb disabled (offline server): no login / init.
        # if self.is_main_process and not self.disable_wandb and wandb is not None:
        #     wandb.login(host=config.wandb_host, key=config.wandb_key)
        #     wandb.init(config=OmegaConf.to_container(config, resolve=True), name=config.config_name,
        #                mode="online", entity=config.wandb_entity, project=config.wandb_project,
        #                dir=config.wandb_save_dir)

        self.output_path = config.logdir

        self.model = NaiveConsistency(config, device=self.device)

        # ---- Stage-2 resume -------------------------------------------------
        # Continue the STUDENT (+EMA) from the latest Stage-2 checkpoint in logdir
        # (or an explicit resume_ckpt); the TEACHER keeps its Stage-1 init -- it is
        # the frozen AR teacher and must not drift. Stage-2 ckpts saved past
        # ema_start_step contain only the EMA shadow ("generator_ema"), so the
        # student restarts from the EMA weights (closest available to the raw
        # student). All loads happen on the *un-wrapped* modules, before FSDP.
        resume_path, resume_step = self._resolve_resume_ckpt(config)
        self._resume_shadow = None
        if resume_path:
            if self.is_main_process:
                print(f"[naive_cd] RESUME @ step {resume_step} from {resume_path} "
                      "(student+EMA from ckpt; teacher keeps Stage-1 init)")
            sd = torch.load(resume_path, map_location="cpu")
            if isinstance(sd, dict) and "generator" in sd:
                payload = sd["generator"]
            elif isinstance(sd, dict) and "generator_ema" in sd:
                payload = sd["generator_ema"]
            else:
                payload = sd
            if isinstance(sd, dict) and "generator_ema" in sd:
                # raw (possibly FSDP-mangled) keys, used to restore the EMA shadow
                self._resume_shadow = sd["generator_ema"]
            clean = {}
            for k, v in payload.items():
                k = k.replace("_fsdp_wrapped_module.", "")
                if k.startswith("model."):
                    k = k[len("model."):]
                clean[k] = v
            missing, unexpected = self.model.generator.model.load_state_dict(clean, strict=False)
            self.model.generator_ema.model.load_state_dict(clean, strict=False)
            if self.is_main_process:
                print(f"[naive_cd] resumed student weights: {len(missing)} missing, "
                      f"{len(unexpected)} unexpected")
            del sd, payload, clean
            self.step = resume_step

        # Sync after the (network-backed) ckpt/T5/VAE loading inside the model ctor,
        # before any FSDP collective, so a fast rank does not race into the first
        # all-gather while a slow rank is still reading from shared storage
        # (-> NCCL watchdog _ALLGATHER_BASE timeout). See utils/distributed.py.
        barrier()
        self.model.generator = fsdp_wrap(
            self.model.generator, sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision, wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=True)
        self.model.generator_ema = fsdp_wrap(
            self.model.generator_ema, sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision, wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=True)
        self.model.teacher = fsdp_wrap(
            self.model.teacher, sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=getattr(config, "real_score_fsdp_wrap_strategy", config.generator_fsdp_wrap_strategy),
            cpu_offload=True)
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder, sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision, wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=True)

        self.generator_optimizer = torch.optim.AdamW(
            [p for p in self.model.generator.parameters() if p.requires_grad],
            lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)

        dataset = SwapLatentLMDBDataset(config.data_path, max_pair=int(1e8))
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
        num_workers = int(getattr(config, "num_workers", 8))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=num_workers,
            persistent_workers=num_workers > 0)
        if self.is_main_process:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ema_weight = getattr(config, "ema_weight", 0.95)
        self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)
        # On resume, continue the EMA target from the saved shadow rather than
        # re-initialising it from the (already EMA-valued) student weights.
        if self._resume_shadow is not None:
            self._restore_ema_shadow(self._resume_shadow)
            self._resume_shadow = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.previous_time = None

    # ---- resume helpers ----------------------------------------------------
    @staticmethod
    def _step_from_ckpt(path):
        # .../checkpoint_model_001000/model.pt -> 1000
        name = os.path.basename(os.path.dirname(path))
        try:
            return int(name.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            return 0

    def _resolve_resume_ckpt(self, config):
        """Return (model_pt_path, start_step) to resume from, else (None, 0).

        Priority: explicit ``resume_ckpt`` (a dir or a model.pt) > auto-scan of
        ``logdir`` for the highest ``checkpoint_model_XXXXXX``. Auto-resume is on
        by default and can be turned off with ``auto_resume: false``.
        """
        explicit = getattr(config, "resume_ckpt", None)
        if explicit:
            path = explicit if str(explicit).endswith(".pt") else os.path.join(str(explicit), "model.pt")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"resume_ckpt given but not found: {path}")
            return path, self._step_from_ckpt(path)

        if not bool(getattr(config, "auto_resume", True)):
            return None, 0

        logdir = getattr(config, "logdir", None)
        best_path, best_step = None, -1
        if logdir and os.path.isdir(logdir):
            for name in os.listdir(logdir):
                if not name.startswith("checkpoint_model_"):
                    continue
                p = os.path.join(logdir, name, "model.pt")
                if not os.path.isfile(p):
                    continue
                try:
                    s = int(name.rsplit("_", 1)[-1])
                except (ValueError, IndexError):
                    continue
                if s > best_step:
                    best_path, best_step = p, s
        if best_path is not None:
            return best_path, best_step
        return None, 0

    def _restore_ema_shadow(self, saved):
        """Copy a saved EMA shadow into the freshly-initialised EMA_FSDP shadow.

        Keys are matched after stripping FSDP's ``_fsdp_wrapped_module.`` mangling
        so shadows saved by this run's convention (post-wrap names) and any clean
        variant both restore correctly. Unmatched entries keep their init value.
        """
        def _clean(k):
            return k.replace("_fsdp_wrapped_module.", "")

        current = self.generator_ema.shadow
        by_clean = {_clean(k): k for k in current}
        matched = 0
        for k, v in saved.items():
            tgt = by_clean.get(_clean(k))
            if tgt is not None:
                current[tgt] = v.detach().clone().float().cpu()
                matched += 1
        if self.is_main_process:
            print(f"[naive_cd] EMA shadow restored: {matched}/{len(current)} tensors")

    def save(self):
        generator_state_dict = fsdp_state_dict(self.model.generator)
        if self.config.ema_start_step < self.step:
            state_dict = {"generator_ema": self.generator_ema.state_dict()}
        else:
            state_dict = {"generator": generator_state_dict}
        if self.is_main_process:
            ckpt_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(state_dict, os.path.join(ckpt_dir, "model.pt"))
            print("Model saved to", ckpt_dir)

    def fwdbwd_one_step(self, batch, clean_latent):
        self.model.eval()
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        conditional_dict, unconditional_dict = build_swap_conditioning(
            self.model, batch, self.device, self.dtype)

        generator_loss, generator_log_dict = self.model.generator_loss(
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            ema_model=self.generator_ema)
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
        generator_log_dict.update({"generator_loss": generator_loss,
                                   "generator_grad_norm": generator_grad_norm})
        return generator_log_dict

    def train(self):
        start_step = self.step
        max_steps = int(getattr(self.config, "max_steps", 0) or 0)
        while True:
            self.generator_optimizer.zero_grad(set_to_none=True)
            batch = next(self.dataloader)
            clean_latent = batch["clean_latent"].to(device=self.device, dtype=self.dtype)
            generator_log_dict = self.fwdbwd_one_step(batch, clean_latent=clean_latent)

            self.generator_optimizer.step()
            if self.generator_ema is not None:
                self.generator_ema.update(self.model.generator)
            self.step += 1

            reached_end = bool(max_steps) and self.step >= max_steps
            if (not self.config.no_save) and (self.step - start_step) > 0 and (
                    self.step % self.config.log_iters == 0 or reached_end):
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # wandb disabled (offline server): no metric logging.
            # if self.is_main_process and not self.disable_wandb and wandb is not None:
            #     wandb.log({"generator_loss": generator_log_dict["generator_loss"].mean().item(),
            #                "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item()},
            #               step=self.step)

            if self.step % self.config.gc_interval == 0:
                gc.collect()
                torch.cuda.empty_cache()
            barrier()
            if self.is_main_process and self.previous_time is None:
                self.previous_time = time.time()
            if reached_end:
                if self.is_main_process:
                    print(f"Reached max_steps={max_steps}; stopping.")
                break
