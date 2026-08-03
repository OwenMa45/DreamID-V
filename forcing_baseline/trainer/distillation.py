"""Stage-3 trainer: DMD distillation for causal DreamID-V face swapping.

Alternates generator (DMD loss) and critic (denoising loss) updates with a
``dfake_gen_update_ratio``.  The generator is rolled out autoregressively via
Self-Forcing; the bidirectional DreamID-V-Faster teacher (``real_score``, frozen)
and the bidirectional critic (``fake_score``) score the rollout, with CFG applied
to ``img_ref``.  Conditioning (``y`` / ``img_ref``) comes from the swap-latent
LMDB; the ground-truth latent is ignored by the generator (backward-simulated).

Crash-resume: every ``save()`` also (atomically) writes ``<logdir>/resume_state.pt``
holding the FULL training state -- step + raw generator + critic (+ EMA shadow).
DMD is adversarial, so generator and critic must resume *together*; the per-step
``model.pt`` (generator or EMA only, consumed by inference) stays unchanged.
On start-up the latest state in ``logdir`` is picked up automatically
(``auto_resume``, on by default; explicit ``resume_ckpt`` takes priority).
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

from model import DMD
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

        # ---- crash-resume (Stage-3 / 3a) ------------------------------------
        # Resolve BEFORE building the DMD model: the resume payload supersedes
        # the generator / critic init ckpts, so their (multi-GB) loads inside
        # the DMD ctor are skipped; the frozen real_score teacher still loads
        # from real_score_ckpt as usual.
        resume_path = self._resolve_resume_state(config)
        self._resume_payload = None
        if resume_path:
            payload = torch.load(resume_path, map_location="cpu")
            self.step = int(payload.get("step", 0))
            self._resume_payload = payload
            if self.is_main_process:
                print(f"[distillation] RESUME @ step {self.step} from {resume_path} "
                      "(generator+critic+EMA from state; AdamW moments restart)")
            if "generator" in payload:
                config.generator_ckpt = ""
            if "critic" in payload:
                config.fake_score_ckpt = ""

        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Restore raw generator + critic into the *un-wrapped* modules (purely
        # local op; loading after FSDP-wrap goes through a full state-dict
        # all-gather/scatter that can desync ranks on slow shared storage).
        if self._resume_payload is not None:
            self._load_resume_weights(self._resume_payload)

        # Sync after the (network-backed) ckpt/T5/VAE loading inside the model ctor,
        # before any FSDP collective, so a fast rank does not race into the first
        # all-gather while a slow rank is still reading from shared storage
        # (-> NCCL watchdog _ALLGATHER_BASE timeout). See utils/distributed.py.
        barrier()
        self.model.generator = fsdp_wrap(
            self.model.generator, sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision, wrap_strategy=config.generator_fsdp_wrap_strategy)
        self.model.real_score = fsdp_wrap(
            self.model.real_score, sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision, wrap_strategy=config.real_score_fsdp_wrap_strategy)
        self.model.fake_score = fsdp_wrap(
            self.model.fake_score, sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision, wrap_strategy=config.fake_score_fsdp_wrap_strategy)
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder, sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision, wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False))
        self.model.vae = self.model.vae.to(device=self.device, dtype=self.dtype)

        self.generator_optimizer = torch.optim.AdamW(
            [p for p in self.model.generator.parameters() if p.requires_grad],
            lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)
        self.critic_optimizer = torch.optim.AdamW(
            [p for p in self.model.fake_score.parameters() if p.requires_grad],
            lr=getattr(config, "lr_critic", config.lr),
            betas=(getattr(config, "beta1_critic", config.beta1), getattr(config, "beta2_critic", config.beta2)),
            weight_decay=config.weight_decay)

        dataset = SwapLatentLMDBDataset(config.data_path, max_pair=int(1e8))
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
        num_workers = int(getattr(config, "num_workers", 8))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=num_workers,
            persistent_workers=num_workers > 0)
        if self.is_main_process:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ema_weight = getattr(config, "ema_weight", 0.0)
        self.generator_ema = None
        if ema_weight and ema_weight > 0.0:
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)
        if self.step < getattr(config, "ema_start_step", 0):
            self.generator_ema = None

        # On resume past ema_start_step, continue the EMA target from the saved
        # shadow rather than re-initialising it from the resumed student weights.
        if self._resume_payload is not None:
            if self.generator_ema is not None and "generator_ema" in self._resume_payload:
                self._restore_ema_shadow(self._resume_payload["generator_ema"])
            self._resume_payload = None  # free the CPU copies

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    # ---- crash-resume helpers ------------------------------------------------
    @staticmethod
    def _clean_key(k):
        # EMA shadows are collected AFTER FSDP auto-wrap, so their keys carry
        # `_fsdp_wrapped_module.` mangling on inner submodules; strip it so keys
        # match across save/restore (and across differing wrap layouts).
        return k.replace("_fsdp_wrapped_module.", "")

    def _resolve_resume_state(self, config):
        """Return the resume_state.pt path to continue from, else None.

        Priority: explicit ``resume_ckpt`` (a resume_state.pt file, or a dir
        holding one) > auto-scan of ``logdir``. Auto-resume is on by default and
        can be turned off with ``auto_resume: false`` (--no-auto-resume).
        """
        explicit = getattr(config, "resume_ckpt", None)
        if explicit:
            path = str(explicit)
            if os.path.isdir(path):
                path = os.path.join(path, "resume_state.pt")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"resume_ckpt given but not found: {path}")
            return path

        if not bool(getattr(config, "auto_resume", True)):
            return None

        logdir = getattr(config, "logdir", None)
        if logdir:
            path = os.path.join(logdir, "resume_state.pt")
            if os.path.isfile(path):
                return path
        return None

    def _load_resume_weights(self, payload):
        """Restore raw generator + critic into the un-wrapped modules.

        The AdamW moments are NOT restored -- same policy as the Stage-1/2
        resumes: lr is constant (no scheduler state), beta1=0 leaves the first
        moment empty anyway, and the second moment re-warms within a few steps.
        """
        if "generator" in payload:
            sd = {self._clean_key(k): v for k, v in payload["generator"].items()}
            missing, unexpected = self.model.generator.load_state_dict(sd, strict=False)
            if self.is_main_process:
                print(f"[distillation] resumed generator: {len(missing)} missing, "
                      f"{len(unexpected)} unexpected")
        if "critic" in payload:
            sd = {self._clean_key(k): v for k, v in payload["critic"].items()}
            missing, unexpected = self.model.fake_score.load_state_dict(sd, strict=False)
            if self.is_main_process:
                print(f"[distillation] resumed critic (fake_score): {len(missing)} missing, "
                      f"{len(unexpected)} unexpected")

    def _restore_ema_shadow(self, saved):
        """Copy a saved EMA shadow into the freshly-initialised EMA_FSDP shadow
        (keys matched after stripping FSDP mangling; unmatched keep init)."""
        current = self.generator_ema.shadow
        by_clean = {self._clean_key(k): k for k in current}
        matched = 0
        for k, v in saved.items():
            tgt = by_clean.get(self._clean_key(k))
            if tgt is not None:
                current[tgt] = v.detach().clone().float().cpu()
                matched += 1
        if self.is_main_process:
            print(f"[distillation] EMA shadow restored: {matched}/{len(current)} tensors")

    def save(self):
        # fsdp_state_dict is a collective -- EVERY rank must call it (non-rank0
        # ranks get an empty dict back); only rank 0 writes to disk.
        generator_state_dict = fsdp_state_dict(self.model.generator)
        critic_state_dict = fsdp_state_dict(self.model.fake_score)
        if self.generator_ema is not None and self.config.ema_start_step < self.step:
            state_dict = {"generator_ema": self.generator_ema.state_dict()}
        else:
            state_dict = {"generator": generator_state_dict}
        if self.is_main_process:
            ckpt_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(state_dict, os.path.join(ckpt_dir, "model.pt"))
            print("Model saved to", ckpt_dir)

            # Crash-resume state: step + raw generator + critic (+ EMA shadow).
            # DMD is adversarial -- resuming the generator with a re-initialised
            # critic would corrupt the DMD gradient for a while -- so the full
            # state travels together. Only the LATEST state is kept (~17 GB
            # fp32); written to a tmp file and os.replace'd so a crash mid-write
            # never corrupts an existing resume point.
            if bool(getattr(self.config, "save_resume_state", True)):
                resume_state = {
                    "step": self.step,
                    "generator": generator_state_dict,
                    "critic": critic_state_dict,
                }
                if self.generator_ema is not None:
                    resume_state["generator_ema"] = self.generator_ema.state_dict()
                tmp_path = os.path.join(self.output_path, "resume_state.pt.tmp")
                torch.save(resume_state, tmp_path)
                os.replace(tmp_path, os.path.join(self.output_path, "resume_state.pt"))
                print(f"Resume state (step {self.step}) saved to",
                      os.path.join(self.output_path, "resume_state.pt"))

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        conditional_dict, unconditional_dict = build_swap_conditioning(
            self.model, batch, self.device, self.dtype)
        # Derive the latent geometry [B, F, C, H, W] from the data so the rollout
        # noise matches the conditioning resolution (e.g. 640px square crops ->
        # 78x78 latent), instead of trusting the static config value.
        image_or_video_shape = list(batch["clean_latent"].shape)
        image_or_video_shape[0] = conditional_dict["img_ref"].shape[0]

        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict, unconditional_dict=unconditional_dict,
                clean_latent=None, initial_latent=None)
            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})
            return generator_log_dict

        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict, unconditional_dict=unconditional_dict,
            clean_latent=None, initial_latent=None)
        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
        critic_log_dict.update({"critic_loss": critic_loss, "critic_grad_norm": critic_grad_norm})
        return critic_log_dict

    def train(self):
        start_step = self.step
        max_steps = int(getattr(self.config, "max_steps", 0) or 0)
        # A finished run re-launched by mistake must not train (and overwrite
        # the resume state) any further.
        if max_steps and self.step >= max_steps:
            if self.is_main_process:
                print(f"Resumed step {self.step} >= max_steps={max_steps}; nothing to do.")
            return
        while True:
            train_generator = self.step % self.config.dfake_gen_update_ratio == 0

            if train_generator:
                self.generator_optimizer.zero_grad(set_to_none=True)
                batch = next(self.dataloader)
                generator_log_dict = self.fwdbwd_one_step(batch, True)
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            self.critic_optimizer.zero_grad(set_to_none=True)
            batch = next(self.dataloader)
            critic_log_dict = self.fwdbwd_one_step(batch, False)
            self.critic_optimizer.step()
            self.step += 1

            if (self.step >= getattr(self.config, "ema_start_step", 0)) and \
                    (self.generator_ema is None) and (getattr(self.config, "ema_weight", 0.0) > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            reached_end = bool(max_steps) and self.step >= max_steps
            if (not self.config.no_save) and (self.step - start_step) > 0 and (
                    self.step % self.config.log_iters == 0 or reached_end):
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # wandb disabled (offline server): no metric logging.
            # if self.is_main_process and not self.disable_wandb and wandb is not None:
            #     wandb_loss_dict = {"critic_loss": critic_log_dict["critic_loss"].mean().item(),
            #                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()}
            #     if train_generator:
            #         wandb_loss_dict.update({
            #             "generator_loss": generator_log_dict["generator_loss"].mean().item(),
            #             "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
            #             "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()})
            #     wandb.log(wandb_loss_dict, step=self.step)

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
