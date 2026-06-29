"""Stage-1 trainer: causal AR diffusion (teacher forcing) for DreamID-V.

Reads the Stage-0 self-distilled swap-latent LMDB and trains the causal generator
with a flow-matching MSE objective.  Face-swap conditioning (``y`` / ``img_ref``)
is threaded through ``conditional_dict``.
"""
import gc
import os
import time

import torch
import torch.distributed as dist

try:
    import wandb
except ImportError:  # wandb is optional
    wandb = None

from omegaconf import OmegaConf

from model import CausalDiffusion
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

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_wandb = getattr(config, "disable_wandb", True)

        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()
        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb and wandb is not None:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(config=OmegaConf.to_container(config, resolve=True), name=config.config_name,
                       mode="online", entity=config.wandb_entity, project=config.wandb_project,
                       dir=config.wandb_save_dir)

        self.output_path = config.logdir

        # model + FSDP
        self.model = CausalDiffusion(config, device=self.device)
        self.model.generator = fsdp_wrap(
            self.model.generator, sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision, wrap_strategy=config.generator_fsdp_wrap_strategy)
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder, sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision, wrap_strategy=config.text_encoder_fsdp_wrap_strategy)
        self.model.vae = self.model.vae.to(device=self.device, dtype=self.dtype)

        if getattr(config, "generator_ckpt", False):
            print(f"Loading init causal generator from {config.generator_ckpt}")
            self._load_generator_ckpt(config.generator_ckpt)

        self.generator_optimizer = torch.optim.AdamW(
            [p for p in self.model.generator.parameters() if p.requires_grad],
            lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)

        dataset = SwapLatentLMDBDataset(config.data_path, max_pair=int(1e8))
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)
        if self.is_main_process:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ema_weight = getattr(config, "ema_weight", 0.0)
        self.generator_ema = None
        if ema_weight and ema_weight > 0.0:
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        self.max_grad_norm = getattr(config, "max_grad_norm", 10.0)
        self.previous_time = None

    def _load_generator_ckpt(self, path):
        state_dict = torch.load(path, map_location="cpu")
        if "generator" in state_dict:
            state_dict = state_dict["generator"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        fixed = {k.replace("model._fsdp_wrapped_module.", "model."): v for k, v in state_dict.items()}
        self.model.generator.load_state_dict(fixed, strict=False)

    def save(self):
        generator_state_dict = fsdp_state_dict(self.model.generator)
        state_dict = {"generator": generator_state_dict}
        if self.generator_ema is not None and self.config.ema_start_step < self.step:
            state_dict["generator_ema"] = self.generator_ema.state_dict()
        if self.is_main_process:
            ckpt_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(state_dict, os.path.join(ckpt_dir, "model.pt"))
            print("Model saved to", ckpt_dir)

    def train_one_step(self, batch):
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        clean_latent = batch["clean_latent"].to(device=self.device, dtype=self.dtype)
        conditional_dict, unconditional_dict = build_swap_conditioning(
            self.model, batch, self.device, self.dtype)

        batch_size = clean_latent.shape[0]
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        generator_loss, _ = self.model.generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=None)

        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)
        self.generator_optimizer.step()
        if self.generator_ema is not None:
            self.generator_ema.update(self.model.generator)
        self.step += 1

        if self.is_main_process and not self.disable_wandb and wandb is not None:
            wandb.log({"generator_loss": generator_loss.item(),
                       "generator_grad_norm": generator_grad_norm.item()}, step=self.step)

        if self.step % self.config.gc_interval == 0:
            gc.collect()

    def train(self):
        max_steps = int(getattr(self.config, "max_steps", 0) or 0)
        while True:
            batch = next(self.dataloader)
            self.train_one_step(batch)
            reached_end = bool(max_steps) and self.step >= max_steps
            if (not self.config.no_save) and (
                    self.step % self.config.log_iters == 0 or reached_end):
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()
            barrier()
            if self.is_main_process and self.previous_time is None:
                self.previous_time = time.time()
            if reached_end:
                if self.is_main_process:
                    print(f"Reached max_steps={max_steps}; stopping.")
                break
