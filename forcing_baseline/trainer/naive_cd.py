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

try:
    import wandb
except ImportError:
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

        if self.is_main_process and not self.disable_wandb and wandb is not None:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(config=OmegaConf.to_container(config, resolve=True), name=config.config_name,
                       mode="online", entity=config.wandb_entity, project=config.wandb_project,
                       dir=config.wandb_save_dir)

        self.output_path = config.logdir

        self.model = NaiveConsistency(config, device=self.device)
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
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)
        if self.is_main_process:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ema_weight = getattr(config, "ema_weight", 0.95)
        self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.previous_time = None

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

            if self.is_main_process and not self.disable_wandb and wandb is not None:
                wandb.log({"generator_loss": generator_log_dict["generator_loss"].mean().item(),
                           "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item()},
                          step=self.step)

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
