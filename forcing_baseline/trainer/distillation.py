"""Stage-3 trainer: DMD distillation for causal DreamID-V face swapping.

Alternates generator (DMD loss) and critic (denoising loss) updates with a
``dfake_gen_update_ratio``.  The generator is rolled out autoregressively via
Self-Forcing; the bidirectional DreamID-V-Faster teacher (``real_score``, frozen)
and the bidirectional critic (``fake_score``) score the rollout, with CFG applied
to ``img_ref``.  Conditioning (``y`` / ``img_ref``) comes from the swap-latent
LMDB; the ground-truth latent is ignored by the generator (backward-simulated).
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

        if self.is_main_process and not self.disable_wandb and wandb is not None:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(config=OmegaConf.to_container(config, resolve=True), name=config.config_name,
                       mode="online", entity=config.wandb_entity, project=config.wandb_project,
                       dir=config.wandb_save_dir)

        self.output_path = config.logdir

        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

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
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)
        if self.is_main_process:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ema_weight = getattr(config, "ema_weight", 0.0)
        self.generator_ema = None
        if ema_weight and ema_weight > 0.0:
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)
        if self.step < getattr(config, "ema_start_step", 0):
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def save(self):
        generator_state_dict = fsdp_state_dict(self.model.generator)
        if self.generator_ema is not None and self.config.ema_start_step < self.step:
            state_dict = {"generator_ema": self.generator_ema.state_dict()}
        else:
            state_dict = {"generator": generator_state_dict}
        if self.is_main_process:
            ckpt_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(state_dict, os.path.join(ckpt_dir, "model.pt"))
            print("Model saved to", ckpt_dir)

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

            if self.is_main_process and not self.disable_wandb and wandb is not None:
                wandb_loss_dict = {"critic_loss": critic_log_dict["critic_loss"].mean().item(),
                                   "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()}
                if train_generator:
                    wandb_loss_dict.update({
                        "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                        "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                        "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()})
                wandb.log(wandb_loss_dict, step=self.step)

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
