"""Stage-1: causal autoregressive flow-matching diffusion (teacher forcing).

The causal student is trained to predict the flow target on each chunk, while
attending block-causally to clean ground-truth context (teacher forcing) and to
the cached reference-face prefix.  Face-swap conditioning ``y`` / ``img_ref`` is
carried inside ``conditional_dict``.
"""
from typing import Tuple

import torch

from model.base import BaseModel


class CausalDiffusion(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        if getattr(args, "gradient_checkpointing", False):
            self.generator.enable_gradient_checkpointing()

        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.guidance_scale = getattr(args, "guidance_scale", 1.0)
        self.timestep_shift = getattr(args, "timestep_shift", 8.0)
        self.teacher_forcing = getattr(args, "teacher_forcing", True)
        self.noise_augmentation_max_timestep = getattr(args, "noise_augmentation_max_timestep", 0)

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        noise = torch.randn_like(clean_latent)
        batch_size, num_frame = clean_latent.shape[:2]

        index = self._get_timestep(
            0, self.scheduler.num_train_timesteps,
            batch_size, num_frame, self.num_frame_per_block, uniform_timestep=False)
        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)
        noisy_latents = self.scheduler.add_noise(
            clean_latent.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))
        training_target = self.scheduler.training_target(clean_latent, noise, timestep)

        if self.noise_augmentation_max_timestep > 0:
            index_aug = self._get_timestep(
                self.noise_augmentation_max_timestep, 1000,
                batch_size, num_frame, self.num_frame_per_block, uniform_timestep=False)
            timestep_clean_aug = self.scheduler.timesteps[index_aug].to(dtype=self.dtype, device=self.device)
            clean_latent_aug = self.scheduler.add_noise(
                clean_latent.flatten(0, 1), noise.flatten(0, 1), timestep_clean_aug.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
        else:
            clean_latent_aug = clean_latent
            timestep_clean_aug = None

        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=clean_latent_aug if self.teacher_forcing else None,
            aug_t=timestep_clean_aug if self.teacher_forcing else None,
        )

        loss = torch.nn.functional.mse_loss(
            flow_pred.float(), training_target.float(), reduction="none").mean(dim=(2, 3, 4))
        loss = loss * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
        loss = loss.mean()

        return loss, {"x0": clean_latent.detach(), "x0_pred": x0_pred.detach()}
