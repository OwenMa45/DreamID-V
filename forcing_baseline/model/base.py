"""Base models for the causal DreamID-V distillation framework.

``BaseModel`` owns the causal generator, the text encoder and the VAE (used by
Stage-1 AR diffusion and Stage-2 consistency distillation).  ``SelfForcingModel``
additionally builds the bidirectional real / fake score networks and the AR
backward-simulation rollout used by Stage-3 DMD.

All face-swap conditioning (``y`` = source-video+mask latent, ``img_ref`` =
reference-face latent) travels inside ``conditional_dict`` and is threaded
transparently through the rollout pipeline.
"""
from typing import Tuple

import torch
from torch import nn

from pipeline import SelfForcingTrainingPipeline
from utils.loss import get_denoising_loss
from utils.dreamidv_wrapper import (
    DreamIDVDiffusionWrapper,
    DreamIDVTextEncoder,
    DreamIDVVAEWrapper,
)


def _wrapper_kwargs(args):
    return dict(
        model_config=getattr(args, "model_kwargs", {}),
        timestep_shift=getattr(args, "timestep_shift", 8.0),
        local_attn_size=getattr(args, "local_attn_size", -1),
        sink_size=getattr(args, "sink_size", 1),
        num_max_frames=getattr(args, "num_training_frames", 21),
    )


class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 3)
        self.dtype = torch.bfloat16 if getattr(args, "mixed_precision", True) else torch.float32

        self._initialize_models(args, device)

        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(
                args.denoising_step_list, dtype=torch.long, device=self.device)
            if getattr(args, "warp_denoising_step", False):
                timesteps = torch.cat(
                    (self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).to(self.device)
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_models(self, args, device):
        self.generator = DreamIDVDiffusionWrapper(is_causal=True, **_wrapper_kwargs(args))
        self.generator.model.requires_grad_(True)

        self.text_encoder = DreamIDVTextEncoder(args.t5_checkpoint, args.t5_tokenizer)
        self.text_encoder.requires_grad_(False)

        self.vae = DreamIDVVAEWrapper(args.vae_checkpoint)
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(self, min_timestep, max_timestep, batch_size, num_frame,
                      num_frame_per_block, uniform_timestep=False):
        if uniform_timestep:
            return torch.randint(min_timestep, max_timestep, [batch_size, 1],
                                 device=self.device, dtype=torch.long).repeat(1, num_frame)
        timestep = torch.randint(min_timestep, max_timestep, [batch_size, num_frame],
                                 device=self.device, dtype=torch.long)
        # same noise level within a chunk
        timestep = timestep.reshape(timestep.shape[0], -1, num_frame_per_block)
        timestep[:, :, 1:] = timestep[:, :, 0:1]
        return timestep.reshape(timestep.shape[0], -1)


class SelfForcingModel(BaseModel):
    """Adds the bidirectional real/fake score nets and the AR rollout (DMD)."""

    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(getattr(args, "denoising_loss_type", "flow"))()
        self.inference_pipeline = None

    def _initialize_models(self, args, device):
        super()._initialize_models(args, device)

        self.real_score = DreamIDVDiffusionWrapper(is_causal=False, **_wrapper_kwargs(args))
        self.real_score.model.requires_grad_(False)

        self.fake_score = DreamIDVDiffusionWrapper(is_causal=False, **_wrapper_kwargs(args))
        self.fake_score.model.requires_grad_(True)

    def _run_generator(self, image_or_video_shape, conditional_dict: dict,
                       initial_latent: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """Backward-simulate the generator input from noise and roll it out once."""
        noise_shape = list(image_or_video_shape).copy()
        num_frames = noise_shape[1]
        assert num_frames % self.num_frame_per_block == 0

        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = \
            self._consistency_backward_simulation(
                noise=torch.randn(noise_shape, device=self.device, dtype=self.dtype),
                **conditional_dict)

        gradient_mask = None
        pred_image_or_video = pred_image_or_video.to(self.dtype)
        return pred_image_or_video, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _consistency_backward_simulation(self, noise: torch.Tensor, **conditional_dict) -> torch.Tensor:
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()
        return self.inference_pipeline.inference_with_trajectory(noise=noise, **conditional_dict)

    def _initialize_inference_pipeline(self):
        self.inference_pipeline = SelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=False,
            same_step_across_blocks=getattr(self.args, "same_step_across_blocks", True),
            last_step_only=getattr(self.args, "last_step_only", False),
            num_max_frames=getattr(self.args, "num_training_frames", 21),
            context_noise=getattr(self.args, "context_noise", 0),
        )
