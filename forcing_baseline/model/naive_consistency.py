"""Stage-2: causal genuine Consistency Distillation (CD).

Only ground-truth latents are needed.  A frozen causal teacher (initialised from
the Stage-1 AR model, with img_ref CFG) performs one AR step ``latent_t ->
latent_t_next``; the trainable student maps ``latent_t`` and the EMA target maps
``latent_t_next`` to (consistent) clean predictions.  The CD loss is the MSE
between these two predictions.
"""
import random
from typing import Tuple

import torch
import torch.nn.functional as F

from model.base import BaseModel
from utils.scheduler import FlowMatchScheduler
from utils.dreamidv_wrapper import DreamIDVDiffusionWrapper
from pipeline import CausalDiffusionInferencePipeline


class NaiveConsistency(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
            self.generator_ema.model.num_frame_per_block = self.num_frame_per_block
            self.teacher.model.num_frame_per_block = self.num_frame_per_block

        if getattr(args, "generator_ckpt", False):
            print(f"[NaiveConsistency] init student/teacher/ema from {args.generator_ckpt}")
            self.generator.load_checkpoint(args.generator_ckpt, strict=False)
            self.teacher.load_checkpoint(args.generator_ckpt, strict=False)
            self.generator_ema.load_checkpoint(args.generator_ckpt, strict=False)

        if getattr(args, "gradient_checkpointing", False):
            self.generator.enable_gradient_checkpointing()

        self.timestep_shift = getattr(args, "timestep_shift", 8.0)
        self.guidance_scale = args.guidance_scale
        self.num_training_frames = getattr(args, "num_training_frames", 21)
        self.discrete_cd_N = getattr(args, "discrete_cd_N", 48)

        # discrete N-step schedule shared with the teacher AR pipeline
        self.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(num_inference_steps=self.discrete_cd_N, denoising_strength=1.0)
        self.scheduler.sigmas = self.scheduler.sigmas.to(device)
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        self.pipeline = CausalDiffusionInferencePipeline(
            args, device=device, generator=self.teacher,
            text_encoder=self.text_encoder, scheduler=self.scheduler)

    def _initialize_models(self, args, device):
        self.generator = DreamIDVDiffusionWrapper(is_causal=True, **self._wkw(args))
        self.generator.model.requires_grad_(True)

        self.teacher = DreamIDVDiffusionWrapper(is_causal=True, **self._wkw(args))
        self.teacher.model.requires_grad_(False)

        self.generator_ema = DreamIDVDiffusionWrapper(is_causal=True, **self._wkw(args))
        self.generator_ema.model.requires_grad_(False)

        from utils.dreamidv_wrapper import DreamIDVTextEncoder, DreamIDVVAEWrapper
        self.text_encoder = DreamIDVTextEncoder(args.t5_checkpoint, args.t5_tokenizer)
        self.text_encoder.requires_grad_(False)
        self.vae = DreamIDVVAEWrapper(args.vae_checkpoint)
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    @staticmethod
    def _wkw(args):
        from model.base import _wrapper_kwargs
        return _wrapper_kwargs(args)

    def generator_loss(
        self,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        ema_model,
    ) -> Tuple[torch.Tensor, dict]:
        clean_latent = clean_latent.to(self.device).to(torch.bfloat16)
        num_frame = clean_latent.shape[1]
        nfpb = self.num_frame_per_block
        num_blocks = num_frame // nfpb

        # refresh teacher reference (the trainer may have FSDP-wrapped it post-init)
        self.pipeline.generator = self.teacher

        timestep_idx = random.randrange(self.discrete_cd_N - 1)
        t = self.scheduler.timesteps[timestep_idx]
        timestep = t * torch.ones([clean_latent.shape[0], num_frame], device=self.device, dtype=torch.bfloat16)

        noise = torch.randn_like(clean_latent)
        latent_t = self.scheduler.add_noise(
            clean_latent.flatten(0, 1), noise.flatten(0, 1),
            t * torch.ones([clean_latent.shape[0] * num_frame], device=self.device)
        ).unflatten(0, (clean_latent.shape[0], num_frame)).to(torch.bfloat16)

        # teacher AR step latent_t -> latent_t_next, chunk by chunk
        latent_t_next = []
        for chunk_idx in range(1, num_blocks + 1):
            initial_latent = clean_latent[:, :nfpb * (chunk_idx - 1)] if chunk_idx > 1 else None
            latent_t_i = latent_t[:, nfpb * (chunk_idx - 1): nfpb * chunk_idx]
            latent_t_next_i = self.pipeline.inference_for_genuine_cd(
                noisy_input=latent_t_i,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                initial_latent=initial_latent,
                timestep_idx=timestep_idx,
                sampling_steps=self.discrete_cd_N,
                chunksize=nfpb,
            )
            latent_t_next.append(latent_t_next_i)
        latent_t_next = torch.cat(latent_t_next, dim=1)

        t_next = self.scheduler.timesteps[timestep_idx + 1]
        timestep_next = t_next * torch.ones([clean_latent.shape[0], num_frame], device=self.device, dtype=torch.bfloat16)

        _, cm_pred_t = self.generator(latent_t, conditional_dict, timestep, clean_x=clean_latent)

        with torch.no_grad():
            ema_model.copy_to(self.generator_ema)
            _, cm_pred_t_next = self.generator_ema(
                latent_t_next, conditional_dict, timestep_next, clean_x=clean_latent)

        loss = F.mse_loss(cm_pred_t, cm_pred_t_next, reduction="mean")
        log_dict = {
            "unnormalized_loss": F.mse_loss(
                cm_pred_t, cm_pred_t_next, reduction="none").mean(dim=[1, 2, 3, 4]).detach(),
            "t": t.detach(),
            "t_next": t_next.detach(),
        }
        return loss, log_dict
