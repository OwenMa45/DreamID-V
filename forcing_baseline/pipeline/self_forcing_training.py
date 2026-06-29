"""Self-Forcing AR rollout for Stage-3 DMD generator backward simulation.

Adapted from Causal-Forcing's ``SelfForcingTrainingPipeline`` for the DreamID-V
face-swap conditioning:

  * The reference-face latent (``conditional_dict['img_ref']``) is written into the
    KV cache once at the start as the **sink prefix frame** (cache frame 0), so
    every generated video frame attends to the identity reference.
  * Video frames therefore live at cache frames ``1 .. F`` (``current_start`` is
    offset by one frame).
  * The per-frame conditioning ``y`` (``[B, 32, F, h, w]``) is sliced to the
    current chunk before each generator call.
"""
from typing import List, Optional

import torch
import torch.distributed as dist

from utils.scheduler import SchedulerInterface
from utils.dreamidv_wrapper import DreamIDVDiffusionWrapper


class SelfForcingTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: DreamIDVDiffusionWrapper,
                 num_frame_per_block: int = 3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise

        self.kv_cache1 = None
        self.crossattn_cache = None
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        # +1 frame to hold the reference-face sink prefix
        self.kv_cache_size = (num_max_frames + 1) * self.frame_seq_length

    # ----- conditioning helpers -----
    @staticmethod
    def _split_cond(conditional_dict):
        cond = {k: v for k, v in conditional_dict.items() if k not in ("y", "img_ref")}
        y = conditional_dict.get("y", None)
        img_ref = conditional_dict.get("img_ref", None)
        return cond, y, img_ref

    def _block_cond(self, base_cond, y, img_ref, frame_start, num_frames):
        block = dict(base_cond)
        if img_ref is not None:
            block["img_ref"] = img_ref
        if y is not None:
            block["y"] = y[:, :, frame_start:frame_start + num_frames]
        return block

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            indices = torch.randint(low=0, high=num_denoising_steps, size=(num_blocks,), device=device)
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)
        if dist.is_initialized():
            dist.broadcast(indices, src=0)
        return indices.tolist()

    def inference_with_trajectory(self, noise: torch.Tensor,
                                  clean_image_or_video: torch.Tensor = None,
                                  initial_latent: Optional[torch.Tensor] = None,
                                  return_sim_step: bool = False,
                                  **conditional_dict) -> torch.Tensor:
        base_cond, y, img_ref = self._split_cond(conditional_dict)
        batch_size, num_frames, num_channels, height, width = noise.shape
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block
        num_output_frames = num_frames
        output = torch.zeros([batch_size, num_output_frames, num_channels, height, width],
                             device=noise.device, dtype=noise.dtype)

        # Per-frame token count from the actual latent geometry (patch (1,2,2));
        # +1 frame holds the reference-face sink.  480p->1560, 640px square->1521.
        self.frame_seq_length = (height // 2) * (width // 2)
        self.kv_cache_size = (num_frames + 1) * self.frame_seq_length

        self._initialize_kv_cache(batch_size, noise.dtype, noise.device)
        self._initialize_crossattn_cache(batch_size, noise.dtype, noise.device)

        # Step 0: cache reference-face prefix at cache frame 0 (the sink)
        assert img_ref is not None, "DreamID-V rollout requires img_ref conditioning"
        with torch.no_grad():
            self.generator(noisy_image_or_video=None,
                           conditional_dict={**base_cond, "img_ref": img_ref},
                           timestep=None, kv_cache=self.kv_cache1,
                           crossattn_cache=self.crossattn_cache,
                           current_start=0, cache_ref=True)
        ref_offset = 1  # frames; video frames start at cache frame 1

        all_num_frames = [self.num_frame_per_block] * num_blocks
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)

        current_start_frame = 0  # video frame index (output coordinates)
        denoised_pred = None
        timestep = None
        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[:, current_start_frame:current_start_frame + current_num_frames]
            block_cond = self._block_cond(base_cond, y, img_ref, current_start_frame, current_num_frames)
            cache_start = (current_start_frame + ref_offset) * self.frame_seq_length

            for index, current_timestep in enumerate(self.denoising_step_list):
                exit_flag = (index == exit_flags[0]) if self.same_step_across_blocks else (index == exit_flags[block_index])
                timestep = torch.ones([batch_size, current_num_frames], device=noise.device,
                                      dtype=torch.int64) * current_timestep
                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input, conditional_dict=block_cond,
                            timestep=timestep, kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache,
                            current_start=cache_start)
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1), torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input, conditional_dict=block_cond,
                        timestep=timestep, kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache,
                        current_start=cache_start)
                    break

            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # rerun at (near-)zero noise to write clean context into the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            ctx = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1), torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(noisy_image_or_video=ctx, conditional_dict=block_cond,
                               timestep=context_timestep, kv_cache=self.kv_cache1,
                               crossattn_cache=self.crossattn_cache, current_start=cache_start)

            current_start_frame += current_num_frames

        # denoised range bookkeeping (used by DMD timestep schedule)
        if exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.to(noise.device) - self.denoising_step_list[exit_flags[0]].to(noise.device)).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.to(noise.device) - self.denoising_step_list[exit_flags[0] + 1].to(noise.device)).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.to(noise.device) - self.denoising_step_list[exit_flags[0]].to(noise.device)).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1
        return output, denoised_timestep_from, denoised_timestep_to

    # ----- caches -----
    def _initialize_kv_cache(self, batch_size, dtype, device):
        kv_cache1 = []
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })
        self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False,
            })
        self.crossattn_cache = crossattn_cache
