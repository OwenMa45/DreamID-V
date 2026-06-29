"""Causal AR teacher rollout used by Stage-2 Consistency Distillation.

``inference_for_genuine_cd`` reproduces the genuine-CD teacher transition used in
Causal-Forcing's ``NaiveConsistency``: given a noisy chunk at timestep ``t`` and
the clean GT context preceding it, run one AR teacher step (with img_ref CFG) to
obtain the chunk at the next (lower) timestep ``t_next``.

DreamID-V adaptations:
  * reference-face prefix cached at cache-frame 0 for both the positive (real
    img_ref) and negative (zeroed img_ref) branches,
  * clean GT context chunks cached at cache-frames ``1 ..``,
  * conditioning ``y`` sliced to each chunk.
"""
from typing import Optional

import torch

from utils.dreamidv_wrapper import DreamIDVDiffusionWrapper, DreamIDVTextEncoder


class CausalDiffusionInferencePipeline(torch.nn.Module):
    def __init__(self, args, device, generator=None, text_encoder=None, scheduler=None):
        super().__init__()
        self.args = args
        self.device = device
        self.generator = generator if generator is not None else DreamIDVDiffusionWrapper(
            model_config=getattr(args, "model_kwargs", {}), is_causal=True)
        self.text_encoder = text_encoder
        self.scheduler = scheduler  # FlowMatchScheduler shared with the CD model

        self.num_train_timesteps = args.num_train_timestep
        self.shift = getattr(args, "timestep_shift", 5.0)
        self.guidance_scale = args.guidance_scale
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 3)
        self.local_attn_size = self.generator.model.local_attn_size
        self.num_max_frames = getattr(args, "num_training_frames", 21)
        self.kv_cache_size = (self.num_max_frames + 1) * self.frame_seq_length

        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None

    @staticmethod
    def _split_cond(conditional_dict):
        cond = {k: v for k, v in conditional_dict.items() if k not in ("y", "img_ref")}
        return cond, conditional_dict.get("y", None), conditional_dict.get("img_ref", None)

    def _chunk_cond(self, base_cond, y, img_ref, frame_start, num_frames):
        block = dict(base_cond)
        if img_ref is not None:
            block["img_ref"] = img_ref
        if y is not None:
            block["y"] = y[:, :, frame_start:frame_start + num_frames]
        return block

    @torch.no_grad()
    def inference_for_genuine_cd(self, noisy_input, conditional_dict=None, unconditional_dict=None,
                                 initial_latent: Optional[torch.Tensor] = None,
                                 timestep_idx=0, sampling_steps=48, chunksize=3) -> torch.Tensor:
        base_cond, y, img_ref = self._split_cond(conditional_dict)
        base_uncond, y_u, img_ref_u = self._split_cond(unconditional_dict)
        batch_size, num_frames, num_channels, height, width = noisy_input.shape
        assert num_frames == chunksize

        # Per-frame token count from the actual latent geometry (patch (1,2,2)):
        # 480p 60x104 -> 1560, 640px square 78x78 -> 1521.
        self.frame_seq_length = (height // 2) * (width // 2)
        new_kv_cache_size = (self.num_max_frames + 1) * self.frame_seq_length
        if new_kv_cache_size != self.kv_cache_size:
            self.kv_cache_size = new_kv_cache_size
            self.kv_cache_pos = None  # force reallocation at the new size

        self._reset_caches(batch_size, noisy_input.dtype, noisy_input.device)

        # Step 0: cache reference-face prefix (sink) for both branches
        self.generator(noisy_image_or_video=None, conditional_dict={**base_cond, "img_ref": img_ref},
                       timestep=None, kv_cache=self.kv_cache_pos, crossattn_cache=self.crossattn_cache_pos,
                       current_start=0, cache_ref=True)
        self.generator(noisy_image_or_video=None, conditional_dict={**base_uncond, "img_ref": img_ref_u},
                       timestep=None, kv_cache=self.kv_cache_neg, crossattn_cache=self.crossattn_cache_neg,
                       current_start=0, cache_ref=True)

        ref_offset = 1
        zero_t = torch.zeros([batch_size, chunksize], device=noisy_input.device, dtype=torch.int64)

        # Step 1: cache the clean GT context chunks
        cache_start_frame = 0  # in video-frame coordinates
        if initial_latent is not None:
            num_input_frames = initial_latent.shape[1]
            assert num_input_frames % chunksize == 0
            for _ in range(num_input_frames // chunksize):
                ctx_latents = initial_latent[:, cache_start_frame:cache_start_frame + chunksize]
                cur = (cache_start_frame + ref_offset) * self.frame_seq_length
                self.generator(noisy_image_or_video=ctx_latents,
                               conditional_dict=self._chunk_cond(base_cond, y, img_ref, cache_start_frame, chunksize),
                               timestep=zero_t * 0, kv_cache=self.kv_cache_pos,
                               crossattn_cache=self.crossattn_cache_pos, current_start=cur)
                self.generator(noisy_image_or_video=ctx_latents,
                               conditional_dict=self._chunk_cond(base_uncond, y_u, img_ref_u, cache_start_frame, chunksize),
                               timestep=zero_t * 0, kv_cache=self.kv_cache_neg,
                               crossattn_cache=self.crossattn_cache_neg, current_start=cur)
                cache_start_frame += chunksize

        # Step 2: one teacher AR step on the noisy chunk with img_ref CFG
        t = self.scheduler.timesteps[timestep_idx]
        timestep = t * torch.ones([batch_size, chunksize], device=noisy_input.device, dtype=torch.float32)
        cur = (cache_start_frame + ref_offset) * self.frame_seq_length

        flow_pred_cond, _ = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=self._chunk_cond(base_cond, y, img_ref, cache_start_frame, chunksize),
            timestep=timestep, kv_cache=self.kv_cache_pos, crossattn_cache=self.crossattn_cache_pos,
            current_start=cur)
        flow_pred_uncond, _ = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=self._chunk_cond(base_uncond, y_u, img_ref_u, cache_start_frame, chunksize),
            timestep=timestep, kv_cache=self.kv_cache_neg, crossattn_cache=self.crossattn_cache_neg,
            current_start=cur)
        flow_pred = flow_pred_uncond + self.guidance_scale * (flow_pred_cond - flow_pred_uncond)

        latents = self.scheduler.step(
            flow_pred.flatten(0, 1), t, noisy_input.flatten(0, 1)
        ).unflatten(0, (batch_size, chunksize))
        return latents

    # ----- caches -----
    def _reset_caches(self, batch_size, dtype, device):
        if self.kv_cache_pos is None:
            self.kv_cache_pos = self._make_kv_cache(batch_size, dtype, device)
            self.kv_cache_neg = self._make_kv_cache(batch_size, dtype, device)
            self.crossattn_cache_pos = self._make_crossattn_cache(batch_size, dtype, device)
            self.crossattn_cache_neg = self._make_crossattn_cache(batch_size, dtype, device)
        else:
            for cache in (self.kv_cache_pos, self.kv_cache_neg):
                for blk in cache:
                    blk["global_end_index"].fill_(0)
                    blk["local_end_index"].fill_(0)
            for cache in (self.crossattn_cache_pos, self.crossattn_cache_neg):
                for blk in cache:
                    blk["is_init"] = False

    def _make_kv_cache(self, batch_size, dtype, device):
        return [{
            "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        } for _ in range(self.num_transformer_blocks)]

    def _make_crossattn_cache(self, batch_size, dtype, device):
        return [{
            "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
            "is_init": False,
        } for _ in range(self.num_transformer_blocks)]
