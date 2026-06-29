"""Few-step auto-regressive streaming inference for the distilled face-swapper.

Given the conditioning (``y`` = source-video + mask latent, ``img_ref`` =
reference-face latent, ``prompt_embeds``) the distilled causal generator produces
the swapped-video latent chunk by chunk with a KV cache:

  cache frame 0      -> reference-face sink (cached once)
  cache frames 1..F  -> generated video frames

Each chunk is denoised with the (few-step) ``denoising_step_list`` and then
re-cached at (near) zero noise so subsequent chunks attend to clean context.
"""
from typing import List

import torch

from utils.scheduler import FlowMatchScheduler
from utils.dreamidv_wrapper import DreamIDVDiffusionWrapper


class CausalInferencePipeline(torch.nn.Module):
    def __init__(self, args, device, generator=None, scheduler: FlowMatchScheduler = None,
                 denoising_step_list: List[int] = None):
        super().__init__()
        self.args = args
        self.device = device
        self.generator = generator if generator is not None else DreamIDVDiffusionWrapper(
            model_config=getattr(args, "model_kwargs", {}), is_causal=True)

        self.frame_seq_length = 1560
        self.num_transformer_blocks = 30
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 3)
        self.context_noise = getattr(args, "context_noise", 0)
        self.num_max_frames = getattr(args, "num_training_frames", 21)
        self.kv_cache_size = (self.num_max_frames + 1) * self.frame_seq_length

        if scheduler is None:
            scheduler = FlowMatchScheduler(shift=getattr(args, "timestep_shift", 8.0),
                                           sigma_min=0.0, extra_one_step=True)
            scheduler.set_timesteps(1000, training=True)
        self.scheduler = scheduler

        steps = denoising_step_list if denoising_step_list is not None else getattr(
            args, "denoising_step_list", [1000, 750, 500, 250])
        self.denoising_step_list = torch.tensor(steps, dtype=torch.long, device=device)
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

        self.kv_cache = None
        self.crossattn_cache = None

    @staticmethod
    def _split_cond(conditional_dict):
        cond = {k: v for k, v in conditional_dict.items() if k not in ("y", "img_ref")}
        return cond, conditional_dict.get("y", None), conditional_dict.get("img_ref", None)

    def _block_cond(self, base_cond, y, img_ref, frame_start, num_frames):
        block = dict(base_cond)
        if img_ref is not None:
            block["img_ref"] = img_ref
        if y is not None:
            block["y"] = y[:, :, frame_start:frame_start + num_frames]
        return block

    @torch.no_grad()
    def inference(self, conditional_dict, num_frames, height_lat=60, width_lat=104,
                  num_channels=16, dtype=torch.bfloat16) -> torch.Tensor:
        base_cond, y, img_ref = self._split_cond(conditional_dict)
        assert img_ref is not None and y is not None
        batch_size = img_ref.shape[0]
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block

        # Per-frame token count from the actual latent geometry (patch (1,2,2)):
        # 480p 60x104 -> 1560, 640px square 78x78 -> 1521.  Keeps the KV-cache
        # layout (cache offsets / size) exactly aligned with the model.
        self.frame_seq_length = (height_lat // 2) * (width_lat // 2)
        self.kv_cache_size = (self.num_max_frames + 1) * self.frame_seq_length

        noise = torch.randn([batch_size, num_frames, num_channels, height_lat, width_lat],
                            device=self.device, dtype=dtype)
        output = torch.zeros_like(noise)

        self._init_caches(batch_size, dtype, self.device)

        # cache reference prefix (sink)
        self.generator(noisy_image_or_video=None,
                       conditional_dict={**base_cond, "img_ref": img_ref},
                       timestep=None, kv_cache=self.kv_cache, crossattn_cache=self.crossattn_cache,
                       current_start=0, cache_ref=True)
        ref_offset = 1

        current_start_frame = 0
        for _ in range(num_blocks):
            nf = self.num_frame_per_block
            noisy_input = noise[:, current_start_frame:current_start_frame + nf]
            block_cond = self._block_cond(base_cond, y, img_ref, current_start_frame, nf)
            cur = (current_start_frame + ref_offset) * self.frame_seq_length

            denoised_pred = None
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones([batch_size, nf], device=self.device, dtype=torch.int64) * current_timestep
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input, conditional_dict=block_cond,
                    timestep=timestep, kv_cache=self.kv_cache, crossattn_cache=self.crossattn_cache,
                    current_start=cur)
                if index < len(self.denoising_step_list) - 1:
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1), torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones([batch_size * nf], device=self.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])

            output[:, current_start_frame:current_start_frame + nf] = denoised_pred

            # recache clean context
            context_timestep = torch.ones([batch_size, nf], device=self.device, dtype=torch.int64) * self.context_noise
            ctx = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1), torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep.flatten(0, 1).to(torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            self.generator(noisy_image_or_video=ctx, conditional_dict=block_cond,
                           timestep=context_timestep, kv_cache=self.kv_cache,
                           crossattn_cache=self.crossattn_cache, current_start=cur)

            current_start_frame += nf

        return output

    def _init_caches(self, batch_size, dtype, device):
        self.kv_cache = [{
            "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        } for _ in range(self.num_transformer_blocks)]
        self.crossattn_cache = [{
            "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
            "is_init": False,
        } for _ in range(self.num_transformer_blocks)]
