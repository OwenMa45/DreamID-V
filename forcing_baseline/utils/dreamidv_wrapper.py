"""Model wrappers for the causal DreamID-V face-swap distillation framework.

These mirror Causal-Forcing's ``utils/wan_wrapper.py`` but:
  * use DreamID-V's VAE / T5 / backbone, and
  * thread the face-swap conditioning (``y`` = source-video+mask latent,
    ``img_ref`` = reference-face latent) through ``conditional_dict``.

``DreamIDVDiffusionWrapper`` exposes the exact call signature the Causal-Forcing
pipelines expect (``noisy_image_or_video, conditional_dict, timestep, kv_cache,
crossattn_cache, current_start, cache_start, clean_x, aug_t``) plus a
``cache_ref`` flag used once per rollout to write the reference-face prefix into
the KV cache.
"""
import types
from typing import List, Optional

import torch
from torch import nn

from utils.scheduler import FlowMatchScheduler, SchedulerInterface
from wan.modules.t5 import umt5_xxl
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.vae import _video_vae
from wan.modules.dreamidv_model import WanModel
from wan.modules.causal_dreamidv_model import CausalDreamIDVModel


# ----------------------------------------------------------------------------- text
class DreamIDVTextEncoder(torch.nn.Module):
    """UMT5-XXL text encoder (DreamID-V uses a fixed 'change face' style prompt)."""

    def __init__(self, t5_checkpoint: str, t5_tokenizer: str):
        super().__init__()
        self.text_encoder = umt5_xxl(
            encoder_only=True, return_tokenizer=False,
            dtype=torch.float32, device=torch.device("cpu"),
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(t5_checkpoint, map_location="cpu", weights_only=False))
        self.tokenizer = HuggingfaceTokenizer(name=t5_tokenizer, seq_len=512, clean="whitespace")

    @property
    def device(self):
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0
        return {"prompt_embeds": context}


# ----------------------------------------------------------------------------- vae
class DreamIDVVAEWrapper(torch.nn.Module):
    """Wan2.1 VAE used by DreamID-V, with [B,F,C,H,W] in/out convention."""

    def __init__(self, vae_checkpoint: str, z_dim: int = 16):
        super().__init__()
        mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
        std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
               3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.model = _video_vae(pretrained_path=vae_checkpoint, z_dim=z_dim).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [B, C, F, H, W] -> latent [B, F, C, H, W]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype), 1.0 / self.std.to(device=device, dtype=dtype)]
        out = [self.model.encode(u.unsqueeze(0), scale).float().squeeze(0) for u in pixel]
        out = torch.stack(out, dim=0).permute(0, 2, 1, 3, 4)
        return out

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype), 1.0 / self.std.to(device=device, dtype=dtype)]
        decode_fn = self.model.cached_decode if use_cache else self.model.decode
        out = [decode_fn(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0) for u in zs]
        out = torch.stack(out, dim=0).permute(0, 2, 1, 3, 4)
        return out


# ----------------------------------------------------------------------------- backbone
_DEFAULT_MODEL_CONFIG = dict(
    model_type="i2v", patch_size=(1, 2, 2), text_len=512, in_dim=48, dim=1536,
    ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16, num_heads=12,
    num_layers=30, qk_norm=True, cross_attn_norm=True, eps=1e-6, in_dim_ref_conv=16,
)

FRAME_SEQ_LENGTH = 1560  # 480p latent: (60/2) * (104/2)


class DreamIDVDiffusionWrapper(torch.nn.Module):
    def __init__(self, model_config: Optional[dict] = None, timestep_shift: float = 8.0,
                 is_causal: bool = False, local_attn_size: int = -1, sink_size: int = 1,
                 num_max_frames: int = 21, **kwargs):
        super().__init__()
        cfg = dict(_DEFAULT_MODEL_CONFIG)
        if model_config:
            cfg.update(model_config)
        # model_kwargs may carry scheduler-level params (e.g. timestep_shift) that
        # the backbone constructor does not accept. timestep_shift is consumed as a
        # dedicated arg above, so drop such non-architecture keys from cfg.
        for _non_arch_key in ("timestep_shift",):
            cfg.pop(_non_arch_key, None)

        if is_causal:
            self.model = CausalDreamIDVModel(local_attn_size=local_attn_size, sink_size=sink_size, **cfg)
        else:
            # bidirectional teacher / critic backbone (window_size for global attn)
            bcfg = dict(cfg)
            bcfg["window_size"] = (-1, -1)
            self.model = WanModel(**bcfg)
        self.model.eval()

        self.is_causal = is_causal
        self.uniform_timestep = not is_causal
        self.frame_seq_length = FRAME_SEQ_LENGTH
        self.seq_len = num_max_frames * self.frame_seq_length

        self.scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.post_init()

    # ----- checkpoint helpers -----
    def load_checkpoint(self, path: str, strict: bool = False):
        sd = torch.load(path, map_location="cpu")
        for key in ("generator", "model", "generator_ema", "state_dict"):
            if isinstance(sd, dict) and key in sd:
                sd = sd[key]
                break
        fixed = {}
        for k, v in sd.items():
            k = k.replace("model._fsdp_wrapped_module.", "model.")
            if k.startswith("model."):
                k = k[len("model."):]
            fixed[k] = v
        missing, unexpected = self.model.load_state_dict(fixed, strict=strict)
        print(f"[DreamIDVDiffusionWrapper] loaded {path}: "
              f"{len(missing)} missing, {len(unexpected)} unexpected")

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.model, "enable_gradient_checkpointing"):
            self.model.enable_gradient_checkpointing()

    # ----- flow <-> x0 conversion (identical to Causal-Forcing) -----
    def _convert_flow_pred_to_x0(self, flow_pred, xt, timestep):
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps])
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred, xt, timestep):
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device),
            [x0_pred, xt, scheduler.sigmas, scheduler.timesteps])
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    # ----- forward -----
    def forward(self, noisy_image_or_video, conditional_dict, timestep,
                kv_cache: Optional[List[dict]] = None,
                crossattn_cache: Optional[List[dict]] = None,
                current_start: Optional[int] = None,
                cache_start: Optional[int] = None,
                clean_x: Optional[torch.Tensor] = None,
                aug_t: Optional[torch.Tensor] = None,
                cache_ref: bool = False):
        prompt_embeds = conditional_dict["prompt_embeds"]
        img_ref = conditional_dict.get("img_ref", None)
        y = conditional_dict.get("y", None)            # [B, 32, F, H, W]

        # ---- reference-prefix KV caching (causal rollout, once per video) ----
        if cache_ref:
            ref_t = torch.zeros([img_ref.shape[0], 1], device=img_ref.device, dtype=torch.long)
            self.model(
                None, t=ref_t, context=prompt_embeds, seq_len=self.seq_len,
                img_ref=img_ref, y=None,
                kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                current_start=current_start, cache_start=cache_start)
            return None

        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds, seq_len=self.seq_len,
                img_ref=None, y=y,
                kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                current_start=current_start, cache_start=cache_start,
            ).permute(0, 2, 1, 3, 4)
        elif clean_x is not None:
            # teacher forcing (causal student)
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds, seq_len=self.seq_len,
                img_ref=img_ref, y=y,
                clean_x=clean_x.permute(0, 2, 1, 3, 4), aug_t=aug_t,
            ).permute(0, 2, 1, 3, 4)
        elif self.is_causal:
            # causal diffusion-forcing (no cache, no teacher-forcing context)
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds, seq_len=self.seq_len,
                img_ref=img_ref, y=y).permute(0, 2, 1, 3, 4)
        else:
            # bidirectional teacher / critic (DreamID-V WanModel expects list inputs).
            x_list = list(noisy_image_or_video.permute(0, 2, 1, 3, 4))   # [C, F, h, w] per sample
            ctx_list = list(prompt_embeds)                               # [L, D] per sample
            y_list = list(y) if y is not None else None                  # [32, F, h, w] per sample
            ref_list = list(img_ref) if img_ref is not None else None    # [16, 1, h, w] per sample
            # exact video token count for this resolution (matches the teacher's
            # own seq_len convention): (h/ph)*(w/pw)*F.  480p -> 32760, 640px -> 31941.
            ph, pw = self.model.patch_size[1], self.model.patch_size[2]
            f0, h0, w0 = x_list[0].shape[1], x_list[0].shape[2], x_list[0].shape[3]
            bidir_seq_len = (h0 // ph) * (w0 // pw) * f0
            out = self.model(
                x_list, t=input_timestep, context=ctx_list, seq_len=bidir_seq_len,
                img_ref=ref_list, y=y_list)
            out = torch.stack(out)                                       # [B, C, F, h, w]
            flow_pred = out.permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])
        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        self.get_scheduler()
