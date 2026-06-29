# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Causal (auto-regressive) DreamID-V backbone.

This module grafts DreamID-V-Faster's face-swap conditioning onto the
Causal-Forcing causal Wan backbone:

* ``patch_embedding`` has ``in_dim=48`` = 16 (noise/latent) + 32 ``y`` channels
  (source-video latent 16 + mask latent 16), channel-concatenated per frame.
* ``ref_conv`` (Conv2d 16->dim) turns the reference-face latent into a single
  *prefix frame* (1560 tokens) that is prepended to the sequence.  In the causal
  model this prefix is treated as the **independent first frame / attention
  sink**: every video frame attends to it, it attends only to itself, and during
  KV-cache rollout it is written once at cache position 0 and never evicted.
* Self-attention is block-causal with a KV cache (chunk-wise, ``num_frame_per_block``
  frames per chunk).  Time modulation is *per frame* so different chunks can sit
  at different noise levels (required for AR diffusion / DMD backward-simulation).

The parameter names exactly match the bidirectional ``WanModel`` in
``dreamidv_model.py`` (``patch_embedding``, ``ref_conv``, ``blocks.*`` q/k/v/o,
``text_embedding``, ``time_embedding``, ``time_projection``, ``head.*``) so the
DreamID-V-Faster checkpoint can be key-copied in (see
``tools/convert_dreamidv_to_causal.py``).
"""
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from .attention import attention
from .dreamidv_model import (
    WanLayerNorm,
    WanRMSNorm,
    rope_apply,
    rope_params,
    sinusoidal_embedding_1d,
)

__all__ = ["CausalDreamIDVModel"]

# wan 1.3B model has a weird channel / head configuration and requires
# max-autotune to work with flexattention, see
# https://github.com/pytorch/pytorch/issues/133254
flex_attention = torch.compile(flex_attention, dynamic=False, mode="default")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """RoPE that places the first local frame at absolute frame ``start_frame``.

    Used in the KV-cache rollout where each chunk is fed individually but must
    receive the absolute temporal position so its rotary phase matches the
    cached prefix / previous chunks.
    """
    n, c = x.size(2), x.size(3) // 2

    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):
    """Block-causal self-attention with optional KV cache.

    Three execution paths:
      * ``kv_cache is None`` and the sequence is twice the per-sample length =>
        teacher-forcing training (``[clean | noisy]`` halves), flex-attention with
        the TF mask.
      * ``kv_cache is None`` otherwise => diffusion-forcing training, flex-attention
        with the block-causal mask.
      * ``kv_cache is not None`` => KV-cache AR inference (one chunk at a time).
    """

    def __init__(self, dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        # Global attention (local_attn_size == -1) must attend the entire KV cache,
        # including the reference-face sink at frame 0; a huge cap disables the
        # local-window truncation in forward() and makes this resolution-agnostic
        # (the old constant 32760 = 21*1560 was tied to the 480p token count and
        # would drop the sink for longer/larger sequences such as 640px -> 1521).
        self.max_attention_size = (1 << 30) if local_attn_size == -1 else local_attn_size * 1560

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask,
                kv_cache=None, current_start=0, cache_start=None):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                # teacher forcing: clean & noisy halves share rope positions
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query, roped_key = [], []
                for ii in range(2):
                    roped_query.append(rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v))
                    roped_key.append(rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v))
                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)
            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat(
                [roped_query,
                 torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                             device=q.device, dtype=v.dtype)], dim=1)
            padded_roped_key = torch.cat(
                [roped_key,
                 torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                             device=k.device, dtype=v.dtype)], dim=1)
            padded_v = torch.cat(
                [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                device=v.device, dtype=v.dtype)], dim=1)

            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask,
            )[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            x = attention(
                roped_query,
                kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
                kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
            )
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanCrossAttention(nn.Module):
    """Text cross-attention with a per-block context cache.

    Matches DreamID-V's plain cross-attention (no CLIP image branch); the context
    (text embedding) is constant across the AR rollout so its projected K/V are
    cached on the first call.
    """

    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, crossattn_cache=None):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"][:, :k.shape[1]] = k
                crossattn_cache["v"][:, :v.shape[1]] = v
                crossattn_cache["is_init"] = True
            else:
                k = crossattn_cache["k"][:, :context.shape[1]]
                v = crossattn_cache["v"][:, :context.shape[1]]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        x = attention(q, k, v, k_lens=context_lens)
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self, dim, ffn_dim, num_heads, local_attn_size=-1, sink_size=0,
                 qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = CausalWanCrossAttention(dim, num_heads, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim))

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens,
                block_mask, kv_cache=None, crossattn_cache=None, current_start=0, cache_start=None):
        # e: [B, F, 6, C] per-frame modulation
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes, freqs, block_mask, kv_cache, current_start, cache_start)
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2))
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        # e: [B, F, 1, C]
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0])
        return x


class CausalDreamIDVModel(ModelMixin, ConfigMixin):
    r"""Auto-regressive DreamID-V face-swap backbone."""

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _no_split_modules = ["CausalWanAttentionBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type="i2v",
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=48,
                 dim=1536,
                 ffn_dim=8960,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=12,
                 num_layers=30,
                 local_attn_size=-1,
                 sink_size=1,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 in_dim_ref_conv=16):
        super().__init__()

        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(dim, ffn_dim, num_heads, local_attn_size,
                                    sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # reference-face prefix conv (identity conditioning -> attention sink)
        self.ref_conv = nn.Conv2d(in_dim_ref_conv, dim, kernel_size=patch_size[1:], stride=patch_size[1:])

        # rope freqs
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ], dim=1)

        self.init_weights()
        self.gradient_checkpointing = False
        self.block_mask = None
        self.num_frame_per_block = 1

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    # ------------------------------------------------------------------ masks
    @staticmethod
    def _block_partition(num_frames, num_frame_per_block, independent_first_frame=True):
        """Return per-frame ``[block_start, block_end)`` (in frame units).

        With ``independent_first_frame`` the reference-prefix frame 0 is its own
        block; video frames 1.. are grouped into chunks of ``num_frame_per_block``.
        """
        starts = [0] * num_frames
        ends = [0] * num_frames
        if independent_first_frame:
            starts[0], ends[0] = 0, 1
            f = 1
            while f < num_frames:
                e = min(f + num_frame_per_block, num_frames)
                for i in range(f, e):
                    starts[i], ends[i] = f, e
                f = e
        else:
            f = 0
            while f < num_frames:
                e = min(f + num_frame_per_block, num_frames)
                for i in range(f, e):
                    starts[i], ends[i] = f, e
                f = e
        return starts, ends

    @classmethod
    def _prepare_blockwise_causal_attn_mask(cls, device, num_frames, frame_seqlen,
                                            num_frame_per_block=1, local_attn_size=-1,
                                            independent_first_frame=True):
        total_length = num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        _, blk_end = cls._block_partition(num_frames, num_frame_per_block, independent_first_frame)
        for fidx in range(num_frames):
            s = fidx * frame_seqlen
            ends[s:s + frame_seqlen] = blk_end[fidx] * frame_seqlen

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None,
                                       Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length,
                                       _compile=False, device=device)
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[causal-dreamidv] block-causal mask, block={num_frame_per_block} frames")
        return block_mask

    @classmethod
    def _prepare_teacher_forcing_mask(cls, device, num_frames, frame_seqlen,
                                      num_frame_per_block=1, independent_first_frame=True):
        """TF mask over a ``[clean | noisy]`` doubled sequence (each half ``num_frames``).

        clean tokens => block-causal within the clean half.
        noisy tokens => own (noisy) block + clean context frames in *previous* blocks.
        """
        total_length = num_frames * frame_seqlen * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        clean_ends = num_frames * frame_seqlen

        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        blk_start, blk_end = cls._block_partition(num_frames, num_frame_per_block, independent_first_frame)

        # clean half
        for fidx in range(num_frames):
            s = fidx * frame_seqlen
            context_ends[s:s + frame_seqlen] = blk_end[fidx] * frame_seqlen

        # noisy half (offset by clean_ends)
        for fidx in range(num_frames):
            s = clean_ends + fidx * frame_seqlen
            e = s + frame_seqlen
            noise_noise_starts[s:e] = clean_ends + blk_start[fidx] * frame_seqlen
            noise_noise_ends[s:e] = clean_ends + blk_end[fidx] * frame_seqlen
            # clean context = clean frames strictly before this noisy block
            noise_context_ends[s:e] = blk_start[fidx] * frame_seqlen

        def attention_mask(b, h, q_idx, kv_idx):
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = kv_idx < noise_context_ends[q_idx]
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)
            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None,
                                       Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length,
                                       _compile=False, device=device)
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[causal-dreamidv] teacher-forcing mask, block={num_frame_per_block} frames")
        return block_mask

    # ------------------------------------------------------------------ helpers
    def _ref_tokens(self, img_ref):
        """[B, 16, 1, H, W] -> ([B, frame_seqlen, dim], (h, w)).

        Per-sample: [16, 1, H, W] -transpose-> [1, 16, H, W] -ref_conv->
        [1, dim, h, w] -> [1, h*w, dim].
        """
        outs = []
        for u in img_ref:
            u = u.transpose(0, 1)                        # [1, 16, H, W]
            u = self.ref_conv(u)                         # [1, dim, h, w]
            h, w = u.shape[-2], u.shape[-1]
            outs.append(u.flatten(2).transpose(1, 2))    # [1, h*w, dim]
        return torch.cat(outs, dim=0), (h, w)

    def _embed_time(self, t):
        """t: [B, F] -> e:[B*F, dim], e0:[B, F, 6, dim]."""
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(self.patch_embedding.weight))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        return e, e0

    def _embed_context(self, context):
        return self.text_embedding(torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context
        ]))

    # ------------------------------------------------------------------ inference
    def _forward_inference(self, x, t, context, seq_len, img_ref=None, y=None,
                           kv_cache=None, crossattn_cache=None, current_start=0, cache_start=0):
        """One KV-cache step.

        * Reference caching:  ``x is None`` and ``img_ref`` given.  The reference
          prefix tokens are written to the cache at ``current_start`` (= 0) and
          nothing is returned.
        * Video chunk:        ``x`` = noisy latent [B, 16, F, H, W], ``y`` the
          per-chunk conditioning [B, 32, F, H, W].
        """
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        cache_ref = x is None
        if cache_ref:
            x, (h, w) = self._ref_tokens(img_ref)          # [B, h*w, dim]
            grid_sizes = torch.stack([torch.tensor([1, h, w], dtype=torch.long)
                                      for _ in range(x.shape[0])])
        else:
            if y is not None:
                x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
            x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
            grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x]
            x = torch.cat(x)

        seq_lens = torch.tensor([x.size(1)], dtype=torch.long)

        # per-frame time modulation
        if cache_ref:
            t = torch.zeros([x.shape[0], 1], device=device, dtype=torch.long)
        e, e0 = self._embed_time(t)

        context = self._embed_context(context)
        context_lens = None

        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs,
                      context=context, context_lens=context_lens, block_mask=self.block_mask)

        for block_index, block in enumerate(self.blocks):
            kwargs.update({
                "kv_cache": kv_cache[block_index],
                "crossattn_cache": crossattn_cache[block_index] if crossattn_cache is not None else None,
                "current_start": current_start,
                "cache_start": cache_start,
            })
            x = block(x, **kwargs)

        if cache_ref:
            return None

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    # ------------------------------------------------------------------ training
    def _forward_train(self, x, t, context, seq_len, img_ref=None, y=None,
                       clean_x=None, aug_t=None):
        """Diffusion-forcing (``clean_x is None``) or teacher-forcing forward.

        The reference prefix is prepended as frame 0 (independent first frame /
        sink).  Output strips the reference frame so only the ``F`` video frames
        are returned.
        """
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        ref_tokens, _ = self._ref_tokens(img_ref)          # [B, frame_seqlen, dim]

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_f = x[0].shape[2]
        grid_h, grid_w = x[0].shape[3], x[0].shape[4]
        x = [u.flatten(2).transpose(1, 2) for u in x]
        x = torch.cat(x)                                   # [B, F*fs, dim]
        frame_seqlen = grid_h * grid_w

        # prepend reference frame -> F+1 frames
        x = torch.cat([ref_tokens, x], dim=1)
        num_frames = grid_f + 1
        grid_sizes = torch.stack([torch.tensor([num_frames, grid_h, grid_w], dtype=torch.long)
                                  for _ in range(x.shape[0])])

        # per-frame timestep: ref frame at t=0
        ref_t = torch.zeros([t.shape[0], 1], device=device, dtype=t.dtype)
        t_full = torch.cat([ref_t, t], dim=1)              # [B, F+1]

        # build masks lazily
        if self.block_mask is None:
            if clean_x is not None:
                self.block_mask = self._prepare_teacher_forcing_mask(
                    device, num_frames=num_frames, frame_seqlen=frame_seqlen,
                    num_frame_per_block=self.num_frame_per_block, independent_first_frame=True)
            else:
                self.block_mask = self._prepare_blockwise_causal_attn_mask(
                    device, num_frames=num_frames, frame_seqlen=frame_seqlen,
                    num_frame_per_block=self.num_frame_per_block, local_attn_size=self.local_attn_size,
                    independent_first_frame=True)

        e, e0 = self._embed_time(t_full)
        context = self._embed_context(context)
        context_lens = None

        if clean_x is not None:
            # build clean half (also reference-prefixed), share rope with noisy half
            if y is not None:
                clean_x = [torch.cat([u, v], dim=0) for u, v in zip(clean_x, y)]
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]
            clean_x = torch.cat(clean_x)
            clean_x = torch.cat([ref_tokens, clean_x], dim=1)
            x = torch.cat([clean_x, x], dim=1)             # [B, 2*(F+1)*fs, dim]

            if aug_t is None:
                aug_t = torch.zeros_like(t)
            aug_t_full = torch.cat([ref_t, aug_t], dim=1)
            e_clean, e0_clean = self._embed_time(aug_t_full)
            e0 = torch.cat([e0_clean, e0], dim=1)

        seq_lens = torch.tensor([num_frames * frame_seqlen], dtype=torch.long)

        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs,
                      context=context, context_lens=context_lens, block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kw):
                return module(*inputs, **kw)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(block), x, **kwargs, use_reentrant=False)
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]                     # keep noisy half
            # `e` was built from the noisy-half timesteps already (see above)

        # strip reference frame (frame 0)
        x = x[:, frame_seqlen:]
        e_video = e.unflatten(dim=0, sizes=t_full.shape)[:, 1:]  # [B, F, dim]
        grid_sizes_out = torch.stack([torch.tensor([grid_f, grid_h, grid_w], dtype=torch.long)
                                      for _ in range(x.shape[0])])

        x = self.head(x, e_video.unsqueeze(2))
        x = self.unpatchify(x, grid_sizes_out)
        return torch.stack(x)

    def forward(self, *args, **kwargs):
        if kwargs.get("kv_cache", None) is not None:
            return self._forward_inference(*args, **kwargs)
        return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        nn.init.zeros_(self.head.head.weight)
