"""Stage-A: convert bidirectional DreamID-V-Faster weights -> causal init ckpt.

The causal student (``CausalDreamIDVModel``) shares the DreamID-V-Faster layer
names and shapes (``patch_embedding`` in_dim=48, ``ref_conv``, per-block
``self_attn`` q/k/v/o + norm_q/norm_k, ``cross_attn``, ``ffn``, ``norm*`` /
``modulation``, ``text_/time_embedding``, ``time_projection``, ``head``).  The
only structural change is *causal* self-attention (KV-cache + block mask) which
reuses the same projection weights, so a whitelist key-copy is sufficient.

Output format is consumed by ``DreamIDVDiffusionWrapper.load_checkpoint`` and the
trainers (``{"generator": {"model.<key>": tensor}}``).

Example
-------
    python -m tools.convert_dreamidv_to_causal \
        --dreamidv_ckpt /path/to/dreamidv_faster.pth \
        --output_ckpt   checkpoints/causal_init.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wan.modules.causal_dreamidv_model import CausalDreamIDVModel  # noqa: E402


DEFAULT_CFG = dict(
    model_type="i2v", patch_size=(1, 2, 2), text_len=512, in_dim=48, dim=1536,
    ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16, num_heads=12,
    num_layers=30, qk_norm=True, cross_attn_norm=True, eps=1e-6, in_dim_ref_conv=16,
    local_attn_size=-1, sink_size=1,
)


def _unwrap_state_dict(sd):
    for key in ("generator", "model", "state_dict"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
            break
    fixed = {}
    for k, v in sd.items():
        k = k.replace("model._fsdp_wrapped_module.", "model.")
        if k.startswith("model."):
            k = k[len("model."):]
        fixed[k] = v
    return fixed


def main():
    p = argparse.ArgumentParser(description="Stage-A: DreamID-V-Faster -> causal init ckpt")
    p.add_argument("--dreamidv_ckpt", required=True)
    p.add_argument("--output_ckpt", required=True)
    args = p.parse_args()

    print(f"Loading bidirectional weights from {args.dreamidv_ckpt}")
    teacher_sd = _unwrap_state_dict(torch.load(args.dreamidv_ckpt, map_location="cpu"))

    causal = CausalDreamIDVModel(**DEFAULT_CFG)
    causal_sd = causal.state_dict()

    copied, shape_mismatch, missing = {}, [], []
    for k, v in causal_sd.items():
        if k in teacher_sd and tuple(teacher_sd[k].shape) == tuple(v.shape):
            copied[k] = teacher_sd[k]
        elif k in teacher_sd:
            shape_mismatch.append((k, tuple(teacher_sd[k].shape), tuple(v.shape)))
        else:
            missing.append(k)

    extra = [k for k in teacher_sd if k not in causal_sd]
    print(f"copied={len(copied)}  missing_in_teacher={len(missing)}  "
          f"shape_mismatch={len(shape_mismatch)}  teacher_only={len(extra)}")
    for k, ts, cs in shape_mismatch:
        print(f"  [shape] {k}: teacher{ts} vs causal{cs}")
    for k in missing[:20]:
        print(f"  [init-random] {k}")

    load_info = causal.load_state_dict(copied, strict=False)
    print(f"after load: missing={len(load_info.missing_keys)} unexpected={len(load_info.unexpected_keys)}")

    out = {"generator": {f"model.{k}": v for k, v in causal.state_dict().items()}}
    os.makedirs(os.path.dirname(os.path.abspath(args.output_ckpt)), exist_ok=True)
    torch.save(out, args.output_ckpt)
    print(f"Saved causal init checkpoint -> {args.output_ckpt}")


if __name__ == "__main__":
    main()
