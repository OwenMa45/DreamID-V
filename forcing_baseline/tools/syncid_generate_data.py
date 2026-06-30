"""Stage-0: SyncID-Pipe self-distillation data generation.

Run the original *bidirectional* DreamID-V-Faster pipeline as a teacher over a
corpus of (driving-video, reference-face) pairs and store, per sample:

  * ``clean_latent`` : teacher swapped-video latent   [F, 16, h, w]
  * ``y``            : source-video latent (16) + DWPose-mask latent (16)
                       channel-concatenated conditioning  [32, F, h, w]
  * ``img_ref``      : reference-face latent           [16, 1, h, w]
  * ``prompts``      : fixed text prompt

into an LMDB consumed by Stages 1/2/3.  The DWPose mask is generated
automatically (``pose/extract.process_dwpose``) when not supplied.

The teacher is loaded from the sibling ``dreamidv_wan_faster`` package.  Because
the forcing_baseline VAE wrapper and the teacher's ``WanVAE`` apply the *same*
Wan2.1 latent normalisation (``scale=[mean, 1/std]``), the stored latents live in
the exact space the causal student is trained/inferred in.

Example
-------
    python -m tools.syncid_generate_data \
        --dreamidv_root /path/to/DreamID-V \
        --ckpt_dir /path/to/Wan2.1-ckpts \
        --dreamidv_ckpt /path/to/dreamidv_faster.pth \
        --manifest corpus.jsonl --output_lmdb data/swap_latents \
        --size 832*480 --frame_num 81 --sampling_steps 12
"""
import argparse
import json
import math
import os
import random
import sys

import torch
from tqdm import tqdm

# forcing_baseline package root on the path (for utils.*)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.swap_data import LMDBWriter  # noqa: E402


def _load_teacher(args):
    """Instantiate the original bidirectional DreamID-V-Faster teacher pipeline."""
    sys.path.insert(0, args.dreamidv_root)
    sys.path.insert(0, os.path.join(args.dreamidv_root, "pose"))
    import dreamidv_wan_faster
    from dreamidv_wan_faster.configs import WAN_CONFIGS

    cfg = WAN_CONFIGS[args.task]
    teacher = dreamidv_wan_faster.DreamIDV(
        config=cfg, checkpoint_dir=args.ckpt_dir, dreamidv_ckpt=args.dreamidv_ckpt,
        device_id=args.device_id, rank=0, t5_fsdp=False, dit_fsdp=False,
        use_usp=False, t5_cpu=False)
    return teacher, cfg


def _maybe_make_mask(args, ref_video_path, given_mask):
    if given_mask and os.path.exists(given_mask):
        return given_mask
    temp_dir = os.path.join(os.path.dirname(ref_video_path), "temp_generated")
    base = os.path.basename(ref_video_path).split(".")[0]
    pose_path = os.path.join(temp_dir, base + "_pose.mp4")
    mask_path = os.path.join(temp_dir, base + "_mask.mp4")
    if not os.path.exists(mask_path):
        os.makedirs(temp_dir, exist_ok=True)
        from pose.extract import process_dwpose
        process_dwpose(ref_video_path, pose_path, mask_path)
    return mask_path


@torch.no_grad()
def _teacher_swap_latent(teacher, cfg, paths, size, frame_num, sampling_steps,
                         shift, guide_scale_img, seed, context_path):
    """Replicate DreamIDV.generate but return the *latent* tensors.

    Returns (clean_latent [F,16,h,w], y [32,F,h,w], img_ref [16,1,h,w]).
    """
    from dreamidv_wan_faster.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    device = teacher.device
    dtype = teacher.param_dtype

    latents_ref = teacher.load_image_latent_ref_ip_video(paths, size, device, frame_num)
    y_i_v = latents_ref["video"].to(device, dtype)      # [16, F, h, w]
    msk = latents_ref["mask"].to(device, dtype)         # [16, F, h, w]
    img_ref = latents_ref["image"].to(device, dtype)    # [16, 1, h, w]

    z_dim = teacher.vae.model.z_dim
    target_shape = (z_dim, y_i_v.shape[1], y_i_v.shape[2], y_i_v.shape[3])
    seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (teacher.patch_size[1] * teacher.patch_size[2]) * target_shape[1])

    # fixed text embedding ("change face"), identical to the original repo
    context = [t.to(device) for t in torch.load(context_path)]

    y = torch.cat([y_i_v, msk], dim=0)                  # [32, F, h, w]
    arg_pos = {"context": context, "seq_len": seq_len, "y": [y], "img_ref": [img_ref]}
    arg_neg = {"context": context, "seq_len": seq_len, "y": [y], "img_ref": [torch.zeros_like(img_ref)]}

    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    noise = torch.randn(*target_shape, dtype=torch.float32, device=device, generator=g)
    latents = [noise]

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=teacher.num_train_timesteps, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(sampling_steps, device=device, shift=shift)

    with torch.cuda.amp.autocast(dtype=dtype):
        for t in tqdm(scheduler.timesteps, leave=False):
            timestep = torch.stack([t])
            pos = teacher.model(latents, t=timestep, **arg_pos)[0]
            neg = teacher.model(latents, t=timestep, **arg_neg)[0]
            noise_pred = pos + guide_scale_img * (pos - neg)
            temp_x0 = scheduler.step(noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0),
                                     return_dict=False, generator=g)[0]
            latents = [temp_x0.squeeze(0)]

    clean = latents[0]                                   # [16, F, h, w]
    clean_latent = clean.permute(1, 0, 2, 3).contiguous()  # [F, 16, h, w]
    return clean_latent.float().cpu(), y.float().cpu(), img_ref.float().cpu()


def main():
    args = _parse_args()
    teacher, cfg = _load_teacher(args)
    from dreamidv_wan_faster.configs import SIZE_CONFIGS
    size = SIZE_CONFIGS[args.size]

    context_path = args.context_path or os.path.join(
        args.dreamidv_root, "dreamidv_wan_faster", "context.pth")

    with open(args.manifest, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if args.max_samples > 0:
        items = items[: args.max_samples]
    print(f"[stage0] loaded {len(items)} groups from {args.manifest}")
    if not items:
        raise SystemExit(
            f"[stage0] manifest {args.manifest} is empty -- run build_manifest first "
            "(check INPUT_DIR and the _ref.jpg/_mask.mp4 suffixes).")

    writer = LMDBWriter(args.output_lmdb)
    # The LMDB stores one global shape for every row, so all kept samples must
    # share the same latent geometry [F, 16, h, w].  The first accepted sample
    # fixes it; any clip that resolves to a different shape (too few frames /
    # odd resolution) is skipped so the dataset stays loadable.
    ref_shape = None
    accepted = 0
    for item in tqdm(items, desc="Stage-0 self-distill"):
        ref_video = item["ref_video"]
        ref_image = item["ref_image"]
        prompt = item.get("prompt", args.prompt)
        try:
            mask_path = _maybe_make_mask(args, ref_video, item.get("mask"))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] DWPose failed for {ref_video}: {e}")
            continue
        paths = [ref_video, mask_path, ref_image]
        try:
            clean_latent, y, img_ref = _teacher_swap_latent(
                teacher, cfg, paths, size, args.frame_num, args.sampling_steps,
                args.sample_shift, args.guide_scale_img, args.base_seed, context_path)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] teacher inference failed for {ref_video}: {e}")
            continue

        shape = tuple(clean_latent.shape)
        if ref_shape is None:
            ref_shape = shape
            print(f"[stage0] latent geometry fixed to {ref_shape} (F, 16, h, w)")
        elif shape != ref_shape:
            print(f"[skip] latent shape {shape} != {ref_shape} for {ref_video}")
            continue

        writer.add({"clean_latent": clean_latent, "y": y, "img_ref": img_ref, "prompts": prompt})
        accepted += 1
    writer.close()
    print(f"[stage0] accepted {accepted}/{len(items)} samples")
    if accepted == 0:
        raise SystemExit(
            f"[stage0] no samples written to {args.output_lmdb} -- every item was "
            "skipped. Inspect the [skip] lines above (teacher inference / paths / "
            "latent shape). The LMDB is unusable for training until this is fixed.")


def _parse_args():
    p = argparse.ArgumentParser(description="Stage-0 SyncID-Pipe self-distillation data generation")
    p.add_argument("--dreamidv_root", required=True, help="Path to the DreamID-V repo root")
    p.add_argument("--ckpt_dir", required=True, help="Wan2.1 checkpoint dir (VAE/T5)")
    p.add_argument("--dreamidv_ckpt", required=True, help="dreamidv_faster.pth DiT checkpoint")
    p.add_argument("--manifest", required=True, help="JSONL with {ref_video, ref_image, [mask], [prompt]}")
    p.add_argument("--output_lmdb", required=True, help="Output LMDB directory")
    p.add_argument("--task", default="swapface")
    p.add_argument("--size", default="832*480")
    p.add_argument("--frame_num", type=int, default=81)
    p.add_argument("--sampling_steps", type=int, default=12)
    p.add_argument("--sample_shift", type=float, default=5.0)
    p.add_argument("--guide_scale_img", type=float, default=4.0)
    p.add_argument("--prompt", default="change face")
    p.add_argument("--context_path", default=None,
                   help="Path to context.pth (fixed text embedding); defaults to <root>/dreamidv_wan_faster/context.pth")
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=-1)
    return p.parse_args()


if __name__ == "__main__":
    main()
