"""Streaming face-swap inference with the distilled causal generator.

Pipeline:
  1. DWPose mask is generated from the driving video (``pose.extract.process_dwpose``).
  2. Driving video / mask / reference image are resized & VAE-encoded into the
     DreamID-V conditioning (``y`` = source+mask latent, ``img_ref`` = ref-face latent).
  3. The few-step causal generator rolls the swapped video out chunk by chunk
     (reference-face sink at cache frame 0; KV-cached past frames).
  4. The latent is VAE-decoded and saved to an mp4.

The preprocessing reuses the original DreamID-V transforms so the conditioning is
identical to the teacher's; the VAE wrapper shares the Wan2.1 latent normalisation.
"""
import argparse
import math
import os
import sys

import torch
from omegaconf import OmegaConf


def _add_dreamidv_paths(root):
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "pose"))


@torch.no_grad()
def _encode_clip(vae, frames, transform, device, dtype):
    """frames: list[PIL] -> latent [16, F, h, w] using the shared VAE wrapper."""
    pix = transform(frames).to(device=device, dtype=dtype)        # [C, F, H, W]
    latent = vae.encode_to_latent(pix.unsqueeze(0))               # [1, F, 16, h, w]
    return latent[0].permute(1, 0, 2, 3).contiguous()             # [16, F, h, w]


@torch.no_grad()
def _preprocess(args, vae, device, dtype):
    from decord import VideoReader
    from PIL import Image, ImageOps
    from torchvision.transforms import Compose, Normalize
    from dreamidv_wan_faster.utils.na_resize import NaResize, DivisibleCrop, Rearrange
    from pose.extract import process_dwpose

    size = tuple(int(x) for x in args.size.split("*"))            # (W, H)
    vae_stride, patch = (4, 8, 8), (1, 2, 2)
    res = math.sqrt(size[0] * size[1])
    crop = (vae_stride[1] * patch[1], vae_stride[2] * patch[2])
    video_tf = Compose([NaResize(res, downsample_only=True), DivisibleCrop(crop),
                        Normalize(0.5, 0.5), Rearrange("t c h w -> c t h w")])
    mask_tf = Compose([NaResize(res, downsample_only=True), DivisibleCrop(crop),
                       Rearrange("t c h w -> c t h w")])

    # DWPose mask
    temp_dir = os.path.join(os.path.dirname(args.ref_video), "temp_generated")
    base = os.path.basename(args.ref_video).split(".")[0]
    pose_path = os.path.join(temp_dir, base + "_pose.mp4")
    mask_path = args.mask or os.path.join(temp_dir, base + "_mask.mp4")
    if not os.path.exists(mask_path):
        os.makedirs(temp_dir, exist_ok=True)
        process_dwpose(args.ref_video, pose_path, mask_path)

    def _read(path, n):
        vr = VideoReader(path)
        frames = [Image.fromarray(vr[i].asnumpy()) for i in range(len(vr))][:n]
        n2 = (len(frames) - 1) // 4 * 4 + 1
        return frames[:n2]

    vframes = _read(args.ref_video, args.frame_num)
    vw, vh = vframes[0].size
    mframes = _read(mask_path, args.frame_num)

    video_lat = _encode_clip(vae, vframes, video_tf, device, dtype)   # [16, F, h, w]
    mask_lat = _encode_clip(vae, mframes, mask_tf, device, dtype)     # [16, F, h, w]

    with Image.open(args.ref_image) as img:
        img = img.convert("RGB")
        ratio, target = img.width / img.height, vw / vh
        if ratio > target:
            nw, nh = vw, int(vw / ratio)
        else:
            nh, nw = vh, int(vh * ratio)
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        dw, dh = vw - img.size[0], vh - img.size[1]
        img = ImageOps.expand(img, (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2), fill=(255, 255, 255))
    img_tf = Compose([NaResize(res, downsample_only=True), DivisibleCrop(crop),
                      Normalize(0.5, 0.5), Rearrange("t c h w -> c t h w")])
    img_ref = _encode_clip(vae, [img], img_tf, device, dtype)         # [16, 1, h, w]

    y = torch.cat([video_lat, mask_lat], dim=0).unsqueeze(0)          # [1, 32, F, h, w]
    img_ref = img_ref.unsqueeze(0)                                    # [1, 16, 1, h, w]
    return y, img_ref


@torch.no_grad()
def main():
    args = _parse_args()
    _add_dreamidv_paths(args.dreamidv_root)

    here = os.path.dirname(os.path.abspath(__file__))
    config = OmegaConf.merge(
        OmegaConf.load(os.path.join(here, "configs", "default_config.yaml")),
        OmegaConf.load(args.config_path))

    device = torch.device("cuda")
    dtype = torch.bfloat16 if config.mixed_precision else torch.float32

    from utils.dreamidv_wrapper import (
        DreamIDVDiffusionWrapper, DreamIDVTextEncoder, DreamIDVVAEWrapper)
    from pipeline import CausalInferencePipeline

    generator = DreamIDVDiffusionWrapper(
        model_config=OmegaConf.to_container(config.model_kwargs, resolve=True),
        timestep_shift=config.timestep_shift, is_causal=True,
        local_attn_size=config.local_attn_size, sink_size=config.sink_size,
        num_max_frames=config.num_training_frames)
    if getattr(config, "generator_ckpt", False):
        generator.load_checkpoint(config.generator_ckpt, strict=False)
    generator.model.to(device=device, dtype=dtype).eval()

    vae = DreamIDVVAEWrapper(config.vae_checkpoint)
    vae.model.to(device=device, dtype=dtype).eval()

    y, img_ref = _preprocess(args, vae, device, dtype)
    num_frames = y.shape[2]
    assert num_frames % config.num_frame_per_block == 0, \
        f"latent frames {num_frames} must be divisible by num_frame_per_block"

    # text embedding (fixed "change face"); optionally load the teacher's context.pth
    if args.context_path and os.path.exists(args.context_path):
        ctx = torch.load(args.context_path)
        prompt_embeds = torch.stack([t.to(device=device, dtype=dtype) for t in ctx])
    else:
        text_encoder = DreamIDVTextEncoder(config.t5_checkpoint, config.t5_tokenizer)
        text_encoder.text_encoder.to(device)
        prompt_embeds = text_encoder([args.prompt])["prompt_embeds"].to(dtype)

    conditional_dict = {"prompt_embeds": prompt_embeds, "y": y, "img_ref": img_ref}

    pipeline = CausalInferencePipeline(
        config, device=device, generator=generator,
        denoising_step_list=list(config.denoising_step_list))
    latent = pipeline.inference(
        conditional_dict, num_frames=num_frames,
        height_lat=y.shape[-2], width_lat=y.shape[-1],
        num_channels=config.num_channels, dtype=dtype)               # [1, F, 16, h, w]

    pixels = vae.decode_to_pixel(latent.to(dtype))                   # [1, F, 3, H, W]
    video = pixels[0].permute(1, 0, 2, 3)                            # [3, F, H, W]

    from dreamidv_wan_faster.utils.utils import cache_video
    os.makedirs(os.path.dirname(os.path.abspath(args.save_file)), exist_ok=True)
    cache_video(tensor=video[None], save_file=args.save_file, fps=args.fps,
                nrow=1, normalize=True, value_range=(-1, 1))
    print(f"Saved swapped video -> {args.save_file}")


def _parse_args():
    p = argparse.ArgumentParser(description="Streaming face-swap inference")
    p.add_argument("--config_path", default="configs/inference.yaml")
    p.add_argument("--dreamidv_root", required=True, help="DreamID-V repo root (for transforms/DWPose)")
    p.add_argument("--ckpt_dir", default=None, help="(unused placeholder for parity with teacher CLI)")
    p.add_argument("--ref_video", required=True)
    p.add_argument("--ref_image", required=True)
    p.add_argument("--mask", default=None, help="Optional precomputed mask video")
    p.add_argument("--save_file", default="outputs/swapped.mp4")
    p.add_argument("--size", default="832*480")
    p.add_argument("--frame_num", type=int, default=81)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--prompt", default="change face")
    p.add_argument("--context_path", default=None)
    return p.parse_args()


if __name__ == "__main__":
    main()
