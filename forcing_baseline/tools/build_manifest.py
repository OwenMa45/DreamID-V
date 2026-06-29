"""Build a Stage-0 corpus.jsonl from a LivingSwap-style input directory.

Each group shares a base name and ships three files::

    <base>.mp4        driving / source video (the face to be replaced)
    <base>_mask.mp4   face-region mask video
    <base>_ref.jpg    reference identity face image (the face to swap in)

e.g. ``part_004_0000000_<srchash>_id0_to_<dsthash>_id0{,_mask.mp4,_ref.jpg}``.

Because masks are already provided, Stage-0 needs no DWPose: the ``mask`` field is
filled directly and ``syncid_generate_data.py`` skips pose extraction.

Emits one JSON per *complete* group::

    {"ref_video": ..., "ref_image": ..., "mask": ..., "prompt": "change face"}

Example
-------
    python -m tools.build_manifest \
        --input_dir /path/to/part_004/input \
        --output corpus.jsonl
"""
import argparse
import glob
import json
import os


def build(input_dir, output, prompt, ref_suffix, mask_suffix, video_ext, max_samples):
    refs = sorted(glob.glob(os.path.join(input_dir, "*" + ref_suffix)))
    rows, skipped = [], []
    for ref in refs:
        base = ref[: -len(ref_suffix)]
        video = base + video_ext
        mask = base + mask_suffix
        if os.path.exists(video) and os.path.exists(mask):
            rows.append({"ref_video": video, "ref_image": ref, "mask": mask, "prompt": prompt})
        else:
            skipped.append(base)
    if max_samples > 0:
        rows = rows[:max_samples]

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[build_manifest] wrote {len(rows)} groups -> {output} "
          f"(skipped {len(skipped)} incomplete)")
    for b in skipped[:10]:
        print("  [skip incomplete]", os.path.basename(b))
    return len(rows)


def main():
    p = argparse.ArgumentParser(description="Build Stage-0 corpus.jsonl from a paired input dir")
    p.add_argument("--input_dir", required=True, help="Directory holding <base>{.mp4,_mask.mp4,_ref.jpg}")
    p.add_argument("--output", required=True, help="Output corpus.jsonl path")
    p.add_argument("--prompt", default="change face")
    p.add_argument("--ref_suffix", default="_ref.jpg")
    p.add_argument("--mask_suffix", default="_mask.mp4")
    p.add_argument("--video_ext", default=".mp4")
    p.add_argument("--max_samples", type=int, default=-1)
    args = p.parse_args()
    build(args.input_dir, args.output, args.prompt, args.ref_suffix,
          args.mask_suffix, args.video_ext, args.max_samples)


if __name__ == "__main__":
    main()
