"""Merge per-GPU Stage-0 LMDB shards into the single LMDB the trainer reads.

Each shard was produced by ``tools.syncid_generate_data`` with ``--num_shards``/
``--shard_id`` and shares the exact storage layout (``{name}_{i}_data`` rows plus a
``{name}_shape`` header written on close). This tool streams every shard's rows
into one output LMDB, re-indexing them contiguously, so the merged result is
identical to a single-GPU run and needs no training-side changes.

Example
-------
    python -m tools.merge_lmdb_shards \
        --shards data/swap_latents_shard0 ... data/swap_latents_shard7 \
        --output data/swap_latents
"""
import argparse
import os
import sys

import lmdb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.swap_data import LMDBWriter  # noqa: E402

_ARRAY_FIELDS = ("clean_latent", "y", "img_ref")


def _shape(txn, name):
    raw = txn.get(f"{name}_shape".encode())
    if raw is None:
        return None
    return tuple(int(x) for x in raw.decode().split())


def _merge_one(shard_path, writer, ref_row_shapes):
    if not os.path.isdir(shard_path):
        print(f"[merge] MISSING shard dir, skipping: {shard_path}")
        return 0, ref_row_shapes
    env = lmdb.open(shard_path, readonly=True, lock=False, readahead=False, meminit=False)
    added = 0
    try:
        # NOTE: never call env.close() while inside this `with` block -- closing an
        # env with a live txn is UB in lmdb and corrupts the next env.begin().
        with env.begin() as txn:
            clean_shape = _shape(txn, "clean_latent")
            if clean_shape is None:
                print(f"[merge] shard has no data (not finalized?), skipping: {shard_path}")
                return 0, ref_row_shapes
            n = clean_shape[0]
            shapes = {name: _shape(txn, name) for name in _ARRAY_FIELDS}
            row_shapes = {name: shp[1:] for name, shp in shapes.items()}

            # lock the per-row geometry to the first non-empty shard; skip mismatches
            # so the merged LMDB stays loadable (single global shape per field).
            if ref_row_shapes is None:
                ref_row_shapes = row_shapes
            elif row_shapes != ref_row_shapes:
                print(f"[merge] shard {shard_path} row shapes {row_shapes} != "
                      f"{ref_row_shapes}; skipping whole shard.")
                return 0, ref_row_shapes

            for i in range(n):
                sample = {}
                for name in _ARRAY_FIELDS:
                    buf = txn.get(f"{name}_{i}_data".encode())
                    arr = np.frombuffer(buf, dtype=np.float16).reshape(row_shapes[name])
                    sample[name] = np.ascontiguousarray(arr)
                praw = txn.get(f"prompts_{i}_data".encode())
                sample["prompts"] = praw.decode() if praw is not None else "change face"
                writer.add(sample)
                added += 1
    finally:
        env.close()
    print(f"[merge] +{added} samples from {shard_path}")
    return added, ref_row_shapes


def main():
    p = argparse.ArgumentParser(description="Merge Stage-0 LMDB shards into one LMDB")
    p.add_argument("--shards", nargs="+", required=True, help="Shard LMDB dirs to merge")
    p.add_argument("--output", required=True, help="Merged output LMDB dir")
    args = p.parse_args()

    writer = LMDBWriter(args.output)
    total, ref_row_shapes = 0, None
    for shard in args.shards:
        added, ref_row_shapes = _merge_one(shard, writer, ref_row_shapes)
        total += added
    writer.close()
    print(f"[merge] DONE: {total} samples -> {args.output}")
    if total == 0:
        raise SystemExit("[merge] no samples merged -- check the shard logs above.")


if __name__ == "__main__":
    main()
