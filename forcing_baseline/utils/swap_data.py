"""VAE-encoding helpers and an incremental LMDB writer for Stage-0 data gen.

The conditioning layout matches the DreamID-V-Faster backbone:
  * ``y``       = [source-video latent (16) ; mask latent (16)]  -> [32, F, h, w]
  * ``img_ref`` = reference-face latent                          -> [16, 1, h, w]
  * ``clean_latent`` = teacher swapped-video latent              -> [F, 16, h, w]
"""
import lmdb
import numpy as np
import torch


@torch.no_grad()
def encode_video_latent(vae, pixels: torch.Tensor) -> torch.Tensor:
    """pixels [C, F, H, W] in [-1, 1] -> latent [F, 16, h, w] (frame-first)."""
    pixels = pixels.unsqueeze(0)                       # [1, C, F, H, W]
    latent = vae.encode_to_latent(pixels)             # [1, F, 16, h, w]
    return latent[0]


@torch.no_grad()
def build_conditioning(vae, source_video: torch.Tensor, mask_video: torch.Tensor,
                       ref_image: torch.Tensor):
    """Return (y [32, F, h, w], img_ref [16, 1, h, w]).

    * source_video / mask_video : [3, F, H, W] in [-1, 1]
    * ref_image                 : [3, 1, H, W] in [-1, 1]
    """
    src = encode_video_latent(vae, source_video).permute(1, 0, 2, 3)   # [16, F, h, w]
    msk = encode_video_latent(vae, mask_video).permute(1, 0, 2, 3)     # [16, F, h, w]
    y = torch.cat([src, msk], dim=0)                                   # [32, F, h, w]
    ref = encode_video_latent(vae, ref_image).permute(1, 0, 2, 3)      # [16, 1, h, w]
    return y, ref


class LMDBWriter:
    """Incremental, multi-array LMDB writer compatible with ``utils.lmdb_``."""

    def __init__(self, path: str, map_size: int = 1 << 42):
        self.env = lmdb.open(path, map_size=map_size)
        self.count = 0
        self.row_shapes = {}

    def add(self, sample: dict):
        with self.env.begin(write=True) as txn:
            for name, val in sample.items():
                key = f"{name}_{self.count}_data".encode()
                if isinstance(val, str):
                    txn.put(key, val.encode())
                else:
                    arr = val.detach().cpu().half().numpy() if isinstance(val, torch.Tensor) else np.asarray(val, dtype=np.float16)
                    arr = np.ascontiguousarray(arr)
                    self.row_shapes[name] = arr.shape
                    txn.put(key, arr.tobytes())
            if "prompts" not in self.row_shapes:
                self.row_shapes["prompts"] = ()
        self.count += 1

    def close(self):
        with self.env.begin(write=True) as txn:
            for name, row_shape in self.row_shapes.items():
                full_shape = (self.count,) + tuple(row_shape)
                txn.put(f"{name}_shape".encode(), " ".join(map(str, full_shape)).encode())
        self.env.sync()
        self.env.close()
        print(f"[LMDBWriter] wrote {self.count} samples to LMDB")
