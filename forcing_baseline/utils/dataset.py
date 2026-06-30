"""Datasets for the causal DreamID-V distillation pipeline.

``SwapLatentLMDBDataset`` reads the Stage-0 self-distilled corpus.  Each row holds:
  * ``clean_latent`` : teacher swapped-video latent   [F, 16, h, w]
  * ``y``            : source-video latent (16) + mask latent (16) channel-concat
                       conditioning                    [32, F, h, w]
  * ``img_ref``      : reference-face latent           [16, 1, h, w]
  * ``prompts``      : (fixed) text prompt string
"""
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb


def cycle(dl):
    while True:
        for data in dl:
            yield data


class TextDataset(Dataset):
    """Plain prompt list (used by Stage-3 DMD, which conditions only on y/img_ref)."""

    def __init__(self, prompt_path):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        return {"prompts": self.prompt_list[idx], "idx": idx}


class SwapLatentLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True, lock=False, readahead=False, meminit=False)
        try:
            self.clean_shape = get_array_shape_from_lmdb(self.env, "clean_latent")
            self.y_shape = get_array_shape_from_lmdb(self.env, "y")
            self.ref_shape = get_array_shape_from_lmdb(self.env, "img_ref")
        except AttributeError as e:
            raise RuntimeError(
                f"LMDB at '{data_path}' has no 'clean_latent' shape metadata -- it is "
                "empty or was not finalized. Re-run Stage-0 (scripts/stage0_gen_data_2h200.sh) "
                "and confirm the '[stage0] accepted N/...' line reports N > 0."
            ) from e
        self.max_pair = max_pair

    def __len__(self):
        return min(self.clean_shape[0], self.max_pair)

    def __getitem__(self, idx):
        clean_latent = retrieve_row_from_lmdb(
            self.env, "clean_latent", np.float16, idx, shape=self.clean_shape[1:])
        y = retrieve_row_from_lmdb(
            self.env, "y", np.float16, idx, shape=self.y_shape[1:])
        img_ref = retrieve_row_from_lmdb(
            self.env, "img_ref", np.float16, idx, shape=self.ref_shape[1:])
        prompts = retrieve_row_from_lmdb(self.env, "prompts", str, idx)
        return {
            "prompts": prompts,
            "clean_latent": torch.tensor(np.ascontiguousarray(clean_latent), dtype=torch.float32),
            "y": torch.tensor(np.ascontiguousarray(y), dtype=torch.float32),
            "img_ref": torch.tensor(np.ascontiguousarray(img_ref), dtype=torch.float32),
        }
