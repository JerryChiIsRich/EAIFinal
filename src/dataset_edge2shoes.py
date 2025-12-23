import os
from glob import glob
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def pil_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), resample=Image.BICUBIC)


def to_tensor_01(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(arr)


class Edge2ShoesDataset(Dataset):
    """
    Supports two formats:

    (1) Folder split:
        root/train/edges/*.png (or jpg)
        root/train/shoes/*.png
        root/val/edges/...
        root/val/shoes/...

    (2) Pix2pix concat format:
        root/train/*.jpg  where image is [edge | shoe] concatenated along width
        root/val/*.jpg

    Returns dict:
      {
        "edge":   float tensor [3,H,W] in [0,1]
        "target": float tensor [3,H,W] in [0,1]
        "name": filename
      }
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        size: int = 512,
        concat_format: bool = False,
        edges_subdir: str = "edges",
        target_subdir: str = "shoes",
        exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
    ):
        self.root = root
        self.split = split
        self.size = size
        self.concat_format = concat_format
        self.exts = exts

        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split folder not found: {split_dir}")

        if concat_format:
            paths = []
            for e in exts:
                paths += glob(os.path.join(split_dir, f"*{e}"))
            self.concat_paths = sorted(paths)
            if len(self.concat_paths) == 0:
                raise FileNotFoundError(f"No concat images found in: {split_dir}")
            self.edge_paths = None
            self.tgt_paths = None
        else:
            edge_dir = os.path.join(split_dir, edges_subdir)
            tgt_dir = os.path.join(split_dir, target_subdir)
            if not os.path.isdir(edge_dir) or not os.path.isdir(tgt_dir):
                raise FileNotFoundError(
                    f"Folder format expects {edge_dir} and {tgt_dir} to exist. "
                    f"Or use --concat_format for pix2pix concat images."
                )
            edge_paths = []
            for e in exts:
                edge_paths += glob(os.path.join(edge_dir, f"*{e}"))
            self.edge_paths = sorted(edge_paths)
            self.tgt_paths = [os.path.join(tgt_dir, os.path.basename(p)) for p in self.edge_paths]
            missing = [p for p in self.tgt_paths if not os.path.exists(p)]
            if missing:
                raise FileNotFoundError(f"Missing targets for {len(missing)} samples. Example: {missing[0]}")

            self.concat_paths = None

    def __len__(self) -> int:
        if self.concat_format:
            return len(self.concat_paths)
        return len(self.edge_paths)

    def _load_pair_concat(self, path: str) -> Tuple[Image.Image, Image.Image]:
        img = pil_rgb(path)
        w, h = img.size
        if w < 2:
            raise ValueError(f"Concat image width too small: {path}")
        w2 = w // 2
        left = img.crop((0, 0, w2, h))      # edge
        right = img.crop((w2, 0, 2 * w2, h))  # target shoe
        return left, right

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.concat_format:
            path = self.concat_paths[idx]
            edge_img, tgt_img = self._load_pair_concat(path)
            name = os.path.basename(path)
        else:
            edge_path = self.edge_paths[idx]
            tgt_path = self.tgt_paths[idx]
            edge_img = pil_rgb(edge_path)
            tgt_img = pil_rgb(tgt_path)
            name = os.path.basename(edge_path)

        edge_img = resize(edge_img, self.size)
        tgt_img = resize(tgt_img, self.size)

        edge = to_tensor_01(edge_img)    # [0,1]
        target = to_tensor_01(tgt_img)   # [0,1]

        return {"edge": edge, "target": target, "name": name}
