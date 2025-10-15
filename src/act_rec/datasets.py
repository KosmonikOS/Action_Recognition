from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from act_rec.preprocessing import preprocess_sequence


class SkeletonNpyDataset(Dataset):
    """
    Simple dataset to wrap raw ``.npy`` skeleton clips for InfoGCN++ style models.

    Each sample is loaded on-the-fly, preprocessed to match the ``(C, T, V, M)``
    layout, and accompanied by a per-frame validity mask.
    """

    def __init__(
        self,
        files: Sequence[str | Path],
        labels: Sequence[int] | None = None,
        *,
        window_size: int = 64,
        p_interval: Sequence[float] = (1.0,),
        random_rotation: bool = False,
        use_velocity: bool = False,
    ) -> None:
        self.files = [Path(f) for f in files]
        if labels is not None and len(labels) != len(self.files):
            raise ValueError("labels must have the same length as files.")
        self.labels = list(labels) if labels is not None else None
        self.window_size = window_size
        self.p_interval = tuple(p_interval)
        self.random_rotation = random_rotation
        self.use_velocity = use_velocity

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int | None, torch.Tensor, int]:
        sequence = np.load(self.files[idx])
        data_tensor, mask_tensor = preprocess_sequence(
            sequence,
            window_size=self.window_size,
            p_interval=self.p_interval,
            random_rotation=self.random_rotation,
            use_velocity=self.use_velocity,
        )
        label = self.labels[idx] if self.labels is not None else None
        return data_tensor, label, mask_tensor, idx
