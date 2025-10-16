from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from act_rec.preprocessing import preprocess_sequence


class SkeletonNpyDataset(Dataset):
    """
    Minimal dataset wrapper for ``.npy`` skeleton clips.

    Each raw file yields one or more windowed clips. When ``preload`` is enabled
    every clip is fully preprocessed during initialisation and served directly
    from memory.
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
        preload: bool = False,
    ) -> None:
        self.files = [Path(f) for f in files]
        if labels is not None and len(labels) != len(self.files):
            raise ValueError("labels must have the same length as files.")
        self.labels = list(labels) if labels is not None else None
        self.window_size = window_size
        self.p_interval = tuple(p_interval)
        self.random_rotation = random_rotation
        self.use_velocity = use_velocity
        self.preload = preload

        self._clip_index: list[tuple[int, int]] = []
        self._file_meta: list[dict[str, int]] = []
        self._preloaded_clips: list[tuple[torch.Tensor, torch.Tensor]] | None = None

        self._build_index()
        if self.preload:
            self._preload_all_clips()

    def __len__(self) -> int:
        return len(self._clip_index)

    def _build_index(self) -> None:
        self._clip_index.clear()
        self._file_meta.clear()

        for file_idx, path in enumerate(self.files):
            seq = np.load(path)
            if seq.ndim != 3:
                raise ValueError(f"Expected sequence with 3 dims (T, V, C); got shape {seq.shape} from {path}")
            seq = seq.astype(np.float32, copy=False)

            num_frames = seq.shape[0]
            if num_frames == 0:
                raise ValueError(f"Sequence {path} contains no frames.")

            frame_activity = np.abs(seq).sum(axis=2) > 0  # (T, V)
            if frame_activity.any():
                frame_mask = frame_activity.any(axis=1)
                last_valid = int(np.where(frame_mask)[0][-1]) + 1
            else:
                last_valid = num_frames
            valid_len = max(last_valid, 1)

            num_clips = max(1, math.ceil(valid_len / self.window_size))
            self._file_meta.append({"valid_len": valid_len, "num_clips": num_clips})

            for clip_idx in range(num_clips):
                self._clip_index.append((file_idx, clip_idx))

    def _load_sequence(self, file_idx: int) -> np.ndarray:
        seq = np.load(self.files[file_idx]).astype(np.float32, copy=False)
        valid_len = self._file_meta[file_idx]["valid_len"]
        if valid_len < seq.shape[0]:
            return seq[:valid_len]
        return seq

    def _preload_all_clips(self) -> None:
        self._preloaded_clips = [None] * len(self._clip_index)
        offset = 0
        for file_idx, path in enumerate(self.files):
            meta = self._file_meta[file_idx]
            num_clips = meta["num_clips"]
            if num_clips == 0:
                continue

            sequence = np.load(path).astype(np.float32, copy=False)
            valid_len = meta["valid_len"]
            if valid_len < sequence.shape[0]:
                sequence = sequence[:valid_len]

            for local_clip_idx in range(num_clips):
                data_tensor, mask_tensor = preprocess_sequence(
                    sequence,
                    window_size=self.window_size,
                    p_interval=self.p_interval,
                    random_rotation=self.random_rotation,
                    use_velocity=self.use_velocity,
                    clip_index=local_clip_idx,
                    valid_frame_num=valid_len,
                )
                self._preloaded_clips[offset + local_clip_idx] = (
                    data_tensor.contiguous(),
                    mask_tensor.contiguous(),
                )
            offset += num_clips

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int | None, torch.Tensor, int]:
        if self.preload and self._preloaded_clips is not None:
            data_tensor, mask_tensor = self._preloaded_clips[idx]
            file_idx, _ = self._clip_index[idx]
        else:
            file_idx, clip_idx = self._clip_index[idx]
            sequence = self._load_sequence(file_idx)
            meta = self._file_meta[file_idx]
            data_tensor, mask_tensor = preprocess_sequence(
                sequence,
                window_size=self.window_size,
                p_interval=self.p_interval,
                random_rotation=self.random_rotation,
                use_velocity=self.use_velocity,
                clip_index=clip_idx,
                valid_frame_num=meta["valid_len"],
            )

        label = self.labels[file_idx] if self.labels is not None else None
        return data_tensor, label, mask_tensor, idx
