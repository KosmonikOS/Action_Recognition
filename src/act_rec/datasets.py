from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from act_rec.preprocessing import preprocess_sequence


class SkeletonNpyDataset(Dataset):
    """
    Dataset wrapper for ``.npy`` skeleton sequences following the InfoGCN++
    preprocessing pipeline.

    Each entry corresponds to a single source sequence. Temporal crops are drawn
    on-the-fly using the provided ``p_interval`` so repeated epochs expose the
    model to different sub-clips, just like the original feeder implementation.
    """

    def __init__(
        self,
        files: Sequence[str | Path],
        labels: Sequence[int] | None = None,
        *,
        window_size: int = 64,
        p_interval: Sequence[float] = (1.0,),
        random_rotation: bool = True,
        use_velocity: bool = True,
        preload: bool = True,
        preload_to_tensor: bool = True,
        repeat: int = 1,
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
        self.preload_to_tensor = preload_to_tensor
        self.repeat = max(int(repeat), 1)

        if self.preload_to_tensor and not self.preload:
            raise ValueError("preload_to_tensor=True requires preload=True.")

        self._file_meta: list[dict[str, int]] = []
        self._preloaded_sequences: list[np.ndarray | torch.Tensor] | None = None

        self._index_metadata()

    def __len__(self) -> int:
        return len(self.files) * self.repeat

    def _index_metadata(self) -> None:
        self._file_meta.clear()
        preloaded: list[np.ndarray | torch.Tensor] | None = [] if self.preload else None

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

            self._file_meta.append({"valid_len": valid_len})

            if preloaded is not None:
                seq_valid = seq[:valid_len].astype(np.float32, copy=False)
                if self.preload_to_tensor:
                    preloaded.append(torch.tensor(seq_valid, dtype=torch.float32))
                else:
                    preloaded.append(np.array(seq_valid, copy=True))

        if self.preload:
            self._preloaded_sequences = preloaded
        else:
            self._preloaded_sequences = None

    def _load_sequence(self, file_idx: int) -> tuple[np.ndarray | torch.Tensor, int]:
        if self._preloaded_sequences is not None:
            sequence = self._preloaded_sequences[file_idx]
        else:
            sequence = np.load(self.files[file_idx]).astype(np.float32, copy=False)
            valid_len = self._file_meta[file_idx]["valid_len"]
            sequence = sequence[:valid_len]
        valid_len = self._file_meta[file_idx]["valid_len"]
        return sequence, valid_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int | None, torch.Tensor, int]:
        file_idx = idx % len(self.files)
        sequence, valid_len = self._load_sequence(file_idx)

        data_tensor, mask_tensor = preprocess_sequence(
            sequence,
            window_size=self.window_size,
            p_interval=self.p_interval,
            random_rotation=self.random_rotation,
            use_velocity=self.use_velocity,
            valid_frame_num=valid_len,
        )

        label = self.labels[file_idx] if self.labels is not None else None
        return data_tensor.contiguous(), label, mask_tensor.contiguous(), file_idx
