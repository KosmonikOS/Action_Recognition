from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def valid_crop_resize(
    data_numpy: np.ndarray | torch.Tensor,
    valid_frame_num: int,
    p_interval: Sequence[float],
    window: int,
) -> torch.Tensor:
    """
    Crop a valid temporal window and resize it to ``window`` frames.

    This is a close copy of the routine used in the original InfoGCN++
    preprocessing (`feeders/tools.py`). The input is expected to be of shape
    ``(C, T, V, M)`` where ``C`` is the number of channels (e.g. x, y, score),
    ``T`` the number of frames, ``V`` the number of joints, and ``M`` the number
    of persons. The result has the same layout with the temporal dimension
    resampled to ``window``.
    """
    data_tensor = torch.as_tensor(data_numpy, dtype=torch.float32)
    if data_tensor.dim() != 4:
        raise ValueError(f"Expected data with 4 dimensions (C, T, V, M); got shape {tuple(data_tensor.shape)}")

    C, T, V, M = data_tensor.shape
    begin = 0
    end = valid_frame_num if valid_frame_num > 0 else T
    valid_size = end - begin
    if valid_size <= 0:
        raise ValueError("valid_frame_num must be positive or detectable from data.")

    p_vals = tuple(p_interval) if isinstance(p_interval, Sequence) else (float(p_interval[0]),)
    if len(p_vals) == 0:
        raise ValueError("p_interval must contain at least one value.")

    if len(p_vals) == 1:
        p = float(p_vals[0])
        bias = int((1 - p) * valid_size / 2)
        start = max(begin + bias, 0)
        stop = min(end - bias, T)
        data = data_tensor[:, start:stop, :, :]
    else:
        low, high = float(p_vals[0]), float(p_vals[1])
        if low > high:
            low, high = high, low
        p = np.random.rand() * (high - low) + low
        cropped_length = int(np.floor(valid_size * p))
        cropped_length = int(np.minimum(np.maximum(cropped_length, 64), valid_size))
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        start = begin + bias
        stop = start + cropped_length
        data = data_tensor[:, start:stop, :, :]

    if data.size(1) == 0:
        raise ValueError("Cropping resulted in an empty sequence. Check p_interval or input data.")

    cropped_length = data.size(1)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data.unsqueeze(0).unsqueeze(0)
    data = F.interpolate(
        data,
        size=(C * V * M, window),
        mode="bilinear",
        align_corners=False,
    )
    data = data[0, 0]
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous()
    return data


def random_rot(data_numpy: np.ndarray | torch.Tensor, theta: float = 0.3) -> torch.Tensor:
    """
    Apply a random in-plane rotation to the sequence (XY joints only).

    Parameters
    ----------
    data_numpy: np.ndarray
        Array of shape ``(C, T, V, M)``.
    theta: float
        Maximum absolute rotation angle (radians) sampled per axis.
    """
    data_torch = torch.as_tensor(data_numpy, dtype=torch.float32).clone()
    if data_torch.size(0) < 2:
        return data_torch

    angle = float(torch.empty(1).uniform_(-theta, theta))
    cos_theta, sin_theta = math.cos(angle), math.sin(angle)

    rot = torch.tensor(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
        dtype=data_torch.dtype,
    )

    coords = data_torch[:2].reshape(2, -1)
    coords = rot @ coords
    data_torch[:2] = coords.view_as(data_torch[:2])
    return data_torch


def preprocess_sequence(
    sequence: np.ndarray | torch.Tensor,
    *,
    window_size: int = 64,
    p_interval: Sequence[float] = (1.0,),
    random_rotation: bool = False,
    use_velocity: bool = False,
    valid_frame_num: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess a single skeleton sequence stored as ``(T, V, C)`` numpy array.

    The routine mirrors the InfoGCN++ feeder: a temporal crop is chosen according
    to ``p_interval``, resized to ``window_size`` frames, then optional spatial
    augmentations (random rotation) and temporal differencing (velocity) are
    applied. The output tensor has shape ``(C, window_size, V, 1)`` – matching the
    expected input layout of the SODE backbone.
    """
    sequence_tensor = torch.as_tensor(sequence, dtype=torch.float32)
    if sequence_tensor.dim() != 3:
        raise ValueError(f"Expected sequence with shape (T, V, C); got {tuple(sequence_tensor.shape)}")

    T, V, C = sequence_tensor.shape
    if T == 0:
        raise ValueError("Sequence must contain at least one frame.")

    data = sequence_tensor.permute(2, 0, 1).unsqueeze(-1)  # (C, T, V, 1)

    if valid_frame_num is not None:
        valid_frame_num = int(valid_frame_num)
        if valid_frame_num <= 0:
            raise ValueError("valid_frame_num must be positive.")
        valid_frame_num = min(valid_frame_num, data.size(1))
    else:
        frame_activity = data.abs().sum(dim=(0, 2, 3)) > 0  # (T,)
        if frame_activity.any():
            valid_indices = torch.nonzero(frame_activity, as_tuple=False)
            valid_frame_num = int(valid_indices[-1].item() + 1)
        else:
            valid_frame_num = T
        valid_frame_num = max(valid_frame_num, 1)

    data = data[:, :valid_frame_num, :, :]

    clip = valid_crop_resize(data, valid_frame_num, p_interval, window_size)
    mask = clip.abs().sum(dim=0, keepdim=True).sum(dim=2, keepdim=True) > 0
    if random_rotation:
        clip = random_rot(clip)
    if use_velocity:
        clip[:, :-1] = clip[:, 1:] - clip[:, :-1]
        clip[:, -1] = 0.0

    data_tensor = clip.float()
    mask_tensor = mask.to(torch.bool)
    return data_tensor, mask_tensor
