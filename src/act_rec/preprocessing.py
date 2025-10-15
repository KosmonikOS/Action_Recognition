from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def valid_crop_resize(
    data_numpy: np.ndarray,
    valid_frame_num: int,
    p_interval: Sequence[float],
    window: int,
) -> np.ndarray:
    """
    Crop a valid temporal window and resize it to ``window`` frames.

    This is a close copy of the routine used in the original InfoGCN++
    preprocessing (`feeders/tools.py`). The input is expected to be of shape
    ``(C, T, V, M)`` where ``C`` is the number of channels (e.g. x, y, score),
    ``T`` the number of frames, ``V`` the number of joints, and ``M`` the number
    of persons. The result has the same layout with the temporal dimension
    resampled to ``window``.
    """
    if data_numpy.ndim != 4:
        raise ValueError(f"Expected data with 4 dimensions (C, T, V, M); got shape {data_numpy.shape}")

    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num if valid_frame_num > 0 else T
    valid_size = end - begin
    if valid_size <= 0:
        raise ValueError("valid_frame_num must be positive or detectable from data.")

    p_vals = tuple(p_interval) if isinstance(p_interval, Sequence) else (float(p_interval),)
    if len(p_vals) == 0:
        raise ValueError("p_interval must contain at least one value.")

    if len(p_vals) == 1:
        p = float(p_vals[0])
        bias = int((1 - p) * valid_size / 2)
        start = max(begin + bias, 0)
        stop = min(end - bias, T)
        data = data_numpy[:, start:stop, :, :]
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
        data = data_numpy[:, start:stop, :, :]

    if data.shape[1] == 0:
        raise ValueError("Cropping resulted in an empty sequence. Check p_interval or input data.")

    cropped_length = data.shape[1]
    data_tensor = torch.tensor(data, dtype=torch.float32)
    data_tensor = data_tensor.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data_tensor = data_tensor.unsqueeze(0).unsqueeze(0)
    data_tensor = F.interpolate(
        data_tensor,
        size=(C * V * M, window),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    data_tensor = data_tensor.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous()
    return data_tensor.numpy()


def _rot(rot: torch.Tensor) -> torch.Tensor:
    """
    Build 3D rotation matrices for each frame.

    Parameters
    ----------
    rot: torch.Tensor
        Tensor of shape ``(T, 3)`` containing rotation angles (rx, ry, rz) per frame.
    """
    cos_r, sin_r = rot.cos(), rot.sin()
    zeros = torch.zeros(rot.shape[0], 1, device=rot.device)
    ones = torch.ones(rot.shape[0], 1, device=rot.device)

    r1 = torch.stack((ones, zeros, zeros), dim=-1)
    rx2 = torch.stack((zeros, cos_r[:, 0:1], sin_r[:, 0:1]), dim=-1)
    rx3 = torch.stack((zeros, -sin_r[:, 0:1], cos_r[:, 0:1]), dim=-1)
    rx = torch.cat((r1, rx2, rx3), dim=1)

    ry1 = torch.stack((cos_r[:, 1:2], zeros, -sin_r[:, 1:2]), dim=-1)
    r2 = torch.stack((zeros, ones, zeros), dim=-1)
    ry3 = torch.stack((sin_r[:, 1:2], zeros, cos_r[:, 1:2]), dim=-1)
    ry = torch.cat((ry1, r2, ry3), dim=1)

    rz1 = torch.stack((cos_r[:, 2:3], sin_r[:, 2:3], zeros), dim=-1)
    r3 = torch.stack((zeros, zeros, ones), dim=-1)
    rz2 = torch.stack((-sin_r[:, 2:3], cos_r[:, 2:3], zeros), dim=-1)
    rz = torch.cat((rz1, rz2, r3), dim=1)

    rot = rz.matmul(ry).matmul(rx)
    return rot


def random_rot(data_numpy: np.ndarray, theta: float = 0.3) -> np.ndarray:
    """
    Apply a random 3D rotation to the sequence, following InfoGCN++ augments.

    Parameters
    ----------
    data_numpy: np.ndarray
        Array of shape ``(C, T, V, M)``.
    theta: float
        Maximum absolute rotation angle (radians) sampled per axis.
    """
    data_torch = torch.from_numpy(data_numpy).float()
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V * M)
    rot = torch.zeros(3).uniform_(-theta, theta)
    rot = rot.repeat(T, 1)
    rot_mats = _rot(rot)
    data_torch = torch.matmul(rot_mats, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()
    return data_torch.numpy()


def preprocess_sequence(
    sequence: np.ndarray,
    *,
    window_size: int = 64,
    p_interval: Sequence[float] = (1.0,),
    random_rotation: bool = False,
    use_velocity: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess a single skeleton sequence stored as ``(T, V, C)`` numpy array.

    Parameters
    ----------
    sequence: np.ndarray
        Raw skeleton with ``T`` frames, ``V`` joints and ``C`` channels (e.g. x, y, score).
    window_size: int
        Target number of frames after temporal interpolation/cropping.
    p_interval: Sequence[float]
        Temporal sampling interval used for random cropping (same semantics as InfoGCN++).
    random_rotation: bool
        Whether to apply random 3D rotation augmentation.
    use_velocity: bool
        If ``True``, convert coordinates to frame-to-frame motion (velocity) and zero the last frame.

    Returns
    -------
    data_tensor: torch.Tensor
        Float tensor with shape ``(C, window_size, V, 1)`` ready for InfoGCN++ style models.
    mask_tensor: torch.Tensor
        Boolean tensor with shape ``(1, window_size, 1, 1)`` marking valid frames.
    """
    if sequence.ndim != 3:
        raise ValueError(f"Expected sequence with shape (T, V, C); got {sequence.shape}")

    sequence = sequence.astype(np.float32, copy=False)
    T, V, C = sequence.shape

    data = np.transpose(sequence, (2, 0, 1))  # (C, T, V)
    data = data[..., np.newaxis]  # (C, T, V, 1)

    valid_frame = data.sum(axis=0, keepdims=True).sum(axis=2, keepdims=True)
    valid_frame_num = int(np.sum(np.squeeze(valid_frame).sum(-1) != 0))
    if valid_frame_num == 0:
        valid_frame_num = T

    data = valid_crop_resize(data, valid_frame_num, p_interval, window_size)
    if random_rotation:
        data = random_rot(data)
    if use_velocity:
        data[:, :-1] = data[:, 1:] - data[:, :-1]
        data[:, -1] = 0.0

    mask = np.abs(data).sum(axis=0, keepdims=True).sum(axis=2, keepdims=True) > 0

    data_tensor = torch.from_numpy(data).float()
    mask_tensor = torch.from_numpy(mask).bool()

    return data_tensor, mask_tensor
