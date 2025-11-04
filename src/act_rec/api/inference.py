from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch

from act_rec.labeling import YoloPoseVideoLabeler
from act_rec.model.sode import SODE
from act_rec.params import YoloPoseVideoInferenceParams
from act_rec.preprocessing import preprocess_sequence


WINDOW_SIZE = 64
CONF_THRESHOLD = 0.20
MULTI_PERSON_TOLERANCE = 15


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_checkpoint_path() -> Path:
    env_path = os.getenv("SODE_CHECKPOINT_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (_project_root() / "models" / "sode.pt").resolve()


def _resolve_yolo_weights_path() -> Path:
    env_path = os.getenv("YOLO_WEIGHTS_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (_project_root() / "models" / "yolo11x-pose.pt").resolve()


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def get_yolo_device_string() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def get_labeler() -> YoloPoseVideoLabeler:
    params = YoloPoseVideoInferenceParams()
    params.device = get_yolo_device_string()
    weights_path = _resolve_yolo_weights_path()
    if not weights_path.exists():
        raise FileNotFoundError(
            f"YOLO pose weights are missing at {weights_path}. "
            "Update YOLO_WEIGHTS_PATH or place the checkpoint at the default location."
        )
    return YoloPoseVideoLabeler(model_path=str(weights_path), params=params)


@lru_cache(maxsize=1)
def get_sode_bundle() -> tuple[SODE, dict[int, str]]:
    checkpoint_path = _resolve_checkpoint_path()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SODE checkpoint not found at {checkpoint_path}. "
            "Set SODE_CHECKPOINT_PATH to override the default location."
        )

    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint.get("model")
    if state_dict is None:
        raise KeyError("Checkpoint must contain a 'model' state dict.")

    label_to_idx = checkpoint.get("label_to_idx")
    if not label_to_idx:
        raise KeyError("Checkpoint must contain 'label_to_idx' mapping.")

    num_classes = len(label_to_idx)
    model = SODE(
        num_class=num_classes,
        num_point=17,
        num_person=1,
        graph="act_rec.graph.coco.Graph",
        in_channels=3,
        T=WINDOW_SIZE,
        n_step=3,
        num_cls=4,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return model, idx_to_label


def skeleton_from_result(result) -> np.ndarray | None:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return None
    data = getattr(keypoints, "data", None)
    if data is None or data.shape[0] != 1:
        return None
    return data[0].cpu().numpy()


def _interpolate_linear_1d(y: np.ndarray, mask_valid: np.ndarray) -> np.ndarray:
    if mask_valid.all():
        return y
    if (~mask_valid).all():
        return np.zeros_like(y)
    x = np.arange(y.shape[0])
    xp = x[mask_valid]
    fp = y[mask_valid]
    return np.interp(x, xp, fp).astype(np.float32)


def _interpolate_joints_over_time(xy: np.ndarray, conf: np.ndarray, conf_thr: float) -> np.ndarray:
    xy_interp = xy.copy()
    valid = conf >= conf_thr
    T, V, _ = xy.shape
    for v in range(V):
        mask = valid[:, v]
        for c in range(2):
            xy_interp[:, v, c] = _interpolate_linear_1d(xy[:, v, c], mask)
    return xy_interp


def _center_and_scale_sequence(joints_xy: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    left_hip, right_hip = 11, 12
    left_sh, right_sh = 5, 6
    pelvis = (joints_xy[:, left_hip, :] + joints_xy[:, right_hip, :]) / 2.0
    pelvis_valid = np.isfinite(pelvis).all(axis=1)
    if pelvis_valid.any():
        pelvis_ref = pelvis[pelvis_valid][0]
    else:
        pelvis_ref = np.zeros(2, dtype=np.float32)
    spine = (joints_xy[:, left_sh, :] + joints_xy[:, right_sh, :]) / 2.0
    torso_vec = spine - pelvis
    torso_len = np.linalg.norm(torso_vec, axis=1)
    torso_valid = torso_len > eps
    if torso_valid.any():
        scale = float(np.median(torso_len[torso_valid]))
    else:
        scale = 1.0
    scale = max(scale, eps)
    joints_centered = joints_xy - pelvis_ref[None, None, :]
    joints_cs = joints_centered / scale
    return joints_cs.astype(np.float32)


def normalize_skeleton_sequence(sequence: np.ndarray, conf_thr: float) -> np.ndarray:
    seq_np = np.asarray(sequence, dtype=np.float32)
    xy = seq_np[:, :, :2]
    conf = seq_np[:, :, 2]
    xy = _interpolate_joints_over_time(xy, conf, conf_thr)
    joints_cs = _center_and_scale_sequence(xy)
    return np.concatenate([joints_cs, conf[:, :, None]], axis=2).astype(np.float32)


def preprocess_for_sode(window: np.ndarray, window_size: int, conf_thr: float) -> torch.Tensor:
    normalized = normalize_skeleton_sequence(window, conf_thr)
    data_tensor, _ = preprocess_sequence(
        normalized,
        window_size=window_size,
        p_interval=(1.0,),
        random_rotation=False,
        use_velocity=False,
        valid_frame_num=normalized.shape[0],
    )
    return data_tensor


def predict_from_window(model: SODE, window_np: np.ndarray, device: torch.device, window_size: int) -> np.ndarray:
    data_tensor = preprocess_for_sode(window_np, window_size, CONF_THRESHOLD)
    inputs = data_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits, *_ = model(inputs)
        scores = logits[:, :, -1]
        probs = torch.softmax(scores, dim=-1)
    return probs.squeeze(0).cpu().numpy()


def generate_prediction_records(video_path: Path) -> Iterator[dict[str, float | str | int]]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found at {video_path}")

    model, idx_to_label = get_sode_bundle()
    device = get_device()
    labeler = get_labeler()

    buffer: list[np.ndarray] = []
    skip_counter = 0
    start_ts = time.perf_counter()
    frame_count = 0
    window_index = 0
    last_valid_frame_idx = -1

    stream = labeler.model(
        str(video_path),
        stream=True,
        device=labeler.params.device,
        imgsz=labeler.params.imgsz,
        rect=labeler.params.rect,
        batch=labeler.params.batch,
        vid_stride=labeler.params.vid_stride,
        verbose=labeler.params.verbose,
    )

    for frame_idx, result in enumerate(stream):
        skeleton = skeleton_from_result(result)
        if skeleton is None:
            skip_counter += 1
            if skip_counter >= MULTI_PERSON_TOLERANCE:
                buffer.clear()
            continue

        skip_counter = 0
        buffer.append(skeleton)
        frame_count += 1
        last_valid_frame_idx = frame_idx

        if len(buffer) < WINDOW_SIZE:
            continue

        window_np = np.stack(buffer, axis=0)
        buffer.clear()
        probs = predict_from_window(model, window_np, device, WINDOW_SIZE)
        pred_idx = int(probs.argmax())
        pred_label = idx_to_label.get(pred_idx, str(pred_idx))
        pred_conf = float(probs[pred_idx])

        yield {
            "window_index": window_index,
            "frame_idx": frame_idx,
            "elapsed_s": time.perf_counter() - start_ts,
            "prediction": pred_label,
            "confidence": pred_conf,
            "frames_in_window": WINDOW_SIZE,
            "complete_window": True,
        }
        window_index += 1

    if frame_count > 0 and len(buffer) > 0:
        window_np = np.stack(buffer, axis=0)
        probs = predict_from_window(model, window_np, device, WINDOW_SIZE)
        pred_idx = int(probs.argmax())
        pred_label = idx_to_label.get(pred_idx, str(pred_idx))
        pred_conf = float(probs[pred_idx])
        yield {
            "window_index": window_index,
            "frame_idx": last_valid_frame_idx if last_valid_frame_idx >= 0 else frame_count - 1,
            "elapsed_s": time.perf_counter() - start_ts,
            "prediction": pred_label,
            "confidence": pred_conf,
            "frames_in_window": len(window_np),
            "complete_window": len(window_np) == WINDOW_SIZE,
        }


def prediction_ndjson_stream(video_path: Path) -> Iterable[bytes]:
    for record in generate_prediction_records(video_path):
        yield (json.dumps(record) + "\n").encode("utf-8")
