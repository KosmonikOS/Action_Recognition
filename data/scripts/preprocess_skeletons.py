import argparse
import hashlib
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("skeleton_preproc")

def load_and_combine_csvs(csv_paths: Iterable[Path]) -> pd.DataFrame:
    combined = []
    for p in csv_paths:
        logger.info("Loading CSV: %s", p)
        df = pd.read_csv(p)
        df["_base_dir"] = str(p.parent.resolve())
        combined.append(df)
    if not combined:
        raise ValueError("No CSVs provided or all empty")
    out = pd.concat(combined, ignore_index=True)
    logger.info("Combined %d rows from %d CSVs", len(out), len(combined))
    return out


def resolve_skeleton_path(row: pd.Series) -> Path:
    base = Path(row["_base_dir"])
    rel = Path(str(row["skeleton_path"]))
    return (base / rel).resolve()


def safe_load_npy(path: Path) -> np.ndarray | None:
    try:
        return np.load(path, allow_pickle=False)
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def interpolate_linear_1d(y: np.ndarray, mask_valid: np.ndarray) -> np.ndarray:
    n = y.shape[0]
    if mask_valid.all():
        return y
    if (~mask_valid).all():
        return np.zeros_like(y)
    x = np.arange(n)
    xp = x[mask_valid]
    fp = y[mask_valid]
    return np.interp(x, xp, fp)


def interpolate_joints_over_time(xy: np.ndarray, conf: np.ndarray, conf_thr: float) -> np.ndarray:
    T, V, _ = xy.shape
    out = xy.copy()
    valid = conf >= conf_thr
    for v in range(V):
        m = valid[:, v]
        for c in range(2):
            out[:, v, c] = interpolate_linear_1d(xy[:, v, c], m)
    return out


def center_and_scale_sequence(J: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, float]:
    left_hip, right_hip = 11, 12
    left_sh, right_sh = 5, 6
    pelvis = (J[:, left_hip, :] + J[:, right_hip, :]) / 2.0
    pelvis_valid = np.isfinite(pelvis).all(axis=1)
    if not pelvis_valid.any():
        pelvis_ref = np.zeros(2, dtype=np.float32)
    else:
        pelvis_ref = pelvis[pelvis_valid][0]
    spine = (J[:, left_sh, :] + J[:, right_sh, :]) / 2.0
    torso_vec = spine - pelvis
    torso_len = np.linalg.norm(torso_vec, axis=1)
    torso_valid = torso_len > eps
    if torso_valid.any():
        scale = float(np.median(torso_len[torso_valid]))
    else:
        scale = 1.0
    scale = max(scale, eps)
    J_centered = J - pelvis_ref[None, None, :]
    J_cs = J_centered / scale
    return J_cs.astype(np.float32), pelvis_ref.astype(np.float32), float(scale)


def generate_windows(
    T: int, window_len: int, window_stride: int, tail_policy: str = "keep"
) -> Iterator[tuple[int, int, bool]]:
    if T <= 0:
        return
    is_padded = False
    if T <= window_len:
        if tail_policy == "drop" and T < window_len:
            return
        elif tail_policy == "pad" and T < window_len:
            yield 0, T, True
        else:
            yield 0, T, False
        return
    s = 0
    while s + window_len <= T:
        yield s, s + window_len, False
        s += window_stride
    r = T - s
    if r > 0:
        if tail_policy == "keep":
            yield s, T, False
        elif tail_policy == "pad":
            yield s, T, True
        else:
            pass


def uniform_sample_indices(t: int, target_len: int) -> np.ndarray:
    if t <= 0:
        return np.zeros((target_len,), dtype=np.int64)
    if t == target_len:
        return np.arange(t, dtype=np.int64)
    inds = np.linspace(0, t - 1, num=target_len)
    return np.round(inds).astype(np.int64)


def preprocess_full_to_Jcs(skel: np.ndarray, conf_thr: float) -> tuple[np.ndarray, np.ndarray]:
    if skel.ndim != 3 or skel.shape[1] != 17 or skel.shape[2] != 3:
        raise ValueError(f"Expected (T,17,3), got {skel.shape}")
    xy = skel[:, :, :2].astype(np.float32)
    conf = skel[:, :, 2].astype(np.float32)
    xy = interpolate_joints_over_time(xy, conf, conf_thr)
    J_cs, _, _ = center_and_scale_sequence(xy)
    return J_cs, conf


def build_segment(J_seg: np.ndarray, conf_seg: np.ndarray) -> np.ndarray:
    return np.concatenate([J_seg, conf_seg[:, :, None]], axis=2).astype(np.float32)


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess 2D skeletons (T,17,3: x,y,score) into windowed, multi-modal tensors."
    )
    p.add_argument(
        "--input-csvs",
        nargs="+",
        required=True,
        help="CSV(s) with columns: label,video_path,n_frames,skeleton_path (relative to CSV).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Path to write the combined output CSV (same columns; multiple rows per video if windowed).",
    )
    p.add_argument("--out-skeleton-dir", required=True, help="Directory (created) to store preprocessed .npy segments.")
    p.add_argument(
        "--conf-thr",
        type=float,
        default=0.20,
        help="Treat joints with score below this as missing and interpolate (default: 0.20).",
    )
    p.add_argument("--window-len", type=int, default=64, help="Sliding window length (frames).")
    p.add_argument("--window-stride", type=int, default=64, help="Sliding window stride (frames).")
    p.add_argument(
        "--tail-policy",
        choices=["keep", "pad", "drop"],
        default="keep",
        help="How to handle tail shorter than window-len (default: keep).",
    )
    p.add_argument(
        "--resample-len", type=int, default=None, help="If set, uniformly resample each window to this length."
    )
    p.add_argument(
        "--fail-missing", action="store_true", help="Raise on missing/unloadable skeleton files instead of skipping."
    )
    return p.parse_args()


def main() -> None:
    args = parse_arguments()
    input_csvs = [Path(p) for p in args.input_csvs]
    output = Path(args.output)
    out_skel_dir = Path(args.out_skeleton_dir)
    out_skel_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Combining CSVs...")
    df = load_and_combine_csvs(input_csvs)
    required_cols = {"label", "video_path", "n_frames", "skeleton_path"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in input CSVs: {missing_cols}")
    out_rows: list[dict] = []
    logger.info("Preprocessing %d skeleton files -> %s", len(df), out_skel_dir)
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Preprocess"):
        skel_path = resolve_skeleton_path(row)
        if not skel_path.exists():
            msg = f"Skeleton file not found: {skel_path}"
            if args.fail_missing:
                raise FileNotFoundError(msg)
            logger.warning(msg)
            continue
        skel = safe_load_npy(skel_path)
        if skel is None:
            if args.fail_missing:
                raise RuntimeError(f"Failed to load: {skel_path}")
            continue
        try:
            J_cs_full, conf_full = preprocess_full_to_Jcs(skel, conf_thr=args.conf_thr)
        except Exception as e:
            if args.fail_missing:
                raise
            logger.warning("Preprocess (to J_cs) failed for %s: %s", skel_path, e)
            continue
        T = J_cs_full.shape[0]
        win_iter = generate_windows(T, args.window_len, args.window_stride, tail_policy=args.tail_policy)
        base_rel = str(row["skeleton_path"])
        base_hash = hashlib.md5(base_rel.encode("utf-8")).hexdigest()
        seg_idx = 0
        for s, e, needs_pad in win_iter:
            J_seg = J_cs_full[s:e]
            conf_seg = conf_full[s:e]
            ts = J_seg.shape[0]
            if needs_pad and ts < args.window_len:
                pad_len = args.window_len - ts
                pad = np.zeros((pad_len, J_seg.shape[1], J_seg.shape[2]), dtype=J_seg.dtype)
                conf_pad = np.zeros((pad_len, conf_seg.shape[1]), dtype=conf_seg.dtype)
                J_seg = np.concatenate([J_seg, pad], axis=0)
                conf_seg = np.concatenate([conf_seg, conf_pad], axis=0)
                ts = J_seg.shape[0]
            if args.resample_len is not None and ts > 0:
                inds = uniform_sample_indices(ts, args.resample_len)
                J_seg = J_seg[inds]
                conf_seg = conf_seg[inds]
                ts = J_seg.shape[0]
            X = build_segment(J_seg, conf_seg)
            fname = f"{base_hash}_s{s}_e{e}_seg{seg_idx}.npy"
            out_path = out_skel_dir / fname
            try:
                np.save(out_path, X)
            except Exception as e:
                if args.fail_missing:
                    raise
                logger.warning("Failed to save %s: %s", out_path, e)
                continue
            out_rows.append(
                {
                    "label": row["label"],
                    "video_path": row["video_path"],
                    "n_frames": int(X.shape[0]),
                    "skeleton_path": f"{Path(args.out_skeleton_dir).name}/{fname}",
                }
            )
            seg_idx += 1
        if seg_idx == 0:
            logger.warning("No segments produced for %s (T=%d).", skel_path, T)
    out_df = pd.DataFrame(out_rows, columns=["label", "video_path", "n_frames", "skeleton_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output, index=False)
    logger.info("Wrote output CSV: %s (%d rows)", output, len(out_df))
    logger.info("Preprocessed segments saved to: %s", out_skel_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
