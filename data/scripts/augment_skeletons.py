import argparse
import logging
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
from tqdm import tqdm


# Configure logging similar to video augmentation script
logging.basicConfig(level="INFO", format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist.

    Args:
        path: Target directory path.
    """

    path.mkdir(parents=True, exist_ok=True)


def _rotation_matrix(angle_deg: float) -> np.ndarray:
    """Compute a 2x2 rotation matrix for the given angle in degrees.

    Args:
        angle_deg: Rotation angle in degrees.

    Returns:
        A 2x2 NumPy array with dtype float32 representing the rotation matrix.
    """

    theta = math.radians(float(angle_deg))
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float32)


def _transform_per_frame(coords: np.ndarray, frame_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply a per-frame transform to coordinates.

    Args:
        coords: Coordinate array of shape (T, J, 2) with dtype float32/float64.
        frame_fn: Function mapping a single-frame array (J, 2) -> (J, 2).

    Returns:
        Transformed coordinates of shape (T, J, 2).
    """

    T = coords.shape[0]
    return np.stack([frame_fn(coords[t]) for t in range(T)], axis=0)


def _rotate_xy(coords: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate coordinates by angle around per-frame centroid.

    Args:
        coords: Array of shape (T, J, 2).
        angle_deg: Rotation angle in degrees.

    Returns:
        Rotated coordinates of shape (T, J, 2).
    """

    R = _rotation_matrix(angle_deg)

    def fn(frame: np.ndarray) -> np.ndarray:
        # Compute centroid per frame
        centroid = frame.mean(axis=0, dtype=np.float32)
        centered = frame - centroid
        return centered @ R.T + centroid

    return _transform_per_frame(coords, fn)


def _scale_xy(coords: np.ndarray, scale: float) -> np.ndarray:
    """Scale coordinates around per-frame centroid.

    Args:
        coords: Array of shape (T, J, 2).
        scale: Multiplicative scale factor.

    Returns:
        Scaled coordinates of shape (T, J, 2).
    """

    s = float(scale)

    def fn(frame: np.ndarray) -> np.ndarray:
        centroid = frame.mean(axis=0, dtype=np.float32)
        centered = frame - centroid
        return centered * s + centroid

    return _transform_per_frame(coords, fn)


def _translate_xy(coords: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate coordinates by dx, dy.

    Args:
        coords: Array of shape (T, J, 2).
        dx: Horizontal shift in pixels.
        dy: Vertical shift in pixels.

    Returns:
        Translated coordinates of shape (T, J, 2).
    """

    shift = np.array([float(dx), float(dy)], dtype=np.float32)
    return coords + shift


def _jitter_xy(coords: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise to coordinates.

    Args:
        coords: Array of shape (T, J, 2).
        sigma: Standard deviation (pixels) for Gaussian noise.

    Returns:
        Jittered coordinates of shape (T, J, 2).
    """

    noise = np.random.normal(0.0, float(sigma), size=coords.shape).astype(np.float32)
    return coords + noise


def _flip_horizontal_xy(coords: np.ndarray) -> np.ndarray:
    """Flip coordinates horizontally around per-frame centroid.

    Args:
        coords: Array of shape (T, J, 2).

    Returns:
        Flipped coordinates of shape (T, J, 2).
    """

    def fn(frame: np.ndarray) -> np.ndarray:
        cx = frame[:, 0].mean(dtype=np.float32)
        out_f = frame.copy()
        out_f[:, 0] = (2.0 * cx) - out_f[:, 0]
        return out_f

    return _transform_per_frame(coords, fn)


def _time_crop(coords: np.ndarray, ratio: float) -> np.ndarray:
    """Temporal crop of coordinates by a ratio.

    Args:
        coords: Array of shape (T, J, 2).
        ratio: Fraction of frames to keep (0, 1].

    Returns:
        Cropped coordinates of shape (~T*ratio, J, 2).
    """

    T = coords.shape[0]
    keep = max(1, int(T * float(ratio)))
    if keep >= T:
        return coords
    start = np.random.randint(0, T - keep + 1)
    return coords[start : start + keep]


def _time_resample(coords: np.ndarray, new_T: int) -> np.ndarray:
    """Temporal resampling to a fixed number of frames.

    Args:
        coords: Array of shape (T, J, 2).
        new_T: Target number of frames (>= 1).

    Returns:
        Resampled coordinates of shape (new_T, J, 2).
    """

    T = coords.shape[0]
    if new_T <= 0:
        return coords
    # Index-based nearest sampling
    idx = np.linspace(0, T - 1, int(new_T)).round().astype(int)
    idx = np.clip(idx, 0, T - 1)
    return coords[idx]


def _apply_xy_aug_and_merge(arr: np.ndarray, fn_xy: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply an XY-only augmentation and merge confidence channel if present.

    Args:
        arr: Input array of shape (T, J, 2|3), dtype arbitrary.
        fn_xy: Function that transforms XY coordinates: (T, J, 2) -> (T', J, 2).

    Returns:
        Augmented array with dtype preserved; shape is (T', J, C) where C equals the
        original number of channels.
    """

    dtype = arr.dtype
    has_conf: bool = arr.shape[-1] == 3
    coords: np.ndarray = arr[..., :2].astype(np.float32)
    conf_opt: np.ndarray | None = arr[..., 2:] if has_conf else None

    aug_xy = fn_xy(coords)

    if has_conf:
        # Narrow type: conf_opt must be present when has_conf is True
        assert conf_opt is not None
        conf_arr: np.ndarray = conf_opt
        if aug_xy.shape[0] != coords.shape[0]:
            # Align confidence via the same time resampling strategy
            new_T = aug_xy.shape[0]
            conf_resampled: np.ndarray = _time_resample(conf_arr, new_T) if new_T != conf_arr.shape[0] else conf_arr
            out = np.concatenate([aug_xy, conf_resampled], axis=-1)
        else:
            out = np.concatenate([aug_xy, conf_arr], axis=-1)
    else:
        out = aug_xy

    return out.astype(dtype, copy=False)


def _get_aug_functions(
    *,
    augmentations: list[str],
    rotate_deg: float,
    scale_factor: float,
    translate_dx: float,
    translate_dy: float,
    jitter_sigma: float,
    time_crop_ratio: float,
    time_resample_T: int,
) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Build mapping from augmentation name to array transform function.

    Args:
        augmentations: List of augmentation names to include.
        center_mode: 'image' or 'centroid' for center-based transforms.
        img: ImageInfo or None when centroid mode is used.
        rotate_deg: Rotation angle in degrees.
        scale_factor: Multiplicative scale factor.
        translate_dx: Horizontal translation in pixels.
        translate_dy: Vertical translation in pixels.
        jitter_sigma: Gaussian noise sigma in pixels.
        time_crop_ratio: Fraction of frames to keep during crop.
        time_resample_T: Target number of frames when resampling.

    Returns:
        Dictionary: name -> function(arr: np.ndarray) -> np.ndarray.
    """

    fns: dict[str, Callable[[np.ndarray], np.ndarray]] = {}

    if "rotate" in augmentations:

        def rot(arr: np.ndarray) -> np.ndarray:
            return _apply_xy_aug_and_merge(arr, lambda xy: _rotate_xy(xy, rotate_deg))

        fns["rot"] = rot

    if "scale" in augmentations:

        def scl(arr: np.ndarray) -> np.ndarray:
            return _apply_xy_aug_and_merge(arr, lambda xy: _scale_xy(xy, scale_factor))

        fns["scale"] = scl

    if "translate" in augmentations:

        def trn(arr: np.ndarray) -> np.ndarray:
            return _apply_xy_aug_and_merge(arr, lambda xy: _translate_xy(xy, translate_dx, translate_dy))

        fns["trans"] = trn

    if "jitter" in augmentations:

        def jit(arr: np.ndarray) -> np.ndarray:
            return _apply_xy_aug_and_merge(arr, lambda xy: _jitter_xy(xy, jitter_sigma))

        fns["jitter"] = jit

    if "flip" in augmentations:

        def flp(arr: np.ndarray) -> np.ndarray:
            return _apply_xy_aug_and_merge(arr, lambda xy: _flip_horizontal_xy(xy))

        fns["flip"] = flp

    if "time_crop" in augmentations:

        def tc(arr: np.ndarray) -> np.ndarray:
            return _apply_xy_aug_and_merge(arr, lambda xy: _time_crop(xy, time_crop_ratio))

        fns["tcrop"] = tc

    if "time_resample" in augmentations:

        def tr(arr: np.ndarray) -> np.ndarray:
            return _apply_xy_aug_and_merge(arr, lambda xy: _time_resample(xy, time_resample_T))

        fns["tresamp"] = tr

    return fns


def _suffix_for_aug(
    name: str,
    *,
    rotate_deg: float,
    scale_factor: float,
    translate_dx: float,
    translate_dy: float,
    jitter_sigma: float,
    time_crop_ratio: float,
    time_resample_T: int,
) -> str:
    """Generate filename suffix for a given augmentation name and params.

    Args:
        name: Internal augmentation key from _get_aug_functions result.
        rotate_deg: Rotation angle.
        scale_factor: Scale factor.
        translate_dx: Shift x.
        translate_dy: Shift y.
        jitter_sigma: Noise sigma.
        time_crop_ratio: Crop ratio.
        time_resample_T: Resample target frames.

    Returns:
        Suffix string without leading underscore.
    """

    if name == "rot":
        return f"rot{round(rotate_deg)}"
    if name == "scale":
        return f"scale{round(scale_factor * 100)}"
    if name == "trans":
        return f"trans_{round(translate_dx)}_{round(translate_dy)}"
    if name == "jitter":
        # Keep simple name to match the notebook
        return "jitter"
    if name == "flip":
        return "flip"
    if name == "tcrop":
        # Use integer percent
        return f"tcrop{round(time_crop_ratio * 100)}"
    if name == "tresamp":
        return f"tresamp{time_resample_T}"
    return name


def _find_skeleton_files(input_dir: Path, glob_pattern: str) -> list[Path]:
    """Find .npy skeleton files in a directory.

    Args:
        input_dir: Directory to search.
        glob_pattern: Glob pattern, e.g. "*.npy".

    Returns:
        Sorted list of file paths.
    """

    files = sorted([p for p in input_dir.glob(glob_pattern) if p.is_file()])
    return files


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for skeleton augmentation.

    Returns:
        Parsed arguments namespace.
    """

    parser = argparse.ArgumentParser(description="Batch-augment skeleton .npy files")
    parser.add_argument("--input-dir", required=True, help="Directory containing input .npy skeleton files")
    parser.add_argument("--output-dir", required=True, help="Directory to write augmented .npy files")
    parser.add_argument("--glob", default="*.npy", help="Glob pattern to select input files (default: *.npy)")
    parser.add_argument(
        "--augmentations",
        nargs="+",
        choices=["rotate", "scale", "translate", "jitter", "flip", "time_crop", "time_resample"],
        default=["rotate", "scale", "translate", "jitter", "flip", "time_crop", "time_resample"],
        help="Augmentations to apply",
    )
    # no image/center options; centroid is always used
    parser.add_argument("--rotate-deg", type=float, default=10.0, help="Rotation angle in degrees")
    parser.add_argument("--scale-factor", type=float, default=1.1, help="Scale factor")
    parser.add_argument("--translate-dx", type=float, default=5.0, help="Translation in x (pixels)")
    parser.add_argument("--translate-dy", type=float, default=-3.0, help="Translation in y (pixels)")
    parser.add_argument("--jitter-sigma", type=float, default=1.5, help="Gaussian noise sigma (pixels)")
    parser.add_argument("--time-crop-ratio", type=float, default=0.8, help="Temporal crop ratio (0,1]")
    parser.add_argument("--time-resample-T", type=int, default=6, help="Temporal resample target frames")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main() -> None:
    """Main entry point for skeleton augmentation CLI."""

    args = parse_arguments()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    glob_pattern = str(args.glob)
    augmentations: list[str] = list(args.augmentations)

    # Seed RNGs
    np.random.seed(int(args.seed))

    # Ensure directories
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found or not a directory: {input_dir}")
    ensure_dir(output_dir)

    # Discover files
    files = _find_skeleton_files(input_dir, glob_pattern)
    if not files:
        logger.warning("No input files found in %s with pattern %s", input_dir, glob_pattern)
        return

    logger.info("Starting skeleton augmentation")
    logger.info("Input dir: %s", input_dir)
    logger.info("Output dir: %s", output_dir)
    logger.info("Files to process: %d", len(files))
    logger.info("Augmentations: %s", augmentations)
    logger.info("Center mode: centroid")

    # Build augmentation functions
    aug_functions = _get_aug_functions(
        augmentations=augmentations,
        rotate_deg=float(args.rotate_deg),
        scale_factor=float(args.scale_factor),
        translate_dx=float(args.translate_dx),
        translate_dy=float(args.translate_dy),
        jitter_sigma=float(args.jitter_sigma),
        time_crop_ratio=float(args.time_crop_ratio),
        time_resample_T=args.time_resample_T,
    )

    # For suffix generation
    def make_suffix(key: str) -> str:
        return _suffix_for_aug(
            key,
            rotate_deg=float(args.rotate_deg),
            scale_factor=float(args.scale_factor),
            translate_dx=float(args.translate_dx),
            translate_dy=float(args.translate_dy),
            jitter_sigma=float(args.jitter_sigma),
            time_crop_ratio=float(args.time_crop_ratio),
            time_resample_T=args.time_resample_T,
        )

    # Process files
    num_by_aug: dict[str, int] = {k: 0 for k in aug_functions}

    for npy_path in tqdm(files, desc="Augmenting", unit="file"):
        try:
            arr = np.load(npy_path)
        except Exception as e:
            logger.error("Failed to load %s: %s", npy_path, e)
            continue

        # Validate shape
        if arr.ndim != 3 or arr.shape[-1] not in (2, 3):
            logger.warning("Skipping %s due to incompatible shape: %s", npy_path.name, arr.shape)
            continue

        for key, fn in aug_functions.items():
            try:
                aug = fn(arr)
            except Exception as e:
                logger.error("Augmentation '%s' failed on %s: %s", key, npy_path.name, e)
                continue

            suffix = make_suffix(key)
            out_path = output_dir / f"{npy_path.stem}_{suffix}.npy"
            try:
                ensure_dir(out_path.parent)
                np.save(out_path, aug)
                num_by_aug[key] += 1
            except Exception as e:
                logger.error("Failed to save %s: %s", out_path, e)

    # Summary
    logger.info("Augmentation completed. Output: %s", output_dir)
    total_out = sum(num_by_aug.values())
    logger.info("Total augmented files: %d", total_out)
    for key, cnt in num_by_aug.items():
        logger.info("  %s -> %d", key, cnt)


if __name__ == "__main__":
    main()
