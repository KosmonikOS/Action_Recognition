import argparse
import logging
import math
import os
import random
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip, vfx
from tqdm import tqdm


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _select_opencv_fourcc_by_extension(ext: str) -> int:
    """Select an appropriate OpenCV fourcc code based on output extension.

    Args:
        ext: Extension including the leading dot, lowercase (e.g., ".mp4").

    Returns:
        OpenCV fourcc integer.
    """
    if ext in {".mp4", ".m4v", ".mov"}:
        return cv2.VideoWriter.fourcc(*"mp4v")
    if ext == ".avi":
        return cv2.VideoWriter.fourcc(*"XVID")
    # Fallback: use mp4v by default
    return cv2.VideoWriter.fourcc(*"mp4v")


def _create_video_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create a cv2.VideoWriter with extension-aware codec and fallbacks.

    Tries the recommended codec for the extension; for .avi falls back to MJPG
    if the primary codec is not available. As a last resort, tries mp4v.

    Args:
        output_path: Target video file path.
        fps: Frames per second for the writer.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        An opened cv2.VideoWriter instance.

    Raises:
        RuntimeError: If a suitable writer could not be created.
    """
    ext = Path(output_path).suffix.lower()
    primary = _select_opencv_fourcc_by_extension(ext)
    writer = cv2.VideoWriter(output_path, primary, fps, (width, height))
    if writer.isOpened():
        return writer

    logger.warning("Primary codec failed for %s. Trying fallbacks.", output_path)

    # Targeted fallback for AVI
    if ext == ".avi":
        mjpg = cv2.VideoWriter.fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, mjpg, fps, (width, height))
        if writer.isOpened():
            logger.info("Falling back to MJPG for AVI output: %s", output_path)
            return writer

    # Last resort fallback
    mp4v = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, mp4v, fps, (width, height))
    if writer.isOpened():
        logger.info("Falling back to mp4v for output: %s", output_path)
        return writer

    raise RuntimeError(f"Could not open VideoWriter for {output_path}")


def _select_moviepy_codec_by_extension(ext: str) -> str:
    """Select an appropriate MoviePy/FFmpeg codec based on output extension.

    Args:
        ext: Extension including the leading dot, lowercase (e.g., ".mp4").

    Returns:
        FFmpeg codec name string.
    """
    if ext in {".mp4", ".m4v", ".mov"}:
        return "libx264"
    if ext == ".avi":
        return "mpeg4"
    # Fallback to a widely-available codec
    return "libx264"


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments containing input CSV paths, CSV output path,
        augmentation output directory, and augmentation options.
    """
    parser = argparse.ArgumentParser(
        description="Augment video datasets with various transformations",
    )
    parser.add_argument(
        "--input-csvs",
        nargs="+",
        required=True,
        help="Paths to input CSV files containing video labels",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV file or directory where it will be saved",
    )
    parser.add_argument(
        "--augmentation-dir",
        required=True,
        help="Directory to save augmented videos",
    )
    parser.add_argument(
        "--augmentations",
        nargs="+",
        choices=["flip", "color_jitter", "speed_change"],
        default=["flip", "color_jitter", "speed_change"],
        help="List of augmentations to apply",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible augmentations",
    )

    return parser.parse_args()


def flip_video(input_path: str, output_path: str) -> None:
    """Apply horizontal flip augmentation to video.

    Args:
        input_path: Path to input video file.
        output_path: Path where augmented video will be saved.

    Raises:
        RuntimeError: If video cannot be opened or processed.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0 or not math.isfinite(float(fps)):
        logger.warning("Invalid FPS '%s' detected for %s. Falling back to 30.0 FPS.", fps, input_path)
        fps = 30.0

    out = _create_video_writer(output_path, fps, width, height)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            flipped_frame = cv2.flip(frame, 1)
            out.write(flipped_frame)
    finally:
        cap.release()
        out.release()


def jitter_color(input_path: str, output_path: str) -> None:
    """Apply color jittering augmentation to video.

    Randomly adjusts saturation and brightness in HSV color space.

    Args:
        input_path: Path to input video file.
        output_path: Path where augmented video will be saved.

    Raises:
        RuntimeError: If video cannot be opened or processed.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0 or not math.isfinite(float(fps)):
        logger.warning("Invalid FPS '%s' detected for %s. Falling back to 30.0 FPS.", fps, input_path)
        fps = 30.0

    out = _create_video_writer(output_path, fps, width, height)

    saturation_factor = random.uniform(0.6, 1.4)
    brightness_factor = random.uniform(0.7, 1.3)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_frame)

            s = np.clip(s * saturation_factor, 0, 255).astype(np.uint8)
            v = np.clip(v * brightness_factor, 0, 255).astype(np.uint8)

            final_hsv = cv2.merge((h, s, v))
            jittered_frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            out.write(jittered_frame)
    finally:
        cap.release()
        out.release()


def change_speed(input_path: str, output_path: str) -> None:
    """Apply speed change augmentation to video.

    Randomly speeds up or slows down the video playback.

    Args:
        input_path: Path to input video file.
        output_path: Path where augmented video will be saved.

    Raises:
        RuntimeError: If video cannot be processed with MoviePy.
    """
    speed_factor = random.choice([0.5, 0.75, 1.25, 1.5])

    clip = VideoFileClip(input_path)

    try:
        final_clip = clip.fx(vfx.speedx, speed_factor)
        ext = Path(output_path).suffix.lower()
        codec = _select_moviepy_codec_by_extension(ext)
        final_clip.write_videofile(output_path, codec=codec, logger=None)
    except Exception as e:
        clip.close()
        raise RuntimeError(f"Could not apply speed effect: {e}") from e
    finally:
        clip.close()
        if "final_clip" in locals():
            final_clip.close()


def load_and_combine_csvs(csv_paths: list[str]) -> pd.DataFrame:
    """Load and combine multiple CSV files into a single DataFrame.

    Args:
        csv_paths: List of paths to CSV files to load and combine.

    Returns:
        Combined DataFrame with all video labels.

    Raises:
        FileNotFoundError: If any of the CSV files cannot be found.
        ValueError: If no valid CSV files are found.
    """
    combined_data = []

    for csv_path in csv_paths:
        logger.info("Loading CSV file: %s", csv_path)
        try:
            # Try reading with header first, handling spaces after commas
            df = pd.read_csv(csv_path, skipinitialspace=True)

            # If required columns are missing, try without header and infer
            has_required = ("video_path" in df.columns or "filename" in df.columns) and ("label" in df.columns)
            if not has_required:
                df = pd.read_csv(csv_path, header=None)
                # Detect schema based on content
                if df.shape[1] == 2:
                    col0 = df[0].astype(str)
                    col1 = df[1].astype(str)

                    def looks_like_path(s: pd.Series) -> pd.Series:
                        # Detect common video extensions or path separators
                        return s.str.contains(
                            r"(\.mp4|\.m4v|\.mov|\.avi|\.mkv)$",
                            case=False,
                            regex=True,
                        ) | s.str.contains(r"[\\\/]", regex=True)

                    if looks_like_path(col0).any() and not looks_like_path(col1).any():
                        df.columns = ["video_path", "label"]
                    elif looks_like_path(col1).any() and not looks_like_path(col0).any():
                        df.columns = ["label", "video_path"]
                    else:
                        df.columns = ["label", "filename"]

            # Ensure we have the required columns
            if "video_path" not in df.columns and "filename" not in df.columns:
                logger.error("CSV file %s missing video path/filename column", csv_path)
                continue

            if "label" not in df.columns:
                logger.error("CSV file %s missing label column", csv_path)
                continue

            # Track the directory of the CSV to resolve relative video paths later
            csv_dir = str(Path(csv_path).parent)
            df["_csv_dir"] = csv_dir

            logger.info("Loaded %d records from %s", len(df), csv_path)
            combined_data.append(df)

        except FileNotFoundError:
            logger.error("CSV file not found: %s", csv_path)
            raise
        except pd.errors.EmptyDataError:
            logger.warning("Empty CSV file: %s", csv_path)
            continue

    if not combined_data:
        raise ValueError("No valid CSV files found")

    combined_df = pd.concat(combined_data, ignore_index=True)
    logger.info("Combined %d total records from %d files", len(combined_df), len(csv_paths))

    return combined_df


def get_augmentation_functions() -> dict[str, Callable[[str, str], None]]:
    """Get mapping of augmentation names to functions.

    Returns:
        Dictionary mapping augmentation names to their corresponding functions.
    """
    return {
        "flip": flip_video,
        "color_jitter": jitter_color,
        "speed_change": change_speed,
    }


def process_augmentation(
    labels_df: pd.DataFrame,
    aug_name: str,
    aug_function: Callable[[str, str], None],
    augmentation_dir: Path,
    csv_base_dir: Path | None = None,
) -> pd.DataFrame:
    """Process videos with a specific augmentation.

    Args:
        labels_df: DataFrame containing video labels and paths.
        aug_name: Name of the augmentation being applied.
        aug_function: Function to apply the augmentation.
        augmentation_dir: Directory to save augmented videos.
        csv_base_dir: Base directory to make saved video paths relative to in output CSV.

    Returns:
        DataFrame with augmented video paths and labels.
    """
    # Create augmentation-specific output directory
    aug_output_dir = augmentation_dir / f"augmented_{aug_name}"
    aug_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Applying augmentation: %s", aug_name)
    logger.info("Results will be saved in: %s", aug_output_dir)

    new_rows = []

    # Determine path mode
    has_video_path = "video_path" in labels_df.columns
    path_mode = "full" if has_video_path else "name"

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc=f"Processing {aug_name}"):
        label = row["label"]

        if path_mode == "full":
            input_video_path = Path(row["video_path"])
            if not input_video_path.is_absolute():
                csv_dir = row.get("_csv_dir", "")
                if csv_dir:
                    input_video_path = Path(str(csv_dir)) / input_video_path
                else:
                    # Try to resolve relative to current working directory
                    logger.warning(
                        "No CSV directory available; using CWD for: %s",
                        input_video_path,
                    )
            filename_only = input_video_path.name
        else:
            filename_only = str(row["filename"])
            csv_dir = row.get("_csv_dir", "")
            if csv_dir:
                input_video_path = Path(str(csv_dir)) / filename_only
            else:
                logger.error("Cannot resolve filename without CSV directory: %s", filename_only)
                continue

        # Generate output filename
        name_parts = filename_only.rsplit(".", 1)
        if len(name_parts) == 2:
            new_filename = f"{aug_name}_{name_parts[0]}.{name_parts[1]}"
        else:
            new_filename = f"{aug_name}_{filename_only}"

        output_video_path = aug_output_dir / new_filename

        if not input_video_path.exists():
            logger.warning("Video file not found: %s", input_video_path)
            continue

        try:
            aug_function(str(input_video_path), str(output_video_path))

            # Store path relative to CSV base dir if provided; else relative to augmentation dir
            if csv_base_dir is not None:
                relative_output_path = os.path.relpath(str(output_video_path), start=str(csv_base_dir))
            else:
                relative_output_path = os.path.relpath(str(output_video_path), start=str(augmentation_dir))
            new_rows.append(
                {
                    "video_path": str(relative_output_path),
                    "label": label,
                    "augmentation": aug_name,
                },
            )

        except Exception as e:
            logger.error("Error processing video %s with %s: %s", input_video_path, aug_name, e)
            continue

    # Create DataFrame with results
    new_df = pd.DataFrame(new_rows)
    logger.info("Successfully processed %d/%d videos for %s", len(new_df), len(labels_df), aug_name)

    return new_df


def main() -> None:
    """Main function to orchestrate video augmentation process."""
    args = parse_arguments()

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Convert paths to Path objects
    input_csvs = [Path(csv_path) for csv_path in args.input_csvs]
    # Determine directories from CLI
    # Augmented videos directory (required)
    augmentation_dir = Path(args.augmentation_dir)

    # Output CSV path (file or directory)
    output_arg = Path(args.output)
    if output_arg.suffix.lower() == ".csv" or output_arg.name.lower().endswith(".csv"):
        output_csv_path = output_arg
    else:
        output_csv_path = output_arg / "augmented_annotations.csv"
    output_csv_dir = output_csv_path.parent

    logger.info("Starting video augmentation process...")
    logger.info("Input CSVs: %s", [str(p) for p in input_csvs])
    logger.info("Augmentation directory: %s", augmentation_dir)
    logger.info("CSV output path: %s", output_csv_path)
    logger.info("Augmentations: %s", args.augmentations)

    # Create required directories
    augmentation_dir.mkdir(parents=True, exist_ok=True)
    output_csv_dir.mkdir(parents=True, exist_ok=True)

    # Load and combine CSV files
    try:
        combined_labels = load_and_combine_csvs([str(p) for p in input_csvs])
        logger.info("Loaded %d videos to process", len(combined_labels))
    except Exception as e:
        logger.error("Failed to load CSV files: %s", e)
        return

    # Get augmentation functions
    augmentation_functions = get_augmentation_functions()

    # Process each augmentation
    all_augmented_data = []

    for aug_name in args.augmentations:
        if aug_name not in augmentation_functions:
            logger.warning("Unknown augmentation: %s", aug_name)
            continue

        aug_function = augmentation_functions[aug_name]

        try:
            augmented_df = process_augmentation(
                combined_labels,
                aug_name,
                aug_function,
                augmentation_dir,
                output_csv_dir,
            )
            all_augmented_data.append(augmented_df)

        except Exception as e:
            logger.error("Failed to process augmentation %s: %s", aug_name, e)
            continue

    # Combine all augmented data and save
    if all_augmented_data:
        # Filter out empty DataFrames
        non_empty_data = [df for df in all_augmented_data if len(df) > 0]

        if non_empty_data:
            final_df = pd.concat(non_empty_data, ignore_index=True)

            try:
                final_df.to_csv(output_csv_path, index=False)
                logger.info("Augmented annotations saved to: %s", output_csv_path)
                logger.info("Total augmented videos: %d", len(final_df))

                # Log statistics by augmentation
                for aug_name in final_df["augmentation"].unique():
                    count = len(final_df[final_df["augmentation"] == aug_name])
                    logger.info("  %s: %d videos", aug_name, count)

            except Exception as e:
                logger.error("Failed to save results: %s", e)
                return
        else:
            logger.warning("No videos were successfully augmented")
    else:
        logger.warning("No augmented data to save")

    logger.info("Video augmentation completed successfully!")


if __name__ == "__main__":
    main()
