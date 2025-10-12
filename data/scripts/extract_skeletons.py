import argparse
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

from act_rec.labeling import YoloPoseVideoLabeler
from act_rec.params import YoloPoseVideoInferenceParams


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments containing input CSV paths and output path.
    """
    parser = argparse.ArgumentParser(
        description="Extract skeletons from videos using YOLO pose estimation",
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
        help="Path where to save the output CSV with skeleton information",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to YOLO pose model",
    )
    parser.add_argument(
        "--skeleton-dir",
        required=True,
        help="Directory to save skeleton files",
    )

    return parser.parse_args()


def load_and_combine_csvs(csv_paths: list[str]) -> pd.DataFrame:
    """Load and combine multiple CSV files into a single DataFrame.

    Args:
        csv_paths: List of paths to CSV files to load and combine.

    Returns:
        Combined DataFrame with all video labels.

    Raises:
        FileNotFoundError: If any of the CSV files cannot be found.
        pd.errors.EmptyDataError: If any CSV file is empty.
    """
    combined_data = []

    for csv_path in csv_paths:
        logger.info("Loading CSV file: %s", csv_path)
        try:
            df = pd.read_csv(csv_path)
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


def process_videos(
    labels_df: pd.DataFrame,
    labeler: YoloPoseVideoLabeler,
    skeleton_dir: Path,
    base_dir: Path,
) -> pd.DataFrame:
    """Process videos to extract skeleton keypoints.

    Args:
        labels_df: DataFrame containing video labels and paths.
        labeler: YOLO pose video labeler instance.
        skeleton_dir: Directory to save skeleton files.
        base_dir: Base directory for resolving relative video paths.

    Returns:
        DataFrame with added skeleton information (n_frames, skeleton_path).
    """
    # Clean the data
    labels_df = labels_df.dropna(subset=["video_path"]).copy()

    n_frames: list[int | None] = []
    skeleton_paths: list[Path | None] = []

    # Create skeleton directory
    skeleton_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing %d videos...", len(labels_df))

    for _, row in tqdm.tqdm(labels_df.iterrows(), total=len(labels_df), desc="Processing videos"):
        video_path = row["video_path"]
        absolute_video_path = base_dir / video_path

        if not absolute_video_path.exists():
            logger.warning("Video file not found: %s", absolute_video_path)
            n_frames.append(None)
            skeleton_paths.append(None)
            continue

        try:
            is_single_person, keypoints = labeler.label_video(
                str(absolute_video_path),
                multi_person_tollerance_n_frames=15,
            )

            if not is_single_person or keypoints is None:
                logger.debug("Skipping multi-person video: %s", video_path)
                n_frames.append(None)
                skeleton_paths.append(None)
                continue

            # Generate unique filename for skeleton
            skeleton_filename = f"{hashlib.md5(video_path.encode()).hexdigest()}.npy"
            skeleton_file_path = skeleton_dir / skeleton_filename

            # Save skeleton keypoints
            np.save(skeleton_file_path, keypoints)

            n_frames.append(keypoints.shape[0])
            skeleton_paths.append(Path(skeleton_dir.name) / skeleton_filename)

            logger.debug("Processed video: %s -> %d frames", video_path, keypoints.shape[0])

        except Exception as e:
            logger.error("Error processing video %s: %s", video_path, e)
            n_frames.append(None)
            skeleton_paths.append(None)

    # Add results to DataFrame
    labels_df["n_frames"] = n_frames
    labels_df["skeleton_path"] = skeleton_paths

    # Log statistics
    successful_videos = sum(1 for n in n_frames if n is not None)
    logger.info("Successfully processed %d/%d videos", successful_videos, len(labels_df))

    return labels_df


def main() -> None:
    """Main function to orchestrate skeleton extraction process."""
    args = parse_arguments()

    # Convert paths to Path objects
    input_csvs = [Path(csv_path) for csv_path in args.input_csvs]
    output_path = Path(args.output)
    model_path = Path(args.model_path)
    skeleton_dir = Path(args.skeleton_dir)

    # Determine base directory (parent of output directory)
    base_dir = output_path.parent

    logger.info("Starting skeleton extraction process...")
    logger.info("Input CSVs: %s", [str(p) for p in input_csvs])
    logger.info("Output path: %s", output_path)
    logger.info("Model path: %s", model_path)
    logger.info("Skeleton directory: %s", skeleton_dir)

    # Initialize YOLO labeler
    params = YoloPoseVideoInferenceParams()
    labeler = YoloPoseVideoLabeler(
        model_path=str(model_path),
        params=params,
    )

    # Load and combine CSV files
    try:
        combined_labels = load_and_combine_csvs([str(p) for p in input_csvs])
    except Exception as e:
        logger.error("Failed to load CSV files: %s", e)
        return

    # Process videos
    try:
        processed_labels = process_videos(
            combined_labels,
            labeler,
            base_dir / skeleton_dir,
            base_dir,
        )
    except Exception as e:
        logger.error("Failed to process videos: %s", e)
        return

    # Save results
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_labels.to_csv(output_path, index=False)
        logger.info("Results saved to: %s", output_path)
    except Exception as e:
        logger.error("Failed to save results: %s", e)
        return

    logger.info("Skeleton extraction completed successfully!")


if __name__ == "__main__":
    main()
