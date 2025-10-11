from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scipy.io import loadmat
from tqdm import tqdm

from data.scripts.dataset_utils import (
    build_video_from_frames,
    collect_frame_files,
    load_normalization,
    normalize_with_rules,
    write_semicolon_csv,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _iter_label_files(labels_dir: Path) -> list[Path]:
    return sorted(labels_dir.glob("*.mat"))


def _extract_action_from_mat(mat_path: Path) -> str | None:
    mat = loadmat(str(mat_path))
    action = mat.get("action")
    if action is None:
        return None
    try:
        while isinstance(action, list | tuple):
            action = action[0]
        if hasattr(action, "ravel"):
            action = action.ravel()[0]
        if isinstance(action, bytes):
            return action.decode("utf-8", errors="ignore")
        return str(action)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Penn dataset: build MP4 videos from frames, normalize labels, and write semicolon CSV.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset root containing frames/ and labels/ directories.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for generated videos (default: 30).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing MP4 files if present.")
    parser.add_argument("--dry-run", action="store_true", help="Only print summary, do not write CSV or videos.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (semicolon-separated). Default: <root>/labels_and_links.csv",
    )
    default_norm = Path(__file__).resolve().parent / "label_normalization.json"
    parser.add_argument(
        "--norm",
        type=Path,
        default=default_norm,
        help="Path to label normalization JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.root.resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root directory does not exist or is not a directory: {root}")

    out_csv: Path = args.out if args.out is not None else (root / "labels_and_links.csv")
    frames_root = root / "frames"
    labels_root = root / "labels"
    videos_root = root / "videos_mp4"

    target, synonyms = load_normalization(args.norm)

    label_files = list(_iter_label_files(labels_root))
    rows: list[tuple[str, str]] = []

    for label_file in tqdm(label_files, desc="Processing Penn videos"):
        base_name = label_file.stem
        raw_label = _extract_action_from_mat(label_file)
        if not raw_label:
            continue
        normalized = normalize_with_rules(raw_label, synonyms, target)
        if normalized is None:
            continue

        frames_dir = frames_root / base_name
        frame_files = collect_frame_files(frames_dir)
        if not frame_files:
            continue

        video_out = videos_root / f"{base_name}.mp4"
        if not args.dry_run and (args.overwrite or not video_out.exists()):
            _ = build_video_from_frames(frame_files, video_out, args.fps)

        if args.dry_run or video_out.exists():
            rel = video_out.relative_to(out_csv.parent).as_posix()
            rows.append((rel, normalized))

    if args.dry_run:
        total = len(rows)
        by_label: dict[str, int] = {}
        for _, label in rows:
            by_label[label] = by_label.get(label, 0) + 1
        logger.info("Found %d videos matching target labels under %s", total, root)
        for label in sorted(by_label):
            logger.info("  %s: %d", label, by_label[label])
        return

    write_semicolon_csv(out_csv, rows)
    logger.info("Wrote %d rows to %s", len(rows), out_csv)


if __name__ == "__main__":
    main()
