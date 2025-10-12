import argparse
import logging
from pathlib import Path

from data.scripts.dataset_utils import (
    iter_video_files,
    load_normalization,
    normalize_with_rules,
    relative_path_from_base,
    write_semicolon_csv,
)


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Kinetics dataset: scan under root, normalize labels, and write semicolon CSV.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Path to Kinetics root (contains train/ val/ test/). CSV paths are relative to this root.",
    )
    parser.add_argument(
        "--extensions",
        dest="extensions",
        nargs="+",
        default=["mp4", "avi", "mov", "mkv", "webm", "mpg", "mpeg"],
        help="Video file extensions to include (without dots).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print summary, do not write the CSV.")
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

    extensions = {ext.lower().lstrip(".") for ext in args.extensions}
    out_csv: Path = args.out if args.out is not None else (root / "labels_and_links.csv")

    target, synonyms = load_normalization(args.norm)

    rows: list[tuple[str, str]] = []
    for file_path in iter_video_files(root, extensions):
        raw_label = file_path.parent.name
        normalized = normalize_with_rules(raw_label, synonyms, target)
        if normalized is None:
            continue
        rel = relative_path_from_base(out_csv.parent, file_path)
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
