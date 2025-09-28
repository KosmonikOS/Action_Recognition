import argparse
import re
from collections.abc import Iterable, Iterator
from pathlib import Path


TARGET_LABELS: set[str] = {
    "bench_press",
    "squat",
    "clean_and_jerk",
    "lunges",
    "pullup",
    "pushup",
    "running_on_treadmill",
    "situp",
    "snatch_weight_lifting",
    "jumping_jacks",
    "jump_rope",
    "handstand_walking",
    "wall_pushups",
    "handstand_pushups",
}


def _normalize_text_basic(text: str) -> str:
    """Lowercase and normalize separators to single spaces, keep a-z0-9 only.

    Returns a snake_case string suitable for dictionary key matching.
    """

    camel_spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    camel_spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", camel_spaced)
    lowered = camel_spaced.lower()

    for ch in ["-", "/", "\\", "_"]:
        lowered = lowered.replace(ch, " ")
    cleaned_chars: list[str] = []
    for ch in lowered:
        if ch.isalnum():
            cleaned_chars.append(ch)
        elif ch.isspace():
            cleaned_chars.append(" ")

    cleaned = "".join(cleaned_chars)
    tokens = [t for t in cleaned.split() if t]
    return "_".join(tokens)


def _normalize_with_rules(label: str) -> str | None:
    """Normalize a raw label to one of TARGET_LABELS or return None if not matched.

    Heuristics and synonyms are based on common dataset variants
    (e.g., Kinetics, HMDB, UCF, custom exports).
    """
    key = _normalize_text_basic(label)

    synonyms: dict[str, str] = {
        "bench_pressing": "bench_press",
        "bench_press": "bench_press",
        "benchpress": "bench_press",
        "bench_press_": "bench_press",
        "bench_press __": "bench_press",
        "bench_press __": "bench_press",
        "bench_press ": "bench_press",
        "bench_press__": "bench_press",
        "bench_press__ __": "bench_press",
        "bench_press_ __": "bench_press",
        "bench_pressings": "bench_press",
        "bench_presses": "bench_press",
        "squat": "squat",
        "squats": "squat",
        "squatting": "squat",
        "body_weight_squats": "squat",
        "bodyweight_squats": "squat",
        "body_weight_squat": "squat",
        "bodyweight_squat": "squat",
        "lunge": "lunges",
        "lunges": "lunges",
        "pull_up": "pullup",
        "pull_ups": "pullup",
        "pullup": "pullup",
        "pullups": "pullup",
        "pull_up_exercise": "pullup",
        "pull_ups __": "pullup",
        "push_up": "pushup",
        "push_ups": "pushup",
        "pushup": "pushup",
        "pushups": "pushup",
        "wall_pushups": "wall_pushups",
        "wall_push_ups": "wall_pushups",
        "wall_pushup": "wall_pushups",
        "handstand_pushups": "handstand_pushups",
        "handstand_push_ups": "handstand_pushups",
        "handstand_pushup": "handstand_pushups",
        "running_on_treadmill": "running_on_treadmill",
        "treadmill_running": "running_on_treadmill",
        "run_on_treadmill": "running_on_treadmill",
        "running_treadmill": "running_on_treadmill",
        "situp": "situp",
        "situps": "situp",
        "sit_up": "situp",
        "sit_ups": "situp",
        "snatch_weightlifting": "snatch_weight_lifting",
        "snatch_weight_lifting": "snatch_weight_lifting",
        "weightlifting_snatch": "snatch_weight_lifting",
        "weight_lifting_snatch": "snatch_weight_lifting",
        "snatch": "snatch_weight_lifting",
        "clean_and_jerk": "clean_and_jerk",
        "jumping_jack": "jumping_jacks",
        "jumping_jacks": "jumping_jacks",
        "jump_jack": "jumping_jacks",
        "jump_jacks": "jumping_jacks",
        "jumpingjack": "jumping_jacks",
        "jump_rope": "jump_rope",
        "jumping_rope": "jump_rope",
        "jumprope": "jump_rope",
        "skipping_rope": "jump_rope",
        "handstand_walking": "handstand_walking",
        "handstand_walk": "handstand_walking",
        "handstand_walks": "handstand_walking",
    }

    if key in synonyms:
        return synonyms[key]

    tokens = set(key.split("_"))

    if {"running", "treadmill"}.issubset(tokens) or ({"run", "treadmill"}.issubset(tokens)):
        return "running_on_treadmill"

    if "snatch" in tokens and ("weightlifting" in tokens or ("weight" in tokens and "lifting" in tokens)):
        return "snatch_weightlifting"

    if tokens.issuperset({"push"}) and ("up" in tokens or "ups" in tokens):
        return "push_up"

    if tokens.issuperset({"pull"}) and ("up" in tokens or "ups" in tokens):
        return "pull_ups"

    return None


def _iter_video_files(root: Path, extensions: set[str]) -> Iterator[Path]:
    for ext in extensions:
        yield from root.rglob(f"*.{ext}")


def _relative_path_str(root: Path, file_path: Path) -> str:
    return file_path.relative_to(root).as_posix()


def build_csv_rows(root: Path, extensions: set[str]) -> list[tuple[str, str]]:
    """Scan the root for videos and return (relative_path, normalized_label) rows.

    Only rows whose parent-folder label normalizes to TARGET_LABELS are included.
    """
    rows: list[tuple[str, str]] = []
    for file_path in _iter_video_files(root, extensions):
        raw_label = file_path.parent.name
        normalized = _normalize_with_rules(raw_label)
        if normalized is None or normalized not in TARGET_LABELS:
            continue
        rel = _relative_path_str(root, file_path)
        rows.append((rel, normalized))
    return rows


def write_semicolon_csv(csv_path: Path, rows: Iterable[tuple[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("path;label\n")
        for rel_path, label in rows:
            f.write(f"{rel_path};{label}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a datasets root for videos organized by label folders and "
            "write a semicolon-separated CSV with root-relative paths and normalized labels."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help=(
            "Path to the root directory containing all datasets (e.g., kinetics400_5per). "
            "Paths in CSV will be relative to this root."
        ),
    )
    default_out = Path(__file__).resolve().parent / "labels_and_links.csv"
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=("Output CSV path (semicolon-separated). Default: src/act_rec/data/labels_and_links.csv"),
    )
    parser.add_argument(
        "--ext",
        "--extensions",
        dest="extensions",
        nargs="+",
        default=["mp4", "avi", "mov", "mkv", "webm", "mpg", "mpeg"],
        help="Video file extensions to include (without dots).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print a summary, do not write the CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.root.resolve()
    extensions = {ext.lower().lstrip(".") for ext in args.extensions}

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root directory does not exist or is not a directory: {root}")

    rows = build_csv_rows(root=root, extensions=extensions)

    if args.dry_run:
        total = len(rows)
        by_label: dict[str, int] = {}
        for _, label in rows:
            by_label[label] = by_label.get(label, 0) + 1
        print(f"Found {total} videos matching target labels under {root}")
        for label in sorted(by_label):
            print(f"  {label}: {by_label[label]}")
        return

    out_path: Path = args.out
    write_semicolon_csv(out_path, rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
