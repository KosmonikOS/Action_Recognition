import json
import re
from collections.abc import Iterator
from pathlib import Path

import cv2


def normalize_text_basic(text: str) -> str:
    """Convert arbitrary text to a snake_case normalization key.

    Args:
        text: Raw text (e.g., a label name) to normalize.

    Returns:
        Snake_case string intended for label normalization lookup, consisting
        only of lowercase [a-z0-9] and single underscores between tokens.
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


def load_normalization(norm_path: Path) -> tuple[set[str], dict[str, str]]:
    """Load target labels and synonyms mapping from a JSON file.

    Args:
        norm_path: Path to JSON with fields `target_labels` and `synonyms`.

    Returns:
        A tuple of (target_labels, synonyms_dict).
    """
    with norm_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    target = set(data.get("target_labels", []))
    synonyms = dict(data.get("synonyms", {}))
    return target, synonyms


def normalize_with_rules(raw_label: str, synonyms: dict[str, str], target: set[str]) -> str | None:
    """Normalize a raw label to one of the target labels, if possible.

    Combines direct synonym lookup with a few heuristic fallbacks for
    common phrasing variations.

    Args:
        raw_label: Original label string.
        synonyms: Mapping of normalization keys to canonical labels.
        target: Allowed set of target labels.

    Returns:
        The normalized label if recognized and allowed, otherwise None.
    """
    key = normalize_text_basic(raw_label)
    normalized: str | None = None
    if key in synonyms:
        normalized = synonyms[key]
    else:
        tokens = set(key.split("_"))
        if {"running", "treadmill"}.issubset(tokens) or {"run", "treadmill"}.issubset(tokens):
            normalized = "running_on_treadmill"
        elif ("snatch" in tokens) and ("weightlifting" in tokens or ("weight" in tokens and "lifting" in tokens)):
            normalized = "snatch_weight_lifting"
        elif ("push" in tokens) and ("up" in tokens or "ups" in tokens):
            normalized = "pushup"
        elif ("pull" in tokens) and ("up" in tokens or "ups" in tokens):
            normalized = "pullup"
        else:
            normalized = None

    if normalized is not None and normalized in target:
        return normalized
    return None


def iter_video_files(root: Path, extensions: set[str]) -> Iterator[Path]:
    """Recursively yield video files under root matching given extensions."""
    for ext in extensions:
        yield from root.rglob(f"*.{ext}")


def relative_path_str(root: Path, file_path: Path) -> str:
    """Return file path as POSIX-style path relative to root."""
    return file_path.relative_to(root).as_posix()


def relative_path_from_base(base_dir: Path, file_path: Path) -> str:
    """Return file path as POSIX-style path relative to base_dir.

    Args:
        base_dir: The directory relative to which paths should be computed.
        file_path: The absolute or nested path to relativize.

    Returns:
        POSIX relative string. If `file_path` is not under `base_dir`, this
        will raise a ValueError; callers should ensure the relationship or
        choose an appropriate base.
    """
    return file_path.relative_to(base_dir).as_posix()


def write_semicolon_csv(csv_path: Path, rows: list[tuple[str, str]]) -> None:
    """Write a semicolon-separated CSV of relative paths and labels."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("video_path, label\n")
        for rel_path, label in rows:
            f.write(f"{rel_path}, {label}\n")


def collect_frame_files(frames_dir: Path) -> list[Path]:
    """Collect frame image files for a single video in natural order."""
    jpgs = sorted(frames_dir.glob("*.jpg"))
    if jpgs:
        return jpgs
    pngs = sorted(frames_dir.glob("*.png"))
    return pngs


def build_video_from_frames(frames: list[Path], out_path: Path, fps: int) -> bool:
    """Build an MP4 video from a sequence of frames."""
    if not frames:
        return False
    first = cv2.imread(str(frames[0]))
    if first is None:
        return False
    height, width = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))
    try:
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            writer.write(frame)
    finally:
        writer.release()
    return out_path.exists() and out_path.stat().st_size > 0
