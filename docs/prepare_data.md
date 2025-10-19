# End-to-End Data Pipeline

This guide explains how to assemble the full video-to-skeleton pipeline using the scripts in `data/scripts`.

All commands below assume you run them from the project root.

---

## 1. Download and Stage Raw Datasets

1. **UCF101**  
   - Download: https://www.crcv.ucf.edu/data/UCF101.php  
   - Extract so that the videos live under `data/ucf101/UCF-101/`. The root you pass to the preparation script must be the directory that directly contains the action-class subfolders.

2. **Kinetics400 (5% subset)**  
   - Download from Kaggle: https://www.kaggle.com/datasets/rohanmallick/kinetics-train-5per  
   - Place the extracted content (e.g., `train/`, `val/`, `test/`) under `data/kinetics400_5per/`.

3. **Penn Action**  
   - Download: http://dreamdragon.github.io/PennAction/  
   - Extract under `data/penn_action/` so that you have `data/penn_action/frames/` and `data/penn_action/labels/`. The script will create `videos_mp4/` next to them.

---

## 2. Prepare Raw Video CSVs

Run each preparation script once the datasets are staged. Every script writes a CSV named `labels_and_links.csv` inside the supplied `--root` (unless you override `--out`). Paths in the CSV are relative to the CSV location.

### `prepare_ucf.py`

```bash
python data/scripts/prepare_ucf.py \
  --root data/ucf101
```

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--root PATH` | ✔️ | – | Dataset root that contains the class folders (e.g., `UCF-101/`). |
| `--ext / --extensions` |  | `avi mp4 mov mkv webm mpg mpeg` | Video filename extensions (without dots). |
| `--out PATH` |  | `<root>/labels_and_links.csv` | Output CSV path; created if missing. |
| `--norm PATH` |  | `label_normalization.json` | Normalisation rules for labels. |
| `--dry-run` |  | `False` | Only print counts per label (no CSV written). |

### `prepare_kinetics.py`

```bash
python data/scripts/prepare_kinetics.py \
  --root data/kinetics400_5per
```

Arguments mirror `prepare_ucf.py` (different dataset root).

### `prepare_penn.py`

```bash
python data/scripts/prepare_penn.py \
  --root data/penn_action \
  --fps 30
```

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--root PATH` | ✔️ | – | Folder containing `frames/` and `labels/`. |
| `--fps INT` |  | `30` | Frame rate used when encoding MP4 videos. |
| `--overwrite` |  | `False` | Rebuild MP4 files even if they already exist. |
| `--out PATH` |  | `<root>/labels_and_links.csv` | Target CSV. |
| `--norm PATH` |  | `label_normalization.json` | Normalisation rules. |
| `--dry-run` |  | `False` | Summarise without writing files. |

The script creates MP4s in `<root>/videos_mp4/` and references them in the CSV.

---

## 3. Augment the Videos

Combine the three CSVs from Step 2 and create augmented videos.

```bash
python data/scripts/augment_videos.py \
  --input-csvs \
    data/ucf101/labels_and_links.csv \
    data/kinetics400_5per/labels_and_links.csv \
    data/penn_action/labels_and_links.csv \
  --output data/augmented/annotations.csv \
  --augmentation-dir data/augmented/videos \
  --augmentations flip color_jitter speed_change \
  --seed 42
```

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--input-csvs PATH [PATH ...]` | ✔️ | – | One or more CSVs with `video_path`/`label`. Relative paths are resolved against each CSV’s folder. |
| `--output PATH` | ✔️ | – | File path for the combined augmented CSV. If a directory is supplied, `augmented_annotations.csv` is created inside it. |
| `--augmentation-dir PATH` | ✔️ | – | Directory where augmented videos are stored (subdirectories per augmentation). |
| `--augmentations` |  | `flip color_jitter speed_change` | Subset of augmentations to run. |
| `--seed INT` |  | `42` | Random seed for reproducibility. |

The resulting CSV records relative paths (to the output directory) of the augmented videos alongside their labels and the augmentation type.

---

## 4. Extract Skeletons with YOLO Pose

Download or prepare a YOLO pose model checkpoint compatible with `act_rec.labeling.YoloPoseVideoLabeler`. Place it under `models/yolo-pose.pt` (or another path of your choice).

```bash
python data/scripts/extract_skeletons.py \
  --input-csvs data/augmented/annotations.csv \
  --output data/processed/skeletons.csv \
  --model-path models/yolo-pose.pt \
  --skeleton-dir skeletons_raw
```

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--input-csvs PATH [PATH ...]` | ✔️ | – | CSVs whose videos should be processed (use the augmented CSV from Step&nbsp;3). |
| `--output PATH` | ✔️ | – | CSV to save extracted metadata (`video_path`, `label`, `n_frames`, `skeleton_path`). |
| `--model-path PATH` | ✔️ | – | YOLO pose weights file. |
| `--skeleton-dir PATH` | ✔️ | – | Directory (relative to `output`’s parent or absolute) where `.npy` skeleton files are written. |

The script keeps skeleton paths relative to `data/processed/` (the parent of `--output` in the example). Ensure the YOLO model and runtime dependencies (GPU drivers, if applicable) are in place.

---

## 5. Preprocess Skeletons

Transform raw `.npy` skeletons into windowed, normalised segments ready for training.

```bash
python data/scripts/preprocess_skeletons.py \
  --input-csvs data/processed/skeletons.csv \
  --output data/processed/preprocessed_skeletons.csv \
  --out-skeleton-dir data/processed/preprocessed_skeletons \
  --conf-thr 0.20 \
  --window-len 64 \
  --window-stride 64 \
  --tail-policy keep
```

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--input-csvs PATH [PATH ...]` | ✔️ | – | CSVs from Step&nbsp;4 containing skeleton metadata. |
| `--output PATH` | ✔️ | – | Final CSV describing preprocessed skeleton segments. |
| `--out-skeleton-dir PATH` | ✔️ | – | Directory for the processed segments (`.npy`); created if absent. |
| `--conf-thr FLOAT` |  | `0.20` | Minimum joint confidence before interpolation. |
| `--window-len INT` |  | `64` | Sliding window length (frames). |
| `--window-stride INT` |  | `64` | Step size between windows. |
| `--tail-policy {keep,pad,drop}` |  | `keep` | How to handle trailing frames shorter than a full window. |
| `--resample-len INT` |  | – | If set, uniformly resample each window to this length. |
| `--fail-missing` |  | `False` | Abort if any skeleton file is missing or invalid. |

The output CSV contains one row per segment with `skeleton_path` relative to the CSV directory.

---

## 6. Final Check

After the steps above, you should have:

- `data/processed/preprocessed_skeletons.csv` (or your chosen `--output` path) listing every labelled segment.
- A directory such as `data/processed/preprocessed_skeletons/` holding the segment `.npy` files referenced in the CSV.

Use these assets for downstream training or evaluation. Keep the intermediate CSVs and videos if you plan to re-run augmentation or skeleton extraction with different options.
