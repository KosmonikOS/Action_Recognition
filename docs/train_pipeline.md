# Training the SODE Classifier

This guide explains how to reproduce the InfoGCN++ training pipeline provided in `experiments/train.ipynb`. It assumes you have already completed the data workflow described in [prepare_data.md](./prepare_data.md) and have preprocessed skeleton segments ready for training.


## 2. Configure training

Open the notebook and edit the cell titled "Paths and label mapping" so the filesystem locations match your assets:

```python
data_root = Path("../data/")
csv_path = data_root / "skeleton_labels.csv"
```

- Point `data_root` to the folder containing the `.npy` segments referenced in the CSV.
- Set `csv_path` to the actual CSV filename you created.
- The cell converts `skeleton_path` entries to absolute paths and builds the `label_to_idx` mapping stored in the checkpoint.

If you plan to execute the notebook with `Run All` (or headlessly), delete or comment out the earlier "Training loop" cell that appears before "Model and optimisation setup". That cell duplicates the real loop but references `optimizer` before it is defined, which would raise a `NameError`.

## 3. Adjust hyperparameters

`train_hparams` collects optimisation, augmentation, and DataLoader settings. Modify the dictionary before you launch training. The default values are the same as in the original InfoGCN++ training script.

| Key | Default | Notes |
| --- | --- | --- |
| `epochs` | `80` | Total passes over the training set. |
| `base_lr` | `1e-2` | Initial learning rate used by the scheduler and optimiser. |
| `optimizer` | `"SGD"` | Descriptive only; the notebook instantiates `AdamW`. |
| `weight_decay` | `1e-4` | L2 regularisation strength. |
| `warmup_epochs` | `5` | Linear warm-up duration at the start of training. |
| `lr_steps` | `[30, 45, 60]` | Epoch indices where the learning rate decays. |
| `lr_decay` | `0.1` | Multiplicative drop applied at each milestone. |
| `grad_clip` | `1.0` | Gradient norm clip used inside the loop. |
| `batch_size` | `32` | Training mini-batch size. |
| `test_batch_size` | `64` | Validation mini-batch size. |
| `num_workers` | `4` | DataLoader workers (set to `0` if multiprocessing causes issues). |
| `prefetch_factor` | `2` | Samples prefetched per worker when `num_workers > 0`. |
| `pin_memory` | `bool(torch.cuda.is_available())` | Enables fast host-to-device transfers. |
| `p_interval_train` | `(0.5, 1.0)` | Temporal crop interval for training (lower bound < 1.0 adds jitter). |
| `p_interval_val` | `(0.95,)` | Deterministic crop interval for validation. |
| `random_rotation` | `True` | Adds spatial rotation augmentation on the fly. |
| `use_velocity` | `False` | Convert coordinates to temporal differences (velocity). |
| `preload` | `True` | Load all skeletons into memory at dataset initialisation. |
| `preload_to_tensor` | `True` | Store preloaded data as tensors (requires `preload=True`). |
| `lambda_cls` | `1.0` | Weight for classification loss. |
| `lambda_recon` | `0.1` | Weight for reconstruction loss. |
| `lambda_feature` | `0.1` | Weight for feature consistency loss. |
| `lambda_kl` | `0.0` | Weight for KL regularisation. |
| `smoothing` | `0.1` | Label smoothing factor. |
| `checkpoint_path` | `"sode_best.pt"` | Relative to the notebook working directory (`experiments/` by default). |

Keep `window_size = 64` in sync with `--window-len` from preprocessing. If you change segment length (for example through `--resample-len`), update both places.

Later in the notebook the SODE model is initialised as:

```python
model = SODE(
    num_class=len(label_to_idx),
    num_point=17,
    num_person=1,
    graph="act_rec.graph.coco.Graph",
    in_channels=3,
    T=window_size,
    n_step=3,
    num_cls=4,
)
```

- `num_point=17` matches the COCO joint layout produced by the preprocessing scripts. Adjust if you adopt a different skeleton topology.
- Increase `num_person` if the dataset contains multi-person clips.
- `n_step` controls how many ODE extrapolations are produced and influences the reconstruction and feature losses (`lambda_recon`, `lambda_feature`).
- `num_cls` decides how many temporal slices feed the classification heads.

## 4. Using trained model

After training, the notebook saves the best model checkpoint to `experiments/sode_best.pt`. You can load it using:

```python
model = torch.load("experiments/sode_best.pt")
```

You can then use the model to make predictions on new data.