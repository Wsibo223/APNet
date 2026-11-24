# AVEC2019

This directory contains the complete training workflow for APNet on the visual features of AVEC2019 (E-DAIC). See the [main README](../README.md) for dataset access and installation.

## Data preparation

Obtain the AVEC2019 visual features and the corresponding PHQ-8 split annotations from the dataset provider. Only the action-unit and head-pose information is used by this configuration. The preparation script accepts combined OpenFace 2.1.0 feature CSVs or an existing split-list layout.

### Option 1: Start from feature and label files

Organize the locally authorized data using the following naming convention:

```text
data/AVEC2019/
├── features/
│   └── <participant_id>_OpenFace2.1.0_Pose_gaze_AUs.csv
└── labels/
    ├── train_split.csv
    ├── dev_split.csv
    └── test_split.csv
```

The label files must contain the participant identifier and PHQ-8 total score. The default column names are `Participant_ID` and `PHQ_Score`; rename the local copies of the files to the above basenames and use `--id-col` / `--score-col` if the column names in your distribution differ. Preserve the original participant order and split membership.

From the repository root (a path without spaces), run:

```bash
python -B AVEC2019/prepare_data.py \
  --project "$PWD" \
  --data "$PWD/data/AVEC2019/features" \
  --labels "$PWD/data/AVEC2019/labels" \
  --out "$PWD/outputs/avec2019_inputs_01"
```

### Option 2: Start from existing split lists

If your data already uses the original split-list format, each of `train_2019.txt`, `dev_2019.txt` and `test_2019.txt` contains one feature CSV path and one PHQ-8 score per line (whitespace-separated). To preserve the original split order and build the corresponding AU and delta-pose inputs:

```bash
python -B AVEC2019/prepare_data.py \
  --project "$PWD" \
  --from-compat /absolute/path/to/AVEC2019_compat \
  --data /absolute/path/to/combined_openface_features \
  --out "$PWD/outputs/avec2019_inputs_01"
```

This option uses the original adjacent pose CSVs when available, falling back to combined OpenFace feature CSVs. Both options make symbolic links to existing AU features, write new pose-difference files and create two compatibility directories:

```text
outputs/avec2019_inputs_01/
├── AVEC2019_compat/
└── AVEC2019_delta_pose_compat/
```

The reproduced configuration uses 163 Train, 55 Dev and 56 Test participants. If your local split or valid feature count differs, verify the dataset version and the applicable splits. The implementation checks counts and alignment; it does not arbitrarily remove participants to match the historical sample counts.

## Training and evaluation

```bash
python -B AVEC2019/train.py \
  --project "$PWD" \
  --compat-root "$PWD/outputs/avec2019_inputs_01/AVEC2019_compat" \
  --delta-root "$PWD/outputs/avec2019_inputs_01/AVEC2019_delta_pose_compat" \
  --out "$PWD/outputs/avec2019_run_01"
```

Both branches are trained from random initialization with the fixed seed and settings in [configs/train.json](configs/train.json). The fusion model is fitted from the newly exported training features with the settings in [configs/fusion.json](configs/fusion.json). The Dev split is used for checkpoint selection and calibration. The Test split is evaluated after the fitted components are fixed.

The terminal reports training loss, learning rate and estimated remaining time, followed by the final Test metrics. Full run records are written to `report.json` and the stage logs.

## Code structure

| Path | Description |
| --- | --- |
| `train.py` | Single entry point for the two branches and fusion |
| `prepare_data.py` | Prepare AU and delta-pose inputs |
| `fusion.py` | Fit the Ridge head and evaluate the fixed model |
| `branches/aee/train.py` | AEE model, anchor and training loop |
| `branches/aee/export.py` | Export AEE predictions and features |
| `branches/pae/train.py` | PAE training and checkpoint selection |
| `branches/pae/model.py` | PAE network |
| `branches/pae/temporal.py` | Temporal encoder blocks |
| `branches/pae/pooling.py` | Masked Top-k evidence pooling |
| `branches/pae/dataset.py`, `dataset_fast.py` | Shared pose processing and the fast pose-only dataset |
| `branches/pae/export.py` | Export PAE predictions and embeddings |
| `configs/` | Fixed training and fusion configuration |

Several internal names (such as ARD and V17) are retained from earlier versions of the implementation. The 2019 entry point supplies AVEC2019 inputs and settings and does not reuse AVEC2017 weights. See [the original-to-current file mapping](LEGACY_NAMES.md).

## Reproducibility

The fixed recipe reproduces the historical AVEC2019 experiment. Its fusion configuration was retained from an earlier comparison of candidates on the same Test split. No candidate search is performed in the released entry point. Results obtained by rerunning this recipe on the same split are historical reproduction results, not an independent evaluation.

The output directory must not exist before running. The code does not overwrite an earlier experiment. Keep the output directory local because it contains individual predictions and label-derived artifacts.
