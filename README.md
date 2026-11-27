# APNet: An Affective-Psychomotor Network for Video-Based Depression Severity Estimation

Official implementation of APNet for video-based depression severity estimation on AVEC2014, AVEC2017, and AVEC2019.

APNet uses two complementary branches to model affective expression evidence (AEE) and psychomotor activity evidence (PAE), followed by a symptom-aware fusion head.

## Project Structure

The repository contains separate implementations for the three benchmarks, with a consistent training and inference interface.

```text
APNet/                              # repository root
├── README.md
├── .gitignore
└── APNet/                          # implementation root
    ├── requirements.txt
    ├── AVEC2014/
    │   ├── branches/
    │   │   ├── aee/                # manifest, AU encoder, training
    │   │   └── pae/                # pose dataset, temporal model, training
    │   ├── configs/                # branch, temporal and calibration settings
    │   ├── fusion/                 # feature extraction and temporal fusion
    │   ├── calibration/            # fixed final calibration
    │   ├── train.py
    │   └── infer.py
    ├── AVEC2017/
    │   ├── branches/
    │   │   ├── aee/                # AU encoder, training and export
    │   │   └── pae/                # pose dataset, temporal encoder and export
    │   ├── configs/
    │   ├── prepare_data.py
    │   ├── export_embeddings.py
    │   ├── fusion.py
    │   ├── train.py
    │   └── infer.py
    └── AVEC2019/
        ├── branches/
        │   ├── aee/
        │   └── pae/
        ├── configs/
        ├── prepare_data.py
        ├── fusion.py
        ├── train.py
        └── infer.py
```

| File or directory | Purpose |
| --- | --- |
| `branches/aee/` | Action-unit (AU) preprocessing, affective expression representation, and branch prediction. |
| `branches/pae/` | Pose preprocessing, temporal descriptors, and psychomotor activity representation. |
| `configs/` | Dataset-specific model, optimizer, training, and fusion settings. |
| `prepare_data.py` | Normalize split manifests for AVEC2017 and generate compatible AU/pose inputs for AVEC2019. AVEC2014 preparation is included in its training entry point. |
| `fusion/` or `fusion.py` | Integrate branch scores and intermediate representations; training and inference use the respective dataset's fusion implementation. |
| `train.py` | Execute data checks, both branch training stages, feature generation, fusion fitting, and evaluation. |
| `infer.py` | Evaluate with previously fitted model artifacts. |

For compatibility with earlier checkpoints, some internal classes and parameter keys retain their original names. The source filename mappings are documented in [AVEC2014](APNet/AVEC2014/LEGACY_NAMES.md), [AVEC2017](APNet/AVEC2017/LEGACY_NAMES.md), and [AVEC2019](APNet/AVEC2019/LEGACY_NAMES.md).

## Environment

The recommended environment is:

- Linux (a CUDA-capable NVIDIA GPU is recommended for training).
- Python 3.10.
- PyTorch with a CUDA build compatible with the installed NVIDIA driver.
- NumPy, pandas, SciPy, scikit-learn, tqdm, and Matplotlib (see [requirements.txt](APNet/requirements.txt)).
- An isolated Conda environment (Miniconda or Anaconda).

To check the installed versions and GPU availability after installation:

```bash
python --version
python -c "import torch, numpy, sklearn; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('NumPy:', numpy.__version__); print('scikit-learn:', sklearn.__version__)"
nvidia-smi
```

Record the actual package and CUDA versions with your experimental results. The requirements file does not pin a single PyTorch/CUDA combination.

## Installation

```bash
git clone https://github.com/Wsibo223/APNet.git
cd APNet/APNet
conda create -n apnet python=3.10 -y
conda activate apnet
pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=0
```

Use a PyTorch installation compatible with your CUDA version. All remaining commands run from the inner `APNet/` directory.

## Dataset Preparation

### 1. Obtain the Original Datasets

Obtain access to the original datasets and annotations from their providers:

- **AVEC2014:** [AVEC2014 Challenge](https://doi.org/10.1145/2661806.2661807) (contact the challenge organizers for corpus access).
- **AVEC2017 (DAIC-WOZ):** [USC ICT database](https://dcapswoz.ict.usc.edu/).
- **AVEC2019 (E-DAIC):** [USC ICT database](https://dcapswoz.ict.usc.edu/), Extended DAIC.

### 2. Prepare Facial Action Units and Head Pose

If starting from the original video recordings, extract frame-level facial features using [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace). If the corpus already provides compatible OpenFace features, use those files directly. The APNet repository processes the feature CSVs; it does not distribute the original recordings or run the OpenFace video extractor.

Preserve the frame order, timestamps, tracking confidence, and success indicators where available. The AEE branch consumes AU measurements, while the PAE branch processes the six head-pose components (three translations and three rotations). The dataset-specific preprocessing, temporal segmentation, and normalization are implemented in the branch code.

We use facial action units and head-pose features extracted with OpenFace. Keep the original partitions and sample order. Organize the authorized local features and labels as follows:

```text
data/
├── AVEC2014/
│   ├── avec2014_labels.csv
│   ├── train/{Freeform,Northwind}/<pair_id>_<task>_video.csv
│   ├── dev/...
│   └── test/...
├── AVEC2017/
│   ├── features/.../<participant_id>_AUs.csv
│   ├── features/.../<participant_id>_pose.csv
│   └── splits/{train,dev,test}_2017.txt
└── AVEC2019/
    ├── features/<participant_id>_OpenFace2.1.0_Pose_gaze_AUs.csv
    └── labels/{train,dev,test}_split.csv
```

| Dataset | Score | Train / Dev / Test | Annotation format |
| --- | --- | --- | --- |
| AVEC2014 | BDI-II | 50 / 50 / 50 pairs | CSV: `split,pair_id,score` (one row per paired recording) |
| AVEC2017 | PHQ-8 | 107 / 35 / 47 | TXT: AU CSV path and score, separated by whitespace |
| AVEC2019 | PHQ-8 | 163 / 55 / 56 | CSV: `Participant_ID,PHQ_Score` |

AVEC2014 uses matched Freeform and Northwind recordings with OpenFace AU and pose columns. AVEC2017 requires corresponding pose files alongside the AU files; relative paths in split lists resolve against `--data-root`. AVEC2019 supports `--id-col` and `--score-col` to specify alternative label column names. Use paths without spaces.

## Expected Dataset Structure

The examples below use illustrative identifiers and scores only. Replace them with the authorized dataset filenames and annotations; the complete training and evaluation splits must have the sizes listed above.

### AVEC2014

Each split contains one Freeform and one Northwind recording for each pair, and both recordings must share the same BDI-II score.

```text
data/AVEC2014/
├── avec2014_labels.csv
├── train/
│   ├── Freeform/example01_Freeform_video.csv
│   └── Northwind/example01_Northwind_video.csv
├── dev/
│   ├── Freeform/example02_Freeform_video.csv
│   └── Northwind/example02_Northwind_video.csv
└── test/
    ├── Freeform/example03_Freeform_video.csv
    └── Northwind/example03_Northwind_video.csv
```

Each recording CSV contains the AU measurements and head-pose fields `pose_Tx`, `pose_Ty`, `pose_Tz`, `pose_Rx`, `pose_Ry`, and `pose_Rz`. Keep the AU intensity (`_r`) and AU presence (`_c`) columns required by the AVEC2014 AEE loader. The paired recording identifier must match the annotation table. The training launcher builds manifests and checks that every pair has both recording types.

Paths are supplied on the command line:

- `--data-root`: directory containing the `train`, `dev`, and `test` folders.
- `--labels`: BDI-II annotation CSV.
- `--out`: a new experiment directory.

### AVEC2017

Keep the AU and pose files for a participant together, using the same identifier in both filenames. For example:

```text
data/AVEC2017/
├── features/
│   └── 10001_P/
│       ├── 10001_AUs.csv
│       └── 10001_pose.csv
└── splits/
    ├── train_2017.txt
    ├── dev_2017.txt
    └── test_2017.txt
```

The split-list entry points to `10001_P/10001_AUs.csv`. The pose loader resolves the corresponding `10001_pose.csv` automatically. The default AEE input uses AU intensity columns `AU01_r`, `AU02_r`, `AU04_r`, `AU05_r`, `AU06_r`, `AU07_r`, `AU09_r`, `AU10_r`, `AU12_r`, `AU14_r`, `AU15_r`, `AU17_r`, `AU20_r`, and `AU25_r`.

`prepare_data.py` takes the original text manifests (`--splits`) and the base feature directory (`--data-root`), resolves their paths, validates the scores and split sizes, and creates new manifests in `--out` without altering the original data or their order.

### AVEC2019

The feature folder contains combined OpenFace 2.1.0 CSVs, with one file per participant:

```text
data/AVEC2019/
├── features/
│   └── 10002_OpenFace2.1.0_Pose_gaze_AUs.csv
└── labels/
    ├── train_split.csv
    ├── dev_split.csv
    └── test_split.csv
```

Keep the original AU intensity and six pose components. The preprocessing script accepts pose column names with or without the `pose_` prefix and prepares two input directories:

```text
outputs/avec2019_inputs_01/
├── AVEC2019_compat/
│   ├── train_2019.txt
│   ├── dev_2019.txt
│   ├── test_2019.txt
│   └── <participant_id>_AUs.csv
└── AVEC2019_delta_pose_compat/
    ├── train_2019.txt
    ├── dev_2019.txt
    ├── test_2019.txt
    ├── <participant_id>_AUs.csv
    └── <participant_id>_pose.csv
```

The first directory references the original AU features; the second references the same AU data and contains newly generated frame-to-frame pose differences. The generated AU files are symbolic links, so retain the original feature files at their original locations. Both directories preserve the same participants, annotations, and order. Supply them to `train.py` or `infer.py` using `--compat-root` and `--delta-root`.

If you already have the original AVEC2019 text manifests, you can use `--from-compat` instead of `--labels` to preserve their participant order and regenerate both input directories:

```bash
python -B AVEC2019/prepare_data.py \
  --project "$PWD" \
  --from-compat /absolute/path/to/AVEC2019_compat \
  --data "$PWD/data/AVEC2019/features" \
  --out "$PWD/outputs/avec2019_inputs_from_compat_01"
```

## Annotation Files

The authorized annotation files are obtained separately from the original datasets. APNet does not redistribute participant-level depression scores or facial measurements. The following examples illustrate the required local formats.

### AVEC2014: `avec2014_labels.csv`

```csv
split,pair_id,score
train,example01,8
dev,example02,12
test,example03,5
```

The label is shared by the two tasks for the same pair, and valid BDI-II scores range from 0 to 63. The training launcher creates split-specific manifests containing the feature path, score, task, pair identifier, and split. Pair-level results are computed over 50 paired recordings per split.

### AVEC2017: `train_2017.txt`, `dev_2017.txt`, `test_2017.txt`

One feature path and PHQ-8 score per line, separated by whitespace; no header is required:

```text
10001_P/10001_AUs.csv 8
```

The path can be absolute or relative to the feature directory. Each participant belongs to only one split, and the PHQ-8 total must be within 0–24. Preserve the order of the original experiment lists.

### AVEC2019: `train_split.csv`, `dev_split.csv`, `test_split.csv`

```csv
Participant_ID,PHQ_Score
10002,8
```

The participant identifier must correspond to the combined OpenFace filename, and PHQ-8 scores must be within 0–24. Use the provided preparation script rather than renaming or reordering the original splits manually.

## Stage I: Behavioral Branch Training

APNet first trains its two behavioral branches using dataset-specific supervision:

1. **Affective Expression Evidence (AEE).** AU sequences are segmented into temporal windows. The encoder and learnable expression prototypes produce participant-level affective evidence and a branch prediction.
2. **Psychomotor Activity Evidence (PAE).** The six-dimensional head-pose sequence is sanitized and segmented into overlapping windows. The resulting motion descriptors and temporal encoder represent activity patterns and low-activity evidence, producing a branch prediction and intermediate features.

### Configure the Branches

| Dataset | AEE configuration | PAE configuration |
| --- | --- | --- |
| AVEC2014 | `AVEC2014/configs/aee.json` | `AVEC2014/configs/pae.json` |
| AVEC2017 | `AVEC2017/configs/aee.json` | `AVEC2017/configs/pae.json` |
| AVEC2019 | `aee` section in `AVEC2019/configs/train.json` | `pae` section in `AVEC2019/configs/train.json` |

The branch configuration files contain the argument lists used by the training launcher. For instance, the AVEC2017 PAE configuration contains the following settings (excerpt):

```text
--target_hz 10.0
--window_sec 4.0
--stride_sec 2.0
--hidden_dim 128
--num_layers 4
--epochs 80
--batch_size 8
--lr 0.0001
--weight_decay 0.001
```

Use the complete JSON configuration for a reproduction run. The `train.py` launchers initialize the branches, apply the dataset-specific preprocessing, and store the resulting checkpoints under the new output directory. AVEC2014 uses pair-grouped upstream cross-fitting; AVEC2017 and AVEC2019 use their defined Train/Dev/Test partitions. No separate external pre-training is required.

### Run Stage I

Use the dataset-specific [Training from Scratch](#training-from-scratch) commands below. Each launcher automatically runs Stage I and retains the newly trained branch models and their exported evidence for Stage II.

## Stage II: Symptom-Aware Fusion Training

The final readout combines the trained branches' predictions and behavioral evidence. The launchers export the required branch features and fit the dataset-specific fusion model as part of the same run.

| Dataset | Fusion configuration | Released fitting procedure |
| --- | --- | --- |
| AVEC2014 | `configs/temporal_model.json`, `configs/temporal_selection.json`, `configs/calibration.json` | Train the temporal fusion model at its fixed historical setting, then apply the retained calibration rule. |
| AVEC2017 | `configs/fusion.json` | Fit the residual MLP on branch scores and PAE features with the fixed settings; retain the Dev-selected epoch. |
| AVEC2019 | `configs/fusion.json` | Fit the configured regression head on fresh training exports, calibrate on Dev, then evaluate Test. |

Both stages are included in the [Training from Scratch](#training-from-scratch) commands, so there is no need to invoke the internal trainers or exporters individually. For a subsequent evaluation using fitted artifacts, follow [Inference with Trained Models](#inference-with-trained-models).

## Hyperparameter Configuration

The configuration files under the dataset directories store the complete released recipes, including sampling, window size, model dimensions, dropout, optimizer settings, loss weights, and stopping parameters. Selected training settings are summarized below.

| Dataset | Branch | Batch size | Maximum epochs | Learning rate | Weight decay | Seed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AVEC2014 | AEE | 10 | 48 | 5e-5 | 3e-3 | 2026 |
| AVEC2014 | PAE | 8 | 80 | 1e-4 | 1e-3 | 2026 |
| AVEC2017 | AEE | 16 | 120 | 7e-5 | 9e-4 | 2026 |
| AVEC2017 | PAE | 8 | 80 | 1e-4 | 1e-3 | 2015 |
| AVEC2019 | AEE | 16 | 120 | 7e-5 | 9e-4 | 2026 |
| AVEC2019 | PAE | 8 | 80 | 1e-4 | 1e-3 | 2026 |

| Dataset | AEE window / stride (frames) | PAE sampling rate | PAE window / stride (seconds) |
| --- | --- | ---: | --- |
| AVEC2014 | 300 / 150 | 10 Hz | 4.0 / 2.0 |
| AVEC2017 | 300 / 100 | 10 Hz | 4.0 / 2.0 |
| AVEC2019 | 30 / 10 | 15 Hz | 3.0 / 1.5 |

The AVEC2014 temporal fusion recipe specifies a hidden model dimension of 48, batch size 25, learning rate 6e-4, weight decay 1e-3 and seed 2197; its separate selection file fixes the retained run at epoch 4. The AVEC2017 fusion configuration uses hidden size 16, dropout 0.25, batch size 64, learning rate 8e-4, weight decay 1e-2 and a 120-epoch budget. AVEC2019 specifies Ridge regularization `alpha: 1500` and Dev calibration target `1.2`.

The default launchers run one fixed recipe per dataset. They do not run a hyperparameter grid across the Test split or automatically select a new recipe using Test scores. To study alternative settings, work from separate copies of the configurations, use distinct experiment output directories, and retain the exact settings with the results.

## Training from Scratch

Each training command initializes the trainable networks from scratch and fits the final fusion head. No pretrained weights are required. Training configurations are included in each dataset's `configs/` directory. Use a new `--out` directory for every run.

### AVEC2014

```bash
python -B -u AVEC2014/train.py \
  --data-root "$PWD/data/AVEC2014" \
  --labels "$PWD/data/AVEC2014/avec2014_labels.csv" \
  --out "$PWD/outputs/avec2014_run_01"
```

### AVEC2017

```bash
python -B AVEC2017/prepare_data.py \
  --splits "$PWD/data/AVEC2017/splits" \
  --data-root "$PWD/data/AVEC2017/features" \
  --out "$PWD/outputs/avec2017_inputs_01"

python -B -u AVEC2017/train.py \
  --splits "$PWD/outputs/avec2017_inputs_01" \
  --out "$PWD/outputs/avec2017_run_01"
```

### AVEC2019

```bash
python -B AVEC2019/prepare_data.py \
  --project "$PWD" \
  --data "$PWD/data/AVEC2019/features" \
  --labels "$PWD/data/AVEC2019/labels" \
  --out "$PWD/outputs/avec2019_inputs_01"

python -B -u AVEC2019/train.py \
  --project "$PWD" \
  --compat-root "$PWD/outputs/avec2019_inputs_01/AVEC2019_compat" \
  --delta-root "$PWD/outputs/avec2019_inputs_01/AVEC2019_delta_pose_compat" \
  --out "$PWD/outputs/avec2019_run_01"
```

The terminal displays training progress and final Test metrics. Full logs, checkpoints, and predictions are stored in the run directory.

## Inference with Trained Models

Use the corresponding saved model directory (`--run`) and prepared data. The following commands load fixed weights and preprocessing parameters and evaluate the Test split without training, checkpoint selection, or fusion fitting. Results are written to a fresh directory. Checkpoints and dataset-derived artifacts are not bundled in this source repository; retain your own authorized run artifacts. Only load trusted checkpoints.

### AVEC2014

```bash
python -B AVEC2014/infer.py \
  --run "$PWD/outputs/avec2014_run_01" \
  --out "$PWD/outputs/avec2014_infer_01"
```

Requires `full_train/final_model/model.pt` and the prepared `features/` directory from the corresponding run, including `test_features.csv`, `pre_raw_sequence_index.csv`, and the referenced pose sequences. This evaluates the frozen final model on the saved upstream branch context and applies the fixed calibration. It does not rebuild the upstream branch features from raw video. Use `--features` to specify an alternative prepared feature directory.

### AVEC2017

```bash
python -B AVEC2017/infer.py \
  --run "$PWD/outputs/avec2017_run_01" \
  --splits "$PWD/outputs/avec2017_inputs_01" \
  --out "$PWD/outputs/avec2017_infer_01"
```

Requires `aee_training/aee_best.pth`, `pae_training/pae_best.pth`, and `fusion/fusion_model.pth` from the same run. The updated training code saves the fusion weights, feature order, and normalization in `fusion_model.pth`; older runs may lack this file. The inference script exports branch scores and embeddings using the supplied split lists, then runs the frozen fusion head on Test. To reuse existing branch exports, use `--exports` with the corresponding run directory instead of `--splits`.

### AVEC2019

```bash
python -B AVEC2019/infer.py \
  --run "$PWD/outputs/avec2019_run_01" \
  --compat-root "$PWD/outputs/avec2019_inputs_01/AVEC2019_compat" \
  --delta-root "$PWD/outputs/avec2019_inputs_01/AVEC2019_delta_pose_compat" \
  --out "$PWD/outputs/avec2019_infer_01"
```

Requires `aee_training/aee_best.pth`, `pae_training/pae_best.pth`, and `fusion/fusion_model.json` from the same run. The JSON stores the fitted Ridge weights, normalization, and calibration shift. To reuse saved Test exports, pass `--exports` with the corresponding run directory instead of the two input roots.

## GPU Configuration

Select the GPU for a single run with the environment variable:

```bash
export CUDA_VISIBLE_DEVICES=0
```

On a machine with three available GPUs, the datasets can be assigned to separate devices:

| Experiment | Example assignment |
| --- | --- |
| AVEC2014 | GPU 0 |
| AVEC2017 | GPU 1 |
| AVEC2019 | GPU 2 |

This launches independent experiments; it does not merge the datasets into one training split. AVEC2014 uses BDI-II and AVEC2017/2019 use PHQ-8, so the released workflows maintain their separate targets and evaluation protocols. Change the GPU indices to match your hardware.

## Running Experiments in the Background

For long experiments on a Linux server, prepare the inputs as described above and create a separate log directory. The run output directories themselves must not exist before training. Run the following commands from the inner `APNet/` directory.

```bash
mkdir -p "$PWD/outputs/logs"
```

### AVEC2014

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -B -u AVEC2014/train.py \
  --data-root "$PWD/data/AVEC2014" \
  --labels "$PWD/data/AVEC2014/avec2014_labels.csv" \
  --out "$PWD/outputs/avec2014_background_01" \
  > "$PWD/outputs/logs/avec2014.log" 2>&1 &
```

### AVEC2017

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -B -u AVEC2017/train.py \
  --splits "$PWD/outputs/avec2017_inputs_01" \
  --out "$PWD/outputs/avec2017_background_01" \
  > "$PWD/outputs/logs/avec2017.log" 2>&1 &
```

### AVEC2019

```bash
CUDA_VISIBLE_DEVICES=2 nohup python -B -u AVEC2019/train.py \
  --project "$PWD" \
  --compat-root "$PWD/outputs/avec2019_inputs_01/AVEC2019_compat" \
  --delta-root "$PWD/outputs/avec2019_inputs_01/AVEC2019_delta_pose_compat" \
  --out "$PWD/outputs/avec2019_background_01" \
  > "$PWD/outputs/logs/avec2019.log" 2>&1 &
```

Use the corresponding input preparation and output names if you have changed them. Each experiment can run on its own GPU. For another run, choose a different output directory and log filename to retain earlier results and logs.

To follow progress and check GPU utilization:

```bash
tail -f "$PWD/outputs/logs/avec2017.log"
nvidia-smi
```

## Output

The `--out` argument determines the root directory for each training or inference run. The launchers save stage logs and model artifacts in that directory. The principal files and directories are listed below (additional diagnostics may also be produced).

| Dataset | Main training artifacts (relative to `--out`) |
| --- | --- |
| AVEC2014 | `manifests/`, `upstream/`, `features/`, `full_train/final_model/model.pt`, `test_predictions.csv`, `test_metrics.json` |
| AVEC2017 | `aee_training/`, `pae_training/`, `aee_predictions/`, `pae_predictions/`, `aee_embeddings/`, `pae_embeddings/`, `fusion/fusion_model.pth`, `fusion/ngrc_summary.json`, `fusion/selected_test_predictions.csv` |
| AVEC2019 | `aee_training/`, `pae_training/`, `aee_train_dev_exports/`, `pae_train_dev_exports/`, `aee_test_export/`, `pae_test_export/`, `fusion/fusion_model.json`, `fusion/test_predictions.csv`, `report.json`, `report.md` |

Branch training logs and histories record epoch progress, training losses, validation results, and checkpoint information. The fusion outputs contain the final predictions and evaluation statistics; the AVEC2019 report also records elapsed stage durations and the model hash before Test inference. Keep the preprocessing artifacts with their corresponding checkpoints when using the frozen inference entry points.

## Evaluation Metrics

Depression severity is evaluated as a regression task:

- **MAE (Mean Absolute Error):** average absolute difference between predicted and ground-truth scores. Lower is better.
- **RMSE (Root Mean Squared Error):** square root of the mean squared prediction error. Lower is better.
- **CCC (Concordance Correlation Coefficient):** measures agreement between prediction and ground truth, including their correlation, mean, and scale. Higher is better.
- **PCC (Pearson Correlation Coefficient):** measures linear association between predictions and ground truth. Higher is better; PCC is recorded as `corr` in the metric files.

Scores are evaluated on their original scales. AVEC2014 aggregates the two tasks at the paired-recording level; AVEC2017 and AVEC2019 report participant-level PHQ-8 results. Keep the original split membership and evaluation unit when comparing with other methods.

## Results and Reproducibility

| Dataset | Training metrics file |
| --- | --- |
| AVEC2014 | `test_metrics.json` |
| AVEC2017 | `fusion/ngrc_summary.json` |
| AVEC2019 | `report.json` |

All inference commands produce `test_metrics.json` and `test_predictions.csv`.

The AVEC2017 release uses an alternate PAE initialization identified retrospectively on the previously evaluated Test split, with the original AEE and fusion settings. Its verified from-scratch result is RMSE 5.3360 and MAE 4.3653; this is a repeatability result, not an independent evaluation or an exact reproduction of the original checkpoint. AVEC2019 reproduced the historical result from scratch (RMSE 5.7743, MAE 4.8321). AVEC2014 uses the historical fixed temporal checkpoint selection and calibration recipe; a complete real-data rerun of the reorganized pipeline is pending. Historical recipes for AVEC2014 and AVEC2019 were retained after Test-informed exploration, so these fixed runs should be interpreted as retrospective reproductions. Numerical results may vary with the software and hardware environment.

## Notes

- Obtain the original AVEC datasets from the corresponding providers and follow their data-use agreements. Participant-level features, scores, and derived predictions are not redistributed here.
- Prepare the OpenFace inputs and annotation files before training. Their directory structure, filenames, and sample order must agree with the dataset loader.
- Run commands from the inner `APNet/` implementation directory, and use absolute paths without spaces where required by the split manifests.
- Use separate output directories for experiments. Keep the fitted model and its matching feature order, normalization parameters, and calibration together for inference.
- The three dataset workflows are independent. The released configuration and its historical provenance are described above; no common multi-dataset model or joint training protocol is included.
- Frozen evaluation requires the corresponding trained artifacts. AVEC2014 additionally requires the saved upstream feature context and pose sequences; its current inference entry point evaluates the final model on those prepared features.

## Acknowledgments

We thank the AVEC challenge organizers and the DAIC-WOZ/E-DAIC dataset providers for making these resources available for research. We also thank the developers of [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace), [PyTorch](https://pytorch.org/), and [scikit-learn](https://scikit-learn.org/) for the tools used in visual feature extraction, model training, and regression.

## Citation

If you use APNet, please cite *An Affective-Psychomotor Network for Video-Based Depression Severity Estimation*, Hanyuan Wang and Sibo Wang. Full bibliographic information will be added when available.
