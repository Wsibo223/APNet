# APNet: An Affective-Psychomotor Network for Video-Based Depression Severity Estimation

This repository contains the visual-only implementation of **APNet** for depression severity estimation on the AVEC2014, AVEC2017 and AVEC2019 benchmarks.

APNet models two types of behavioral evidence using distinct temporal approaches:

- **Affective Expression Evidence (AEE):** facial action-unit patterns and their recurrent expression dynamics.
- **Psychomotor Activity Evidence (PAE):** head-pose dynamics and low-activity evidence.
- **Fusion:** integration of branch predictions and behavioral features using the dataset-specific readout.

## Project Structure

```text
APNet/
├── AVEC2014/
│   ├── branches/
│   │   ├── aee/
│   │   └── pae/
│   ├── fusion/
│   ├── calibration/
│   ├── configs/
│   ├── train.py
│   └── README.md
├── AVEC2017/
│   ├── branches/
│   │   ├── aee/
│   │   └── pae/
│   ├── configs/
│   ├── prepare_data.py
│   ├── export_embeddings.py
│   ├── fusion.py
│   ├── train.py
│   └── README.md
├── AVEC2019/
│   ├── branches/
│   │   ├── aee/
│   │   └── pae/
│   ├── configs/
│   ├── prepare_data.py
│   ├── fusion.py
│   ├── train.py
│   └── README.md
├── requirements.txt
└── README.md
```

- `branches/`: models, data processing and training code for the two evidence branches.
- `configs/`: dataset-specific experimental settings.
- `train.py`: sequential training and evaluation entry point.
- `prepare_data.py`: conversion of local dataset inputs (AVEC2017 and AVEC2019).
- `fusion/` or `fusion.py`: final prediction readout; AVEC2014 additionally uses a fixed calibration rule.

Each dataset directory includes its own detailed instructions and a mapping of historical source file names.

## Environment

We recommend a Linux environment with Python 3.10 and a PyTorch installation compatible with the available NVIDIA GPU and CUDA version.

```bash
conda create -n apnet python=3.10
conda activate apnet
pip install -r requirements.txt
```

Record your Python, PyTorch, CUDA, NumPy and scikit-learn versions when reproducing results. The archived experiments used seed 2026; numerical results can vary across software and hardware environments.

## Dataset Preparation

### 1. Obtain the datasets

Apply for access through the original dataset providers:

- **AVEC2014:** [AVEC2014 challenge paper](https://doi.org/10.1145/2661806.2661807). Contact the challenge organizers for current corpus access instructions.
- **AVEC2017 (DAIC-WOZ):** [USC ICT database website](https://dcapswoz.ict.usc.edu/), DAIC-WOZ application.
- **AVEC2019 (E-DAIC):** [USC ICT database website](https://dcapswoz.ict.usc.edu/), Extended DAIC application.

The corpus, features and annotations must be obtained separately under the applicable data-use agreements.

### 2. Prepare visual features

This repository uses facial action units and head pose extracted by OpenFace. AVEC2014 consumes paired Freeform and Northwind recordings, AVEC2017 consumes AU files with corresponding pose files, and AVEC2019 uses combined OpenFace 2.1.0 CSVs. Preserve the original dataset partitions and the effective sample order.

## Expected Dataset Structure

Organize your authorized local files as follows (paths can be set on the command line):

```text
data/
├── AVEC2014/
│   ├── avec2014_labels.csv
│   ├── train/
│   │   ├── Freeform/<pair_id>_Freeform_video.csv
│   │   └── Northwind/<pair_id>_Northwind_video.csv
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

See the dataset-specific guides for accepted layouts and field names.

## Annotation Files

| Dataset | Target | Effective Train / Dev / Test |
| --- | --- | --- |
| AVEC2014 | BDI-II | 50 / 50 / 50 paired recordings |
| AVEC2017 | PHQ-8 | 107 / 35 / 47 participants |
| AVEC2019 | PHQ-8 | 163 / 55 / 56 participants |

AVEC2014 uses a CSV containing `split`, `pair_id` and `score`. AVEC2017 uses whitespace-separated AU path and score lists. AVEC2019 uses split CSVs containing participant identifiers and scores.

## Stage I: Training the Two Branches

The branch training scripts initialize the models from scratch and learn the respective behavioral representations. AVEC2014 uses the archived out-of-fold branch workflow and paired-view aggregation. AVEC2017 uses its original safe AEE and weak-severe PAE configurations. AVEC2019 uses its own fixed AEE and PAE settings.

## Stage II: Training the Fusion Head

The branches are trained and their predictions and features are exported before the final readout is fitted.

- AVEC2014: shared temporal encoder with six internal evidence heads and fixed piecewise-affine calibration.
- AVEC2017: residual MLP readout operating on the branch scores and PAE features.
- AVEC2019: fitted Ridge regression using the branch-score context and PAE embedding features.

## Hyperparameter Configuration

Training settings are stored in the corresponding `configs/` directory. Use the dataset-specific configuration files rather than copying settings between datasets. The AVEC2017 PAE training arguments are included in `AVEC2017/configs/pae.json`, recovered from the original training checkpoint metadata.

## Training and Evaluation

Run the commands below from the repository root. Each `--out` must be a new path.

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

The training entry point uses the included configuration files and initializes both branches and the fusion model anew.

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

Dataset-specific instructions: [AVEC2014](AVEC2014/README.md) · [AVEC2017](AVEC2017/README.md) · [AVEC2019](AVEC2019/README.md).

## GPU Configuration

```bash
export CUDA_VISIBLE_DEVICES=0
```

## Running Experiments in the Background

For example, on a remote server:

```bash
mkdir -p "$PWD/outputs"
job=$(mktemp -d "$PWD/outputs/avec2014_job_XXXXXX")
nohup python -B -u AVEC2014/train.py \
  --data-root "$PWD/data/AVEC2014" \
  --labels "$PWD/data/AVEC2014/avec2014_labels.csv" \
  --out "$job/result" > "$job/train.log" 2>&1 &
tail -f "$job/train.log"
```

Monitor the GPU with `nvidia-smi`. The training entry points display training progress and final Test metrics. Validation metrics remain available in the complete stage logs and saved histories. Each run directory contains stage logs, generated checkpoints and locally derived predictions.

## Output

| Dataset | Final metrics |
| --- | --- |
| AVEC2014 | `test_metrics.json` |
| AVEC2017 | `fusion/ngrc_summary.json` |
| AVEC2019 | `report.json`, `report.md` |

## Evaluation Metrics

- **MAE:** Mean Absolute Error.
- **RMSE:** Root Mean Square Error.
- **CCC:** Concordance Correlation Coefficient.
- **PCC:** Pearson Correlation Coefficient.

## Notes

- This repository includes source code and configuration files only. Dataset files, participant-level outputs and pretrained checkpoints are excluded.
- AVEC2014 and AVEC2019 use fixed recipes derived from historical experiments on the same Test split. The AVEC2014 upstream temporal checkpoint was originally selected using Test RMSE and the calibration uses historically Test-informed priors. The fixed replay is retrospective and is not an independent estimate of generalization.
- The restructured AVEC2017 and AVEC2014 workflows require validation with the authorized real datasets. The AVEC2019 archived from-scratch run reproduced its historical result; the renamed package also requires a local real-data run.
- Keep generated data and experiment outputs local under `outputs/`.
- Data usage remains subject to the original provider agreements.

## Citation

Please cite *An Affective-Psychomotor Network for Video-Based Depression Severity Estimation* if you use this work. Bibliographic information will be added when available.
