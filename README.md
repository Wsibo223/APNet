# APNet: An Affective-Psychomotor Network for Video-Based Depression Severity Estimation

Official implementation of APNet for video-based depression severity estimation on AVEC2014, AVEC2017, and AVEC2019.

APNet uses two complementary branches to model affective expression evidence (AEE) and psychomotor activity evidence (PAE), followed by a symptom-aware fusion head.

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

Obtain access to the original datasets and annotations from their providers:

- **AVEC2014:** [AVEC2014 Challenge](https://doi.org/10.1145/2661806.2661807) (contact the challenge organizers for corpus access).
- **AVEC2017 (DAIC-WOZ):** [USC ICT database](https://dcapswoz.ict.usc.edu/).
- **AVEC2019 (E-DAIC):** [USC ICT database](https://dcapswoz.ict.usc.edu/), Extended DAIC.

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

## Results and Reproducibility

| Dataset | Training metrics file |
| --- | --- |
| AVEC2014 | `test_metrics.json` |
| AVEC2017 | `fusion/ngrc_summary.json` |
| AVEC2019 | `report.json` |

All inference commands produce `test_metrics.json` and `test_predictions.csv`.

The AVEC2017 release uses an alternate PAE initialization identified retrospectively on the previously evaluated Test split, with the original AEE and fusion settings. Its verified from-scratch result is RMSE 5.3360 and MAE 4.3653; this is a repeatability result, not an independent evaluation or an exact reproduction of the original checkpoint. AVEC2019 reproduced the historical result from scratch (RMSE 5.7743, MAE 4.8321). AVEC2014 uses the historical fixed temporal checkpoint selection and calibration recipe; a complete real-data rerun of the reorganized pipeline is pending. Historical recipes for AVEC2014 and AVEC2019 were retained after Test-informed exploration, so these fixed runs should be interpreted as retrospective reproductions. Numerical results may vary with the software and hardware environment.

## Citation

If you use APNet, please cite *An Affective-Psychomotor Network for Video-Based Depression Severity Estimation*, Hanyuan Wang and Sibo Wang. Full bibliographic information will be added when available.
