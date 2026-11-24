# AVEC2017

This directory contains the AEE and PAE training source, prediction and embedding exporters, and the fixed residual MLP fusion implementation for the archived AVEC2017 configuration.

## Data preparation

Obtain DAIC-WOZ through the original dataset provider. Preserve the original participant partitions and order. Prepare three split lists, `train_2017.txt`, `dev_2017.txt`, `test_2017.txt`, containing an AU CSV path and a PHQ-8 score separated by whitespace.

The data loader expects the corresponding head-pose CSV alongside the AU file (e.g. `<participant_id>_AUs.csv` and `<participant_id>_pose.csv`, or the equivalent AU/pose subdirectories). Use the original files and preprocessing from the reference experiment.

```bash
python -B AVEC2017/prepare_data.py \
  --splits /absolute/path/to/original_split_lists \
  --data-root /absolute/path/to/visual_features \
  --out "$PWD/outputs/avec2017_inputs_01"
```

`--data-root` is the base directory for relative AU paths in the original lists. This step only creates new lists with absolute paths; it does not modify source features or reorder samples.

## Training configuration

The original AEE training arguments, PAE checkpoint hyperparameters and final residual MLP settings are included in `configs/aee.json`, `configs/pae.json` and `configs/fusion.json`, respectively. The training entry point reads these configurations directly. No historical checkpoint is needed.

## Training and evaluation

```bash
python -B -u AVEC2017/train.py \
  --splits "$PWD/outputs/avec2017_inputs_01" \
  --out "$PWD/outputs/avec2017_run_01"
```

The workflow trains both branches from random initialization, exports scores and embeddings, and fits the residual MLP. The final fixed configuration preserves the original legacy RNG replay, feature ranking, model size and Dev-selected epoch protocol.

| Setting | Archived configuration |
| --- | --- |
| AEE | 120 maximum epochs; hidden dimension 64; batch size 16 |
| PAE | 80 maximum epochs; hidden dimension 128; 4 layers; batch size 8; learning rate 1e-4 |
| Residual MLP | 136 ranked inputs, hidden size 16, base weight 0.25, dropout 0.25 |
| Seed | 2026 |

The final aggregate metrics are saved in `fusion/ngrc_summary.json`. The archived fixed run reported Test MAE **4.4571**, RMSE **5.3392**, CCC **0.4177**, PCC **0.5683** (47 participants). Real-data validation after restructuring is still pending.

The top-level training command displays epoch progress, training loss, learning rate and an estimated time to the epoch limit, followed by the final Test metrics. Development and fold-validation results remain in the complete stage logs and saved training histories; the training and selection protocols are unchanged.

See [LEGACY_NAMES.md](LEGACY_NAMES.md) for the former file names.
