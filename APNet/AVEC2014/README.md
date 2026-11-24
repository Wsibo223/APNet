# AVEC2014

This implementation reconstructs the selected historical AVEC2014 pipeline using the source code of both upstream branches, the shared temporal readout and its fixed calibration. No trained model or participant prediction files are distributed.

## Data preparation

Obtain the corpus and annotations using the instructions in the [main README](../README.md). Prepare OpenFace CSVs containing timestamps, confidence, success, facial action-unit intensity/presence and six pose columns (`pose_Tx`, `pose_Ty`, `pose_Tz`, `pose_Rx`, `pose_Ry`, `pose_Rz`).

```text
AVEC2014/
├── avec2014_labels.csv
├── train/{Freeform,Northwind}/<pair_id>_<task>_video.csv
├── dev/{Freeform,Northwind}/<pair_id>_<task>_video.csv
└── test/{Freeform,Northwind}/<pair_id>_<task>_video.csv
```

Use a labels CSV with columns `split,pair_id,score`, one row per paired recording. The BDI-II score must be in [0,63], with 50 matched Freeform/Northwind pairs per split.

## Training and evaluation

```bash
python -B -u AVEC2014/train.py \
  --data-root "$PWD/data/AVEC2014" \
  --labels "$PWD/data/AVEC2014/avec2014_labels.csv" \
  --out "$PWD/outputs/avec2014_run_01"
```

The entry point performs the following stages:
1. Create and validate the split manifests.
2. Train the AEE branch from random initialization using five grouped folds (48 epochs per fold).
3. Train the PAE branch from random initialization using five grouped folds (up to 80 epochs per fold).
4. Export AEE embeddings, upstream prediction context and raw 71-dimensional pose descriptors.
5. Initialize and train the selected temporal encoder with its six internal evidence heads.
6. Apply the fixed final calibration and compute Test metrics.

The temporal stage uses the archived trial 23 architecture (seed 2197, selected epoch 4); upstream branches use seed 2026. Instead of reranking configurations on Test, this entry point stops at the historically fixed epoch. The earlier search used Test RMSE for model/epoch selection, and the final calibration used frozen priors derived from earlier Test-guided exploration before the Dev selector. The fixed calibration is therefore a historical replay rather than an independently selected protocol.

| Configuration file | Contents |
| --- | --- |
| `configs/aee.json` | Original AEE training flags |
| `configs/pae.json` | Original upstream PAE training flags |
| `configs/temporal_model.json` | Selected temporal network and optimizer |
| `configs/temporal_selection.json` | Fixed model/epoch provenance |
| `configs/calibration.json` | Frozen output calibration |

All outputs are written under the new run directory, including stage logs, `upstream/`, `features/`, `full_train/final_model/` and `test_metrics.json`.

The archived final result was MAE **4.9523**, RMSE **6.5432** (50 Test pairs). This restructured end-to-end entry point still requires a full real-data validation. Historical search scripts remain available in their respective directories for provenance, but the normal training entry point does not run them.

The top-level training command displays epoch progress, training loss, learning rate and an estimated time to the epoch limit, followed by the final Test metrics. Development and fold-validation results remain in the complete stage logs and saved training histories; the training and selection protocols are unchanged.

See [LEGACY_NAMES.md](LEGACY_NAMES.md) for the source file mapping.
