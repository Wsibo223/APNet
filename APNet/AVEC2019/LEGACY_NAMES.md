# Source file names

Earlier versions of APNet used the names expression, ARD and numbness for the two visual branches. The AVEC2019 implementation reuses those modules, with the 2019 data and experiment settings passed by the training entry point. The five PAE model/data modules below were identical in the archived 2017 and 2019 packages before this organizational change.

| Previous Python file name | Current path |
| --- | --- |
| `train_expr_v17_residual_target_strengthening_2017.py` | `branches/aee/train.py` |
| `export_expr_v17_predictions_2017.py` | `branches/aee/export.py` |
| `train_ard_retardation_2017_v5_clean_weaksevere_fixed2_reverse_select.py` | `branches/pae/train.py` |
| `train_ard_retardation_2017_v5_clean_weaksevere_fixed2.py` | Its two dataset construction helpers were identical to those in the active trainer, so they are now imported from `branches/pae/train.py`. |
| `export_numb_v5_2019_for_fusion.py` | `branches/pae/export.py` |
| `model_ard_retardation_2017_v2.py` | `branches/pae/temporal.py` |
| `model_ard_retardation_2017_v3_topk.py` | `branches/pae/pooling.py` |
| `model_ard_retardation_2017_v4_severe.py` | `branches/pae/model.py` |
| `dataset_ard_retardation_2017_v2.py` | `branches/pae/dataset.py` |
| `dataset_ard_retardation_2017_v2_fast.py` | `branches/pae/dataset_fast.py` |

The supported top-level commands are `AVEC2019/train.py` and `AVEC2019/prepare_data.py`. The fixed settings are stored in `configs/train.json` and `configs/fusion.json`. The new training entry point writes `aee_best.pth` and `pae_best.pth`; it does not load earlier AVEC2017 or AVEC2019 weights.
