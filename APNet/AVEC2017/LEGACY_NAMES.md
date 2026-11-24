# Source file mapping

| Earlier filename | Current file |
|---|---|
| `train_expr_v17_residual_target_strengthening_2017.py` | `branches/aee/train.py` |
| `export_expr_v17_predictions_2017.py` | `branches/aee/export.py` |
| `train_ard_retardation_2017_v5_clean_weaksevere_fixed2.py` | `branches/pae/train.py` |
| `export_ard_v5_predictions_2017.py` | `branches/pae/export.py` |
| `dataset_ard_retardation_2017_v2.py` | `branches/pae/dataset.py` |
| `dataset_ard_retardation_2017_v2_fast.py` | `branches/pae/dataset_fast.py` |
| `model_ard_retardation_2017_v2.py` | `branches/pae/temporal.py` |
| `model_ard_retardation_2017_v3_topk.py` | `branches/pae/pooling.py` |
| `model_ard_retardation_2017_v4_severe.py` | `branches/pae/model.py` |
| `export_branch_embeddings_v27_2017.py` | `export_embeddings.py` |
| `train_routeA_ngrc_v2_checks_2017.py` | `fusion.py` |

The model class names, parameter keys and output CSV schemas are retained for compatibility. Only the final fixed fusion and its required source are included. Historical grid and alternative runs remain in the original archive.

Newly trained checkpoints are named `aee_best.pth` and `pae_best.pth`; the configuration export command still accepts the original PAE checkpoint filename.
