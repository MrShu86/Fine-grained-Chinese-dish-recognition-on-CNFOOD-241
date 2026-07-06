# Ablation Logs

This folder contains raw training logs for the four added pairwise/core ablation experiments.

| File | Variant | Best Top-1 | Best Top-5 |
|---|---|---:|---:|
| `KD_MBC_train.log` | KD + MBC | 80.84 | 95.46 |
| `KD_ER_train.log` | KD + ER | 83.10 | 97.40 |
| `MBC_ER_train.log` | MBC + ER | 81.07 | 94.21 |
| `KD_MBC_ER_core_train.log` | KD + MBC + ER | 80.93 | 95.72 |

These logs preserve command lines, training progress, validation metrics, and best-checkpoint messages.
