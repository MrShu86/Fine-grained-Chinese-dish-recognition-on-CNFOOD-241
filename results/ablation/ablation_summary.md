# Pairwise and Core Ablation Summary

These experiments were added to address the reviewer concern that isolated component ablations alone do not establish component interaction.

All variants below used seed 42 and disabled strong augmentation to isolate the interaction among KD, margin-based classification (MBC/ArcFace), and embedding regularization (ER/triplet).

| Log file | Variant | Command-level setting | Top-1 | Top-5 | Best checkpoint path reported in log |
|---|---|---|---:|---:|---|
| `artifacts/ablation_logs/KD_MBC_train.log` | KD + MBC | `--kd-use --arcface --no-triplet --no-strong-aug` | 80.84 | 95.46 | `ablation_runs_seed42_serverA/kd_mbc/seed_42/best.pt` |
| `artifacts/ablation_logs/KD_ER_train.log` | KD + ER | `--kd-use --no-arcface --triplet --no-strong-aug` | 83.10 | 97.40 | `ablation_runs_seed42_serverA/kd_er/seed_42/best.pt` |
| `artifacts/ablation_logs/MBC_ER_train.log` | MBC + ER | `--no-kd --arcface --triplet --no-strong-aug` | 81.07 | 94.21 | `ablation_runs_seed42_serverB/mbc_er/seed_42/best.pt` |
| `artifacts/ablation_logs/KD_MBC_ER_core_train.log` | KD + MBC + ER | `--kd-use --arcface --triplet --no-strong-aug` | 80.93 | 95.72 | `ablation_runs_seed42_serverB/core/seed_42/best.pt` |

Interpretation used in the response letter:

- The component effects are not purely additive.
- KD + ER was the strongest reduced combination under this diagnostic setting.
- MBC/ArcFace-containing reduced variants were more sensitive when strong augmentation was disabled.
- The revised manuscript should describe the method as an integrated training recipe with context-dependent complementarity rather than claiming uniform coordinated synergy.
