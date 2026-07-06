# Reviewer Response Evidence Map

This file maps the main reviewer/editor concerns to concrete files in the repository.

| Concern | Evidence files |
|---|---|
| Code and evaluation scripts must be public | `RegNetY-32GF/`, `R_KDAT/`, `ViT-B16/`, `Swin-B/`, `eval/` |
| Five-seed repeated experiments | `results/repeated_runs/repeated_runs_table.md`, `results/repeated_runs/repeated_runs_per_seed_metrics.csv` |
| Per-seed prediction files for paired tests | `results/repeated_runs/per_seed_eval/*/predictions.csv` |
| McNemar's test, paired bootstrap, Wilcoxon test | `results/significance/significance_summary.md`, `results/significance/significance_summary.csv`, `results/significance/significance_summary.json` |
| Original single-run reviewer statistics | `results/reviewer_stats/summary_metrics.csv`, `results/reviewer_stats/bootstrap_ci.csv`, `results/reviewer_stats/paired_tests.json` |
| Per-class F1 for all 241 categories | `results/reviewer_stats/all_241_per_class_f1.csv` |
| Improved/degraded categories | `results/reviewer_stats/improved_categories.csv`, `results/reviewer_stats/degraded_categories.csv` |
| Full confusion matrices | `results/reviewer_stats/confusion_matrix_baseline.csv`, `results/reviewer_stats/confusion_matrix_full.csv` |
| Pairwise/core ablation experiments | `results/ablation/ablation_summary.md`, `artifacts/ablation_logs/KD_MBC_train.log`, `artifacts/ablation_logs/KD_ER_train.log`, `artifacts/ablation_logs/MBC_ER_train.log`, `artifacts/ablation_logs/KD_MBC_ER_core_train.log` |
| Transformer baselines | `results/transformer_baselines/transformer_baselines_table.md`, `results/transformer_baselines/ViT-B16/`, `results/transformer_baselines/Swin-B/` |
| Training logs/loss trajectories | `artifacts/ablation_logs/`, TensorBoard event files in experiment folders where available |
| Model weights | `artifacts/weights/README.md`; final public release should use GitHub Releases or stable external links |
| Response letter and revision planning | `paper/response_to_reviewers_updated.docx`, `paper/response_to_reviewers.md`, `paper/comment_experiment_mapping.md` |

## How the Evidence Supports the Main Response

1. Reviewer concerns about single-run estimates are addressed by the five-seed repeated-run table.
2. Reviewer concerns about statistical significance are addressed by McNemar's test, paired bootstrap confidence intervals, and Wilcoxon signed-rank testing.
3. Reviewer concerns about component interaction are addressed by the new pairwise/core ablation experiments.
4. Reviewer concerns about class-level degradation are addressed by the complete per-class F1 table, improved/degraded category tables, and confusion matrices.
5. Reviewer concerns about missing Transformer baselines are addressed by ViT-B/16 and Swin-B experiments using 300 x 300 input resolution.
6. Editor concerns about public code availability are addressed by the code folders, evaluation scripts, curated result files, and model-weight release plan.
