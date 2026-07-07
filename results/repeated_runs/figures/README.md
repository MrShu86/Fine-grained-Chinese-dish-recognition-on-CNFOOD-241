# Repeated-run Evaluation Figures

This directory contains image-based evaluation evidence for the five-seed repeated experiments.

## Directory Layout

| Directory | Contents |
|---|---|
| `selected/` | Representative baseline/full comparison figures for seed 42 |
| `baseline_seed_1/`, `baseline_seed_25/`, `baseline_seed_42/`, `baseline_seed_50/`, `baseline_seed_100/` | Baseline evaluation figures for each seed |
| `full_seed_1/`, `full_seed_25/`, `full_seed_42/`, `full_seed_50/`, `full_seed_100/` | Full-model evaluation figures for each seed |

## Figure Types

Common figure files include:

| File | Description |
|---|---|
| `class_f1_top50.png` | Classes with the highest F1 scores |
| `class_f1_worst30.png` | Classes with the lowest F1 scores |
| `confusion_matrix_counts.png` | Full confusion matrix using raw counts |
| `confusion_matrix_normalized.png` | Normalized full confusion matrix |
| `confusion_submatrix_worst20.png` | Confusion submatrix for difficult classes |
| `top5_recall_worst30.png` | Worst classes by Top-5 recall |
| `confidence_hist.png` | Prediction confidence histogram |
| `reliability_diagram.png` | Calibration reliability diagram |
| `top_errors_grid.png` / `top_errors_grid_with_text.png` | Representative high-confidence errors |
| `tsne_worstK.png` | t-SNE visualization for difficult classes |

The corresponding metric tables and prediction CSV files are available in `results/repeated_runs/per_seed_eval/`.

