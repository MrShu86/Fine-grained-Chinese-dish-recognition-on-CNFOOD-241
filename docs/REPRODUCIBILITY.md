# Reproducibility Guide

This guide summarizes how to reproduce the main experimental evidence used in the revised manuscript and response letter.

## Environment

The experiments were run with Python 3.8, PyTorch, torchvision, CUDA GPUs, and mixed precision. On the local Windows machine, analysis scripts can be run with the `opencv_py38` conda environment.

Server examples used during revision:

```bash
conda activate opencv_py38
```

or, on the training server:

```bash
python --version
nvidia-smi
```

## Dataset

CNFOOD-241 is expected to be arranged as class folders:

```text
CNFOOD-241/
  train600x600/
    class_001/
    ...
  val600x600/
    class_001/
    ...
```

The repository does not redistribute the dataset.

## Baseline Repeated Runs

Baseline code:

```text
RegNetY-32GF/
```

Repeated-run launcher:

```bash
bash run_repeated_experiments_server.sh baseline
```

The five seeds used in the revision were:

```text
1, 25, 42, 50, 100
```

Curated outputs are stored in:

```text
results/repeated_runs/
```

## Full Model Repeated Runs

Full model code:

```text
R_KDAT/
```

Repeated-run launcher:

```bash
bash run_repeated_experiments_server.sh full
```

The full model uses the same five seeds as the baseline. The teacher checkpoint should point to the corresponding baseline model for each seed or to the selected frozen teacher checkpoint used in the experiment.

## Statistical Tests

Scripts:

```text
eval/compute_significance_stats.py
eval/analyze_reviewer_stats.py
```

Curated outputs:

```text
results/significance/
results/reviewer_stats/
```

The statistical evidence includes:

- McNemar's exact test for paired Top-1 correctness.
- Paired bootstrap 95% confidence intervals for Top-1, Top-5, Macro-F1, and Weighted-F1 improvement.
- Wilcoxon signed-rank test for the 241-class Delta F1 distribution.

See `docs/STATISTICAL_ANALYSIS.md` for the calculation details and exact commands.

## Transformer Baselines

Code:

```text
ViT-B16/
Swin-B/
```

Each folder contains `main.py`, `config.py`, `run_server.sh`, and the run outputs copied to:

```text
results/transformer_baselines/
```

Both Transformer baselines used 300 x 300 input resolution.

## Pairwise/Core Ablations

Launcher:

```bash
bash run_ablation_experiments_server.sh
```

Curated summary:

```text
results/ablation/ablation_summary.md
```

Raw logs:

```text
artifacts/ablation_logs/
```

The additional ablations were designed to address the reviewer's concern that isolated components alone do not support claims about component interaction.
