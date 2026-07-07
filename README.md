# Fine-grained Chinese Dish Recognition on CNFOOD-241

This repository contains the training code, evaluation scripts, pretrained model weights, and experimental results for:

**Fine-grained Chinese dish recognition on CNFOOD-241: A discriminative learning approach for improving category-level discrimination**

The project studies closed-set Chinese dish recognition on CNFOOD-241 using RegNetY-32GF as the main backbone. It includes a baseline model, the proposed discriminative training framework, repeated-run evaluation results, statistical analysis scripts, Transformer baselines, and additional ablation experiments.

## Repository Layout

| Path | Purpose |
|---|---|
| `RegNetY-32GF/` | Baseline RegNetY-32GF training/evaluation code |
| `R_KDAT/` | Full discriminative training code with KD, margin-based classification, embedding regularization, and strong augmentation |
| `ViT-B16/` | ViT-B/16 Transformer baseline code |
| `Swin-B/` | Swin-B Transformer baseline code |
| `eval/` | Evaluation and statistical analysis scripts |
| `results/` | Evaluation outputs, repeated-run statistics, significance tests, Transformer baseline results, and ablation summaries |
| `artifacts/ablation_logs/` | Raw training logs for the added ablation experiments |
| `artifacts/weights/` | Instructions and public release link for pretrained model weights |
| `docs/` | Reproducibility and statistical-analysis documentation |

## Main Revised Results

### Five-seed repeated runs

| Method | Top-1 mean +/- SD | Top-5 mean +/- SD | Macro-F1 mean +/- SD | 95% CI | Seeds |
|---|---:|---:|---:|---:|---|
| Baseline | 83.72 +/- 0.30 | 97.64 +/- 0.11 | 83.11 +/- 0.31 | [83.35, 84.09] | 1, 25, 42, 50, 100 |
| Full model | 84.37 +/- 0.29 | 97.66 +/- 0.11 | 83.41 +/- 0.57 | [84.02, 84.73] | 1, 25, 42, 50, 100 |

See `results/repeated_runs/`.

### Statistical validation

| Analysis | Result |
|---|---:|
| McNemar's exact test | b01/b10 = 4112/3430, p = 4.266e-15 |
| Paired bootstrap Delta Top-1 | 0.651 pp, 95% CI [0.491, 0.817] |
| Paired bootstrap Delta Top-5 | 0.022 pp, 95% CI [-0.047, 0.091] |
| Paired bootstrap Delta Macro-F1 | 0.285 pp, 95% CI [0.049, 0.540] |
| Paired bootstrap Delta Weighted-F1 | 0.597 pp, 95% CI [0.436, 0.766] |
| Wilcoxon signed-rank for class-wise Delta F1 > 0 | p = 0.017803 |

See `results/significance/`.

### Transformer baselines

| Method | Input size | Seed | Top-1 | Top-5 | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|
| ViT-B/16 | 300 | 42 | 82.46 | 96.67 | 81.69 | 82.40 |
| Swin-B | 300 | 42 | 84.35 | 97.78 | 83.34 | 84.29 |

See `results/transformer_baselines/`.

### Additional pairwise/core ablations

| Variant | Setting | Top-1 | Top-5 |
|---|---|---:|---:|
| KD + MBC | KD on, ArcFace/MBC on, ER off, no strong augmentation | 80.84 | 95.46 |
| KD + ER | KD on, MBC off, ER on, no strong augmentation | 83.10 | 97.40 |
| MBC + ER | KD off, MBC on, ER on, no strong augmentation | 81.07 | 94.21 |
| KD + MBC + ER | KD on, MBC on, ER on, no strong augmentation | 80.93 | 95.72 |

See `results/ablation/` and `artifacts/ablation_logs/`.

## Pretrained Weights

Pretrained checkpoints are available from GitHub Releases:

https://github.com/MrShu86/Fine-grained-Chinese-dish-recognition-on-CNFOOD-241/releases/tag/revision-model-weights-v1

The release contains the baseline repeated-run checkpoints, full-model repeated-run checkpoints, Transformer baseline checkpoints, and added ablation checkpoints. See `artifacts/weights/README.md` for details.

## Data

The experiments use the public CNFOOD-241 dataset. The dataset itself is not redistributed in this repository. Set the training and validation paths when running the scripts.

Expected server layout used in the experiments:

```bash
/food_data/food_data/CNFOOD-241/CNFOOD-241/train600x600
/food_data/food_data/CNFOOD-241/CNFOOD-241/val600x600
```

## Reproducibility

Start with:

- `docs/REPRODUCIBILITY.md`
- `docs/STATISTICAL_ANALYSIS.md`
- `artifacts/weights/README.md`

These files explain how to run the code, evaluate checkpoints, and reproduce the reported statistical analyses.

## Notes on Large Files

Model weights are distributed through GitHub Releases rather than committed directly to the repository.
