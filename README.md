# Fine-grained Chinese Dish Recognition on CNFOOD-241

This repository contains the code, evaluation outputs, and revision evidence for the manuscript:

**Fine-grained Chinese dish recognition on CNFOOD-241: A discriminative learning approach for improving category-level discrimination**

The project studies closed-set Chinese dish recognition on CNFOOD-241 using RegNetY-32GF as the main backbone. The revised repository is organized to support both reproducibility and the response to reviewers.

## Repository Layout

| Path | Purpose |
|---|---|
| `RegNetY-32GF/` | Baseline RegNetY-32GF training/evaluation code |
| `R_KDAT/` | Full discriminative training code with KD, margin-based classification, embedding regularization, and strong augmentation |
| `ViT-B16/` | ViT-B/16 Transformer baseline code |
| `Swin-B/` | Swin-B Transformer baseline code |
| `eval/` | Evaluation and statistical analysis scripts |
| `results/` | Curated result files used in the manuscript revision and response letter |
| `artifacts/` | Large local artifacts and raw training logs |
| `paper/` | Manuscript, reviewer comments, response letter, and revision planning files |
| `docs/` | Human-readable reproducibility and reviewer-evidence guides |

## Main Revised Results

### Five-seed repeated runs

| Method | Top-1 mean +/- SD | Top-5 mean +/- SD | Macro-F1 mean +/- SD | 95% CI | Seeds |
|---|---:|---:|---:|---:|---|
| Baseline | 83.72 +/- 0.30 | 97.64 +/- 0.11 | 83.11 +/- 0.31 | [83.35, 84.09] | 1, 25, 42, 50, 100 |
| Full model | 84.36 +/- 0.26 | 97.65 +/- 0.10 | 83.38 +/- 0.51 | [84.04, 84.67] | 1, 25, 42, 50, 100 |

See `results/repeated_runs/`.

### Statistical validation

| Analysis | Result |
|---|---:|
| McNemar's exact test | b01/b10 = 4106/3441, p = 2.039e-14 |
| Paired bootstrap Delta Top-1 | 0.635 pp, 95% CI [0.476, 0.796] |
| Paired bootstrap Delta Top-5 | 0.013 pp, 95% CI [-0.054, 0.082] |
| Paired bootstrap Delta Macro-F1 | 0.249 pp, 95% CI [0.018, 0.498] |
| Paired bootstrap Delta Weighted-F1 | 0.580 pp, 95% CI [0.421, 0.741] |
| Wilcoxon signed-rank for class-wise Delta F1 > 0 | p = 0.041207 |

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
- `docs/REVIEW_RESPONSE_EVIDENCE_MAP.md`
- `docs/STATISTICAL_ANALYSIS.md`
- `artifacts/weights/README.md`

These files explain how the code, weights, predictions, statistical tests, and reviewer-response claims connect to each other.

## Notes on Large Files

Model weights are large and should be shared through GitHub Releases or another stable public download link. Local weights are stored under `artifacts/weights/` but should not be committed directly to git.
