# Model Weights

Pretrained model checkpoints are distributed through GitHub Releases:

https://github.com/MrShu86/Fine-grained-Chinese-dish-recognition-on-CNFOOD-241/releases/tag/revision-model-weights-v1

The repository stores code and evaluation outputs directly, while large `.pt` checkpoint files are provided as release assets.

## Released Checkpoints

The release includes checkpoints for the following experiments:

```text
Baseline RegNetY-32GF repeated runs:
baseline_seed_1.pt
baseline_seed_25.pt
baseline_seed_42.pt
baseline_seed_50.pt
baseline_seed_100.pt

Full discriminative model repeated runs:
full_seed_1.pt
full_seed_25.pt
full_seed_42.pt
full_seed_50.pt
full_seed_100.pt

Transformer baselines:
vit_b16_seed_42.pt
swin_b_seed_42.pt

Additional ablations:
ablation_kd_mbc_seed_42.pt
ablation_kd_er_seed_42.pt
ablation_mbc_er_seed_42.pt
ablation_core_seed_42.pt
```

## Evaluation Outputs

The corresponding evaluation outputs are available in:

```text
results/repeated_runs/
results/significance/
results/transformer_baselines/
results/ablation/
```

Use the scripts in `eval/` to re-evaluate checkpoints or recompute the statistical analyses.
