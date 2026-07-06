# Model Weights

Large model weights should not be committed directly to git. For public release, upload them to GitHub Releases or another stable public storage location and link them from this file.

## Local Files

The following local weight files were moved here during repository cleanup:

| Local file | Notes |
|---|---|
| `best.pt` | Legacy local checkpoint; verify exact role before public release |
| `best (3).pt` | Legacy local checkpoint; verify exact role before public release |

## Weights That Should Be Released

To fully support the revised manuscript and response to reviewers, the public release should include:

1. Baseline RegNetY-32GF five-seed repeated-run checkpoints.
2. Full model five-seed repeated-run checkpoints.
3. ViT-B/16 checkpoint and evaluation outputs.
4. Swin-B checkpoint and evaluation outputs.
5. Added pairwise/core ablation checkpoints and evaluation outputs.

Recommended public naming:

```text
baseline_seed_1.pt
baseline_seed_25.pt
baseline_seed_42.pt
baseline_seed_50.pt
baseline_seed_100.pt
full_seed_1.pt
full_seed_25.pt
full_seed_42.pt
full_seed_50.pt
full_seed_100.pt
vit_b16_seed_42.pt
swin_b_seed_42.pt
ablation_kd_mbc_seed_42.pt
ablation_kd_er_seed_42.pt
ablation_mbc_er_seed_42.pt
ablation_core_seed_42.pt
```

After uploading, add the download URLs and checksums here.

