# ViT-B/16 CNFOOD-241 Baseline

This folder trains a ViT-B/16 transformer baseline with 300 x 300 inputs.

Model: `vit_base_patch16_224` from `timm`, initialized with ImageNet pretrained weights.

## Server usage

```bash
cd /food_data/revise_re/ViT-B16
pip install timm
bash run_server.sh
```

Recommended two-RTX-4090 command:

```bash
cd /food_data/revise_re/ViT-B16
SEED=42 BATCH_SIZE=64 NUM_WORKERS=8 AMP_DTYPE=fp16 bash run_server.sh
```

Outputs:

- `exp_vit_b16_300/best.pt`
- `exp_vit_b16_300/last.pt`
- `exp_vit_b16_300/metrics.json`
- `exp_vit_b16_300/history.json`
- `exp_vit_b16_300/train.log`

