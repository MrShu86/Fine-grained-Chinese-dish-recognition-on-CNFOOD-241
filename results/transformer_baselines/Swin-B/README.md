# Swin-B CNFOOD-241 Baseline

This folder trains a Swin-B hierarchical transformer baseline with 300 x 300 inputs.

Model: `swin_base_patch4_window7_224` from `timm`, initialized with ImageNet pretrained weights.

## Server usage

```bash
cd /food_data/revise_re/Swin-B
pip install timm
bash run_server.sh
```

Recommended two-RTX-4090 command:

```bash
cd /food_data/revise_re/Swin-B
SEED=42 BATCH_SIZE=32 NUM_WORKERS=8 AMP_DTYPE=fp16 bash run_server.sh
```

If memory is enough, you can try:

```bash
BATCH_SIZE=64 bash run_server.sh
```

Outputs:

- `exp_swin_b_300/best.pt`
- `exp_swin_b_300/last.pt`
- `exp_swin_b_300/metrics.json`
- `exp_swin_b_300/history.json`
- `exp_swin_b_300/train.log`

