| Method | Backbone type | Input size | Pretraining | Seed | Top-1 | Top-5 | Macro-F1 | Weighted-F1 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| ViT-B/16 | Plain Vision Transformer | 300 | ImageNet | 42 | 82.46 | 96.67 | 81.69 | 82.40 |
| Swin-B | Hierarchical Vision Transformer | 300 | ImageNet | 42 | 84.35 | 97.78 | 83.34 | 84.29 |

Notes: Both Transformer baselines were trained on the same CNFOOD-241 train/validation split with 300 x 300 input resolution, ImageNet-pretrained weights, AdamW, warmup-cosine learning-rate schedule, MixUp/CutMix, Random Erasing, label smoothing, and fp16 mixed precision. The runs used seed 42 and 30 training epochs.
