# utils/schedulers.py
# -*- coding: utf-8 -*-
import math

def lr_warmup_cosine(
    epoch_idx: int,          # 0-based 的 epoch 索引：第1轮传 0，第2轮传 1
    base_lr: float,          # 这里仅用于计算边界；函数返回的是“倍率”，不是绝对 lr
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.01  # 余弦最低学习率 = base_lr * min_lr_ratio
) -> float:
    """
    返回一个“lr 倍率”（multiplier），外部用 base_lr * 该倍率 设置优化器 lr。
    先线性 warmup，再余弦下降到 base_lr * min_lr_ratio。
    """
    # 保护：非法参数时直接返回 1.0
    if total_epochs <= 0:
        return 1.0
    warmup_epochs = max(int(warmup_epochs), 0)
    min_lr_ratio = float(min_lr_ratio)

    # Warmup：从 0 → 1 线性爬坡
    if warmup_epochs > 0 and epoch_idx < warmup_epochs:
        return float(epoch_idx + 1) / float(warmup_epochs)

    # 余弦阶段
    # 归一化进度 t ∈ [0,1]
    denom = max(1, total_epochs - warmup_epochs)
    t = (epoch_idx - warmup_epochs) / denom
    t = min(max(t, 0.0), 1.0)

    # 余弦倍率 ∈ [min_lr_ratio, 1.0]
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))  # ∈ [0,1]
    mult = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return float(mult)
