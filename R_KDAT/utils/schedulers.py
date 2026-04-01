# utils/schedulers.py
# -*- coding: utf-8 -*-
import math
from typing import Optional, Sequence


def lr_warmup_cosine(
    epoch_idx: int,          # 0-based：第1轮传0，第2轮传1
    base_lr: float,          # 仅用于计算边界；本函数返回“倍率”，不是绝对 lr
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.01  # 余弦最低学习率 = base_lr * min_lr_ratio
) -> float:
    """
    返回一个“lr 倍率”（multiplier），外部用 base_lr * 该倍率 设置优化器 lr。
    先线性 warmup，再余弦下降到 base_lr * min_lr_ratio。
    向后兼容你现有的 main.py 调用。
    """
    # 保护：非法参数时直接返回 1.0（不改动 lr）
    if total_epochs <= 0:
        return 1.0
    warmup_epochs = max(int(warmup_epochs), 0)
    min_lr_ratio = float(min_lr_ratio)

    # Warmup：从 0 → 1 线性爬坡
    if warmup_epochs > 0 and epoch_idx < warmup_epochs:
        return float(epoch_idx + 1) / float(warmup_epochs)

    # 余弦阶段
    denom = max(1, total_epochs - warmup_epochs)
    t = (epoch_idx - warmup_epochs) / denom
    t = min(max(t, 0.0), 1.0)

    cosine = 0.5 * (1.0 + math.cos(math.pi * t))  # ∈ [0,1]
    mult = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return float(mult)


# ========= 进阶：按 step 的 warmup + cosine（可选，用于更平滑的调度） =========
def lr_warmup_cosine_by_step(
    step_idx: int,           # 0-based：当前 step
    total_steps: int,        # 训练总 step 数（epochs * steps_per_epoch）
    warmup_steps: int,
    min_lr_ratio: float = 0.01
) -> float:
    """
    返回“lr 倍率”——线性 warmup（按 step）+ 余弦退火到 min_lr_ratio。
    适合每个 iteration 更新 lr 的训练循环。
    """
    if total_steps <= 0:
        return 1.0
    warmup_steps = max(int(warmup_steps), 0)
    step_idx = max(0, min(step_idx, total_steps - 1))
    min_lr_ratio = float(min_lr_ratio)

    if warmup_steps > 0 and step_idx < warmup_steps:
        return float(step_idx + 1) / float(warmup_steps)

    denom = max(1, total_steps - warmup_steps)
    t = (step_idx - warmup_steps) / denom
    t = min(max(t, 0.0), 1.0)

    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    mult = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return float(mult)


# ========= 进阶：SGDR 余弦重启（可选） =========
def lr_cosine_restarts(
    epoch_idx: int,                # 0-based
    base_lr: float,
    T_0: int,                      # 第一个周期的 epoch 数
    T_mult: float = 2.0,           # 每次重启周期乘以该系数（例如 2.0）
    min_lr_ratio: float = 0.01,
    warmup_in_cycle: int = 0       # 每个周期内额外的 warmup epoch 数（可为0）
) -> float:
    """
    返回“lr 倍率”：SGDR 余弦重启（Stochastic Gradient Descent with Warm Restarts）
    - 周期长度按 T_0, T_0*T_mult, T_0*T_mult^2, ... 逐次增长
    - 每个周期开始可选 warmup_in_cycle 个 epoch 线性升温
    """
    if T_0 <= 0:
        return 1.0
    # 定位当前处于第几个周期，以及该周期内的本地 epoch 索引
    T_cur = T_0
    e = epoch_idx
    cycle_start = 0
    while e >= T_cur:
        e -= T_cur
        cycle_start += T_cur
        T_cur = int(T_cur * T_mult)
        if T_cur <= 0:
            # 防极端 T_mult
            T_cur = 1
            break

    # 周期内 warmup
    if warmup_in_cycle > 0 and e < warmup_in_cycle:
        return float(e + 1) / float(max(1, warmup_in_cycle))

    # 周期内余弦
    # 将 e 映射到 [0,1]
    denom = max(1, T_cur - warmup_in_cycle)
    t = (e - warmup_in_cycle) / denom
    t = min(max(t, 0.0), 1.0)

    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    mult = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return float(mult)


# ========= 便捷函数：统一给优化器写入 LR =========
def set_optimizer_lr(optimizer, base_lr: float, mul: float):
    """
    将 optimizer 的每个 param_group['lr'] 设为 base_lr * mul。
    你也可以传入 param_group 自己的 'base_lr'，此处保持简单。
    """
    lr = float(base_lr) * float(mul)
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# =========（可选）多组 base_lr 的写法 =========
def set_optimizer_lrs(
    optimizer,
    base_lrs: Sequence[float],
    mul: float
):
    """
    当你想对不同 param_group 使用不同 base_lr 时使用。
    例如：backbone / head 不同学习率组。
    """
    assert len(base_lrs) == len(optimizer.param_groups), \
        "base_lrs 数量必须与 param_groups 数量一致"
    for pg, blr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = float(blr) * float(mul)
