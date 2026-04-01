# -*- coding: utf-8 -*-
import torch

@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    """
    通用 top-k 准确率，返回 [topk1, topk2, ...] 百分比（float）。
    兼容两类 target：
      - 硬标签：形状 (B,) 的 long tensor
      - 软/one-hot 标签：形状 (B, C) 的 float tensor（会自动 argmax 到 (B,)）
    """
    if target.dim() == 2:
        target = target.argmax(dim=1)

    assert output.dim() == 2, f"output 应为 (B, C)，但得到 {tuple(output.shape)}"
    assert target.dim() == 1, f"target 应为 (B,)，但得到 {tuple(target.shape)}"
    assert output.size(0) == target.size(0), "batch 大小不一致"

    num_classes = output.size(1)
    maxk = min(max(topk), num_classes)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t().contiguous()      # [maxk, B]
    target = target.view(1, -1).expand_as(pred)

    correct = pred.eq(target)         # [maxk, B]

    res = []
    for k in topk:
        k = min(k, num_classes)
        correct_k = correct[:k].reshape(-1).float().sum(0)  # 标量
        res.append((correct_k * (100.0 / batch_size)).item())
    return res
