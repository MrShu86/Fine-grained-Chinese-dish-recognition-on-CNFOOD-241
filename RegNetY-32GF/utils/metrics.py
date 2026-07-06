# -*- coding: utf-8 -*-
import torch

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """
    通用 top-k 准确率。
    返回一个 list[float]，每个元素是百分比(0-100 的 float)。
    """
    num_classes = output.size(1)
    maxk = min(max(topk), num_classes)
    batch_size = target.size(0)

    # [B, C] -> 取每行 top-k -> [B, maxk] -> 转置成 [maxk, B]
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # [maxk, B]

    res = []
    for k in topk:
        k = min(k, num_classes)
        correct_k = correct[:k].reshape(-1).float().sum(0)  # 标量
        res.append((correct_k * (100.0 / batch_size)).item())
    return res
