# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """多分类 Focal Loss（带 label smoothing）"""
    def __init__(self, gamma: float = 2.0, smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (N, C)
        target: (N,) int64
        """
        num_classes = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / max(1, (num_classes - 1)))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        focal = (1.0 - prob) ** self.gamma
        loss = -(focal * true_dist * log_prob).sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """
    软标签交叉熵（支持 MixUp/CutMix 的 one-hot/soft 标签）
    logits: (N, C)
    soft_targets: (N, C), 每行和为1
    """
    return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


class KDLoss(nn.Module):
    """
    知识蒸馏损失： (1-α)·CE + α·KL(teacher || student)·T^2
    - 若传入软标签 (B,C)，CE 用 soft_cross_entropy
    - 若传入硬标签 (B,)，CE 用交叉熵（支持 label smoothing）
    """
    def __init__(self, alpha: float = 0.6, T: float = 3.0, ce_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.T = T
        self.ce_smoothing = ce_smoothing

    def forward(self,
                logits_s: torch.Tensor,
                logits_t: torch.Tensor,
                target_or_soft: torch.Tensor) -> torch.Tensor:
        # CE 项
        if target_or_soft.dim() == 2 and target_or_soft.dtype.is_floating_point:
            ce = soft_cross_entropy(logits_s, target_or_soft)
        else:
            ce = F.cross_entropy(logits_s, target_or_soft, label_smoothing=self.ce_smoothing)

        # KL(teacher || student) 项（温度缩放 + T^2 修正）
        T = self.T
        p_t = F.softmax(logits_t / T, dim=1)
        log_p_s = F.log_softmax(logits_s / T, dim=1)
        kl = F.kl_div(log_p_s, p_t, reduction='batchmean') * (T * T)

        return (1.0 - self.alpha) * ce + self.alpha * kl


def build_loss_fn(cfg: dict):
    """
    返回“基础监督”损失（不含 KD）。
    KD 由训练循环控制；如需在外部创建 KDLoss，请直接实例化 KDLoss(alpha=..., T=..., ce_smoothing=...)
    """
    if cfg.get("use_focal_loss", False):
        return FocalLoss(gamma=cfg.get("focal_gamma", 2.0),
                         smoothing=cfg.get("label_smoothing", 0.0))
    else:
        return nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
