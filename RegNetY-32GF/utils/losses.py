# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """ 多分类 Focal Loss（带 label smoothing） """
    def __init__(self, gamma=2.0, smoothing=0.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, target):
        # target: (N,) int64
        num_classes = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (num_classes - 1))
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

def build_loss_fn(cfg):
    if cfg["use_focal_loss"]:
        return FocalLoss(gamma=cfg.get("focal_gamma", 2.0),
                         smoothing=cfg["label_smoothing"])
    else:
        return nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
