# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import regnet_y_32gf, RegNet_Y_32GF_Weights


class ArcMarginProduct(nn.Module):
    """
    ArcFace/AM-Softmax 头：
    - 训练：传入 labels 时使用 cos(θ + m) 并乘以 s
    - 推理/软标签：labels=None 时退化为 s * cos(θ)（不加 margin）
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.35, easy_margin=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = float(s)
        self.m = float(m)
        self.easy_margin = bool(easy_margin)

        # 预计算常量（放到 forward 再 to(device)）
        import math
        self._cos_m = math.cos(self.m)
        self._sin_m = math.sin(self.m)
        self._th    = math.cos(math.pi - self.m)
        self._mm    = math.sin(math.pi - self.m) * self.m

    def forward(self, x, labels=None):
        # L2 normalize
        x = F.normalize(x, dim=1)
        W = F.normalize(self.weight, dim=1)
        cos = torch.matmul(x, W.t())                          # (B, C)

        if labels is None:
            # 无标签（推理或软标签批次）：不加 margin
            return cos * self.s

        # 带标签：加角度间隔
        sin   = torch.sqrt(torch.clamp(1.0 - cos * cos, min=1e-7))
        cos_m = torch.as_tensor(self._cos_m, device=x.device, dtype=x.dtype)
        sin_m = torch.as_tensor(self._sin_m, device=x.device, dtype=x.dtype)
        th    = torch.as_tensor(self._th,    device=x.device, dtype=x.dtype)
        mm    = torch.as_tensor(self._mm,    device=x.device, dtype=x.dtype)

        cos_t = cos * cos_m - sin * sin_m                     # cos(θ+m)
        if self.easy_margin:
            cos_t = torch.where(cos > 0, cos_t, cos)
        else:
            cos_t = torch.where(cos > th, cos_t, cos - mm)

        one_hot = F.one_hot(labels, num_classes=cos.size(1)).to(cos.dtype)
        logits = one_hot * cos_t + (1.0 - one_hot) * cos      # 仅在正类加 margin
        return logits * self.s


class RegNetY32(nn.Module):
    def __init__(self, num_classes=241, pretrained=True,
                 use_arcface=False, arc_s=30.0, arc_m=0.35):
        super().__init__()
        weights = RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None
        self.backbone = regnet_y_32gf(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.use_arcface = bool(use_arcface)
        if self.use_arcface:
            self.head = ArcMarginProduct(in_feats, num_classes, s=arc_s, m=arc_m)
        else:
            self.head = nn.Linear(in_feats, num_classes)

    def forward(self, x, return_features=False, labels=None):
        feats = self.backbone(x)  # (B, D)
        if self.use_arcface:
            logits = self.head(feats, labels=labels)  # labels 可为 None
        else:
            logits = self.head(feats)
        if return_features:
            return logits, feats
        return logits


def build_model(num_classes=241, pretrained=True, **kwargs):
    return RegNetY32(num_classes=num_classes, pretrained=pretrained, **kwargs)
