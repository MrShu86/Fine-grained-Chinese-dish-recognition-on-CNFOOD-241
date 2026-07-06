# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision.models import regnet_y_32gf, RegNet_Y_32GF_Weights

class RegNetY32(nn.Module):
    def __init__(self, num_classes=241, pretrained=True):
        super().__init__()
        weights = RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None
        self.backbone = regnet_y_32gf(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        return self.backbone(x)  # logits

def build_model(num_classes=241, pretrained=True):
    model = RegNetY32(num_classes=num_classes, pretrained=pretrained)
    return model
