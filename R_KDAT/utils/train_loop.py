# utils/train_loop.py
# -*- coding: utf-8 -*-
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.cuda.amp import autocast
except Exception:
    from torch.amp.autocast_mode import autocast

# 进度条：若 tqdm 不可用则优雅退化为普通迭代
try:
    from tqdm import tqdm
    def _tqdm_wrapper(iterable, **kwargs):
        return tqdm(iterable, **kwargs)
except Exception:
    def _tqdm_wrapper(iterable, **kwargs):
        return iterable  # 无 tqdm 时直接返回原可迭代对象

from utils.metrics import accuracy as topk_accuracy
from utils.losses import KDLoss, soft_cross_entropy  # 你现有的 KD/soft CE


# ===================== 工具：把软标签转换为硬标签 =====================
def _as_hard_labels(t: torch.Tensor) -> torch.Tensor:
    """
    - 若 t 是 [B] 的 int64，直接返回；
    - 若 t 是 [B, C] 的 one-hot / soft label，返回 argmax 后的 [B] int64。
    """
    if t.dim() == 1:
        return t.long()
    return t.argmax(dim=1).long()


# ===================== MixUp / CutMix =====================
def _rand_bbox(W, H, lam, device):
    import random
    cut_rat = (1. - lam) ** 0.5
    cw, ch = int(W * cut_rat), int(H * cut_rat)
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
    y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
    return x1, y1, x2, y2


def _apply_mixup_cutmix(images, targets, num_classes, mixup_alpha, cutmix_alpha, device):
    """
    返回：images_aug, targets_soft (或 None), aug_type
    - 若都为 0：不做增强，返回 (images, None, None)
    - 若两者都>0：随机二选一
    """
    import numpy as np, random
    B, C, H, W = images.shape
    if mixup_alpha <= 0.0 and cutmix_alpha <= 0.0:
        return images, None, None

    do_mixup = False
    if mixup_alpha > 0 and cutmix_alpha > 0:
        do_mixup = (random.random() < 0.5)
    else:
        do_mixup = (mixup_alpha > 0)

    index = torch.randperm(B, device=device)
    y_onehot = F.one_hot(targets, num_classes=num_classes).float()

    if do_mixup:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        images = lam * images + (1 - lam) * images[index, :]
        targets_soft = lam * y_onehot + (1 - lam) * y_onehot[index, :]
        return images, targets_soft, "mixup"
    else:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        x1, y1, x2, y2 = _rand_bbox(W, H, lam, device)
        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam_adj = 1 - ((x2 - x1) * (y2 - y1)) / (W * H)
        targets_soft = lam_adj * y_onehot + (1 - lam_adj) * y_onehot[index, :]
        return images, targets_soft, "cutmix"


# ===================== Triplet（batch-hard） =====================
class BatchHardTripletLoss(nn.Module):
    """
    在一个 P×K batch 内：
      - 对每个 anchor，选同类 hardest positive 与异类 hardest negative；
      - 距离用欧氏距离；可选对特征做 L2 normalize。
    """
    def __init__(self, margin: float = 0.3, normalize: bool = True):
        super().__init__()
        self.margin = margin
        self.normalize = normalize

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # feats: (B, D)  labels: (B,)
        if self.normalize:
            feats = F.normalize(feats, dim=1)

        # pairwise distance
        dist = torch.cdist(feats, feats, p=2)  # (B, B)

        labels = labels.view(-1, 1)
        eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        is_pos = (labels == labels.t()) & (~eye)
        is_neg = (labels != labels.t())

        # hardest positive: max among positives；若没有正样本则为 0
        pos_dist = torch.where(is_pos, dist, torch.zeros_like(dist))
        hardest_pos, _ = pos_dist.max(dim=1)

        # hardest negative: min among negatives；若没有负样本置为 +inf
        neg_dist = torch.where(is_neg, dist, torch.full_like(dist, float('inf')))
        hardest_neg, _ = neg_dist.min(dim=1)

        triplet = F.relu(hardest_pos - hardest_neg + self.margin)
        return triplet.mean()


# ===================== 训练与验证 =====================
def train_one_epoch(model: nn.Module,
                    dataloader,
                    optimizer: torch.optim.Optimizer,
                    scaler,
                    loss_fn,                   # 你的基础 CE/Focal（对硬标签）
                    device: torch.device,
                    epoch: int,
                    cfg: dict,
                    scheduler_fn=None,
                    teacher: Optional[nn.Module] = None
                    ) -> Tuple[float, float, float]:

    model.train()

    # ---- KD 配置 ----
    kd_cfg = cfg.get("kd", {})
    use_kd = bool(kd_cfg.get("use", False) and teacher is not None)
    kd_loss_fn = KDLoss(alpha=kd_cfg.get("alpha", 0.6),
                        T=kd_cfg.get("T", 3.0),
                        ce_smoothing=cfg.get("label_smoothing", 0.0)) if use_kd else None

    # ---- ArcFace 开关 ----
    use_arc = bool(cfg.get("head", {}).get("arcface", False))

    # ---- Triplet 配置 ----
    trip_cfg = cfg.get("metric", {})
    use_triplet = bool(trip_cfg.get("use_triplet", False))
    triplet_loss_fn = BatchHardTripletLoss(margin=trip_cfg.get("margin", 0.3),
                                           normalize=trip_cfg.get("normalize", True)) if use_triplet else None
    triplet_weight = float(trip_cfg.get("weight", 0.2))

    # ---- MixUp/CutMix 基础参数 + 后期衰减 ----
    mixup_alpha_base = float(cfg.get("mixup_alpha", 0.0))
    cutmix_alpha_base = float(cfg.get("cutmix_alpha", 0.0))
    mixup_alpha, cutmix_alpha = mixup_alpha_base, cutmix_alpha_base
    final_zero = int(cfg.get("mix_sched", {}).get("final_zero_epochs", 0))
    if final_zero > 0:
        total_ep = int(cfg["epochs"])
        if epoch > (total_ep - final_zero):
            remain = max(0, total_ep - epoch + 1)  # 含本轮
            factor = float(remain) / float(final_zero)  # 线性降到 0
            mixup_alpha = mixup_alpha_base * factor
            cutmix_alpha = cutmix_alpha_base * factor

    use_amp = (device.type == "cuda") and (scaler is not None)
    num_classes = int(cfg["num_classes"])

    loss_m, top1_m, top5_m = 0.0, 0.0, 0.0
    n_batches = 0
    lr_show = optimizer.param_groups[0]["lr"]

    iterator = _tqdm_wrapper(
        dataloader,
        total=len(dataloader),
        desc=f"Train E{epoch:03d}",
        ncols=0,
        leave=False
    )

    for images, targets in iterator:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # MixUp/CutMix（与 KD 协同）
        images_aug, targets_soft, aug_type = _apply_mixup_cutmix(
            images, targets, num_classes, mixup_alpha, cutmix_alpha, device
        )

        with autocast(enabled=use_amp):
            # ------- 前向 -------
            if targets_soft is None:
                # 硬标签批次：ArcFace 带 margin（labels=targets）
                if use_arc:
                    logits_s, feats_s = model(images_aug, return_features=True, labels=targets)
                else:
                    logits_s, feats_s = model(images_aug, return_features=True)
                ce_term = loss_fn(logits_s, targets)
                target_for_metric = targets
            else:
                # 软标签批次（MixUp/CutMix）：ArcFace 不加 margin（labels=None）
                if use_arc:
                    logits_s, feats_s = model(images_aug, return_features=True, labels=None)
                else:
                    logits_s, feats_s = model(images_aug, return_features=True)
                ce_term = soft_cross_entropy(logits_s, targets_soft)
                target_for_metric = targets_soft  # 稍后会转成硬标签再算 topk

            # ------- KD -------
            if use_kd:
                with torch.no_grad():
                    logits_t = teacher(images_aug)
                loss = kd_loss_fn(logits_s, logits_t, (targets_soft if targets_soft is not None else targets))
            else:
                loss = ce_term

            # ------- Triplet（仅无软标签批次启用） -------
            if use_triplet and (targets_soft is None):
                trip = triplet_loss_fn(feats_s, targets)
                loss = loss + triplet_weight * trip

        # ------- 反传/更新 -------
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # ------- 训练指标（滚动均值） -------
        with torch.no_grad():
            hard_t = _as_hard_labels(target_for_metric)
            top1, top5 = topk_accuracy(logits_s, hard_t, topk=(1, 5))
            loss_m += float(loss.detach().item())
            top1_m += float(top1)
            top5_m += float(top5)
            n_batches += 1

        # 进度条后缀
        iterator.set_postfix({
            "loss": f"{loss_m/max(1,n_batches):.4f}",
            "top1": f"{top1_m/max(1,n_batches):.2f}%",
            "top5": f"{top5_m/max(1,n_batches):.2f}%",
            "aug": (aug_type or "none"),
            "lr": f"{lr_show:.2e}",
            "KD": "on" if use_kd else "off",
            "Arc": "on" if use_arc else "off",
            "Trip": "on" if use_triplet else "off",
        })

    if hasattr(iterator, "close"):
        iterator.close()

    loss_m /= max(1, n_batches)
    top1_m /= max(1, n_batches)
    top5_m /= max(1, n_batches)
    return loss_m, top1_m, top5_m


@torch.no_grad()
def validate(model: nn.Module,
             dataloader,
             device: torch.device,
             max_batches: int = 0) -> Tuple[float, float, float]:

    model.eval()
    use_amp = (device.type == "cuda")

    loss_m, top1_m, top5_m = 0.0, 0.0, 0.0
    n_batches = 0

    iterator = _tqdm_wrapper(
        dataloader if max_batches == 0 else list(dataloader)[:max_batches],
        total=(len(dataloader) if max_batches == 0 else max_batches),
        desc="Validate",
        ncols=0,
        leave=False
    )

    for bi, batch in enumerate(iterator):
        if max_batches and bi >= max_batches:
            break
        images, targets = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(images)  # ArcFace 在 labels=None 时自动退化为普通 logits
            loss = F.cross_entropy(logits, targets)

        top1, top5 = topk_accuracy(logits, targets, topk=(1, 5))
        loss_m += float(loss.detach().item())
        top1_m += float(top1)
        top5_m += float(top5)
        n_batches += 1

        iterator.set_postfix({
            "loss": f"{loss_m/max(1,n_batches):.4f}",
            "top1": f"{top1_m/max(1,n_batches):.2f}%",
            "top5": f"{top5_m/max(1,n_batches):.2f}%"
        })

    if hasattr(iterator, "close"):
        iterator.close()

    loss_m /= max(1, n_batches)
    top1_m /= max(1, n_batches)
    top5_m /= max(1, n_batches)
    return loss_m, top1_m, top5_m
