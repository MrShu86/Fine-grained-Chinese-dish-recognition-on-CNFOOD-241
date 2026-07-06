# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import time
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

# AMP 兼容导入：优先 cuda.amp，其次 torch.amp
try:
    from torch.cuda.amp import GradScaler
except Exception:
    from torch.amp import GradScaler


def _amp_dtype(cfg):
    name = str(cfg.get("amp_dtype", "bf16")).lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return None


def _autocast_context(device: torch.device, cfg: dict):
    dtype = _amp_dtype(cfg)
    enabled = (device.type == "cuda") and (dtype is not None)
    return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


# ---------- metrics ----------
@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    """Return list of top-k accuracies in %."""
    maxk = max(topk)
    B = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # [B, maxk]
    pred = pred.t()                                                # [maxk, B]
    correct = pred.eq(target.view(1, -1).expand_as(pred))          # [maxk, B]

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / B))
    return res  # [top1%, top5%]


# ---------- mixup / cutmix ----------
def _rand_bbox(W, H, lam):
    """Return a random rectangular bbox for CutMix."""
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2

def apply_mixup(x, y, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    x = lam * x + (1 - lam) * x[idx]
    return x, y, y[idx], lam

def apply_cutmix(x, y, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    B, C, H, W = x.size()
    x1, y1, x2, y2 = _rand_bbox(W, H, lam)
    idx = torch.randperm(B, device=x.device)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(W * H))
    return x, y, y[idx], lam


# ---------- train / val ----------
def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    cfg=None,
    **kwargs
) -> Tuple[float, float, float]:
    """
    兼容：若 cfg 为 None，会尝试从 kwargs['config'] 获取。
    返回: (avg_loss, top1%, top5%)
    """
    if cfg is None:
        cfg = kwargs.get("config", None)
    assert cfg is not None, "train_one_epoch: need cfg/config dict"

    model.train()

    use_scaler = (device.type == "cuda") and (scaler is not None) and scaler.is_enabled()

    mixup_alpha = float(cfg.get("mixup_alpha", 0.0) or 0.0)
    cutmix_alpha = float(cfg.get("cutmix_alpha", 0.0) or 0.0)
    # 若两者同时启用，按 50/50 概率随机选择一种
    use_both = (mixup_alpha > 0) and (cutmix_alpha > 0)

    running_loss = 0.0
    total = 0
    top1_sum = 0.0
    top5_sum = 0.0

    def _cur_lr():
        try:
            return optimizer.param_groups[0]["lr"]
        except Exception:
            return None

    pbar = tqdm(dataloader, ncols=120, desc=f"[Train E{epoch:03d}]")
    for batch in pbar:
        x, y = batch
        x = x.to(device, non_blocking=True)
        if cfg.get("channels_last", False):
            x = x.contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        # 选择增强
        y_a, y_b, lam = y, y, 1.0
        if use_both:
            if torch.rand(1).item() < 0.5:
                x, y_a, y_b, lam = apply_mixup(x, y, mixup_alpha)
            else:
                x, y_a, y_b, lam = apply_cutmix(x, y, cutmix_alpha)
        elif mixup_alpha > 0:
            x, y_a, y_b, lam = apply_mixup(x, y, mixup_alpha)
        elif cutmix_alpha > 0:
            x, y_a, y_b, lam = apply_cutmix(x, y, cutmix_alpha)

        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device, cfg):
            logits = model(x)  # RegNetY-32GF: forward -> logits
            # mixup/cutmix 的加权 CE
            loss = lam * loss_fn(logits, y_a) + (1.0 - lam) * loss_fn(logits, y_b)

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # 统计
        B = y.size(0)
        running_loss += loss.item() * B
        total += B
        top1, top5 = accuracy(logits.detach(), y, topk=(1, 5))
        top1_sum += top1.item() * B
        top5_sum += top5.item() * B

        avg_loss = running_loss / max(total, 1)
        avg_top1 = top1_sum / max(total, 1)
        avg_top5 = top5_sum / max(total, 1)

        postfix = {
            "loss": f"{avg_loss:.4f}",
            "top1": f"{avg_top1:.2f}%",
            "top5": f"{avg_top5:.2f}%"
        }
        lr_now = _cur_lr()
        if lr_now is not None:
            postfix["lr"] = f"{lr_now:.2e}"
        pbar.set_postfix(postfix)

    return running_loss / max(total, 1), top1_sum / max(total, 1), top5_sum / max(total, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: Iterable,
    device: torch.device,
    max_batches: int = 0
) -> Tuple[float, float, float]:
    """
    返回: (avg_loss, top1%, top5%)
    注意：这里的 loss 使用标准 CE（仅用于监控）。
    """
    model.eval()
    ce = nn.CrossEntropyLoss()
    cfg = getattr(dataloader, "_codex_cfg", {}) if hasattr(dataloader, "_codex_cfg") else {}

    total = 0
    running_loss = 0.0
    top1_sum = 0.0
    top5_sum = 0.0

    pbar = tqdm(dataloader, ncols=100, desc="[Val]")
    for bi, (x, y) in enumerate(pbar):
        if (max_batches is not None) and (max_batches > 0) and (bi >= max_batches):
            break
        x = x.to(device, non_blocking=True)
        if cfg.get("channels_last", False):
            x = x.contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        with _autocast_context(device, cfg):
            logits = model(x)
            loss = ce(logits, y)

        B = y.size(0)
        running_loss += loss.item() * B
        total += B

        top1, top5 = accuracy(logits, y, topk=(1, 5))
        top1_sum += top1.item() * B
        top5_sum += top5.item() * B

        avg_loss = running_loss / max(total, 1)
        avg_top1 = top1_sum / max(total, 1)
        avg_top5 = top5_sum / max(total, 1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{avg_top1:.2f}%", top5=f"{avg_top5:.2f}%")

    return running_loss / max(total, 1), top1_sum / max(total, 1), top5_sum / max(total, 1)
