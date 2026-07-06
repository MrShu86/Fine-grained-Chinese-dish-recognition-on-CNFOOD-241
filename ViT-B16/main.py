# -*- coding: utf-8 -*-
import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

try:
    import timm
except ImportError as exc:
    raise SystemExit("Missing dependency: timm. Install with `pip install timm`.") from exc

from config import CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT-B/16 transformer baseline on CNFOOD-241.")
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16", "off"], default=None)
    parser.add_argument("--resume-ckpt", default="")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-mixup-cutmix", action="store_true")
    parser.add_argument("--no-random-erasing", action="store_true")
    parser.add_argument("--no-dataparallel", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def apply_overrides(cfg, args):
    cfg = copy.deepcopy(cfg)
    mapping = {
        "train_dir": args.train_dir,
        "val_dir": args.val_dir,
        "out_dir": args.out_dir,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "amp_dtype": args.amp_dtype,
    }
    for key, value in mapping.items():
        if value is not None:
            cfg[key] = value
    if args.no_pretrained:
        cfg["pretrained"] = False
    if args.no_mixup_cutmix:
        cfg["mixup_alpha"] = 0.0
        cfg["cutmix_alpha"] = 0.0
    if args.no_random_erasing:
        cfg["random_erasing"] = False
    if args.no_dataparallel:
        cfg["dataparallel"] = False
    if args.channels_last:
        cfg["channels_last"] = True
    if args.compile:
        cfg["compile"] = True
    cfg["resume_ckpt"] = args.resume_ckpt
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_transforms(cfg):
    aug = [
        transforms.RandomResizedCrop(cfg["img_size"], scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    if cfg.get("random_erasing", True):
        aug.append(transforms.RandomErasing(
            p=float(cfg.get("random_erasing_p", 0.25)),
            scale=(0.02, 0.20),
            ratio=(0.3, 3.3),
            inplace=False,
        ))
    return transforms.Compose(aug)


def val_transforms(cfg):
    img_size = cfg["img_size"]
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.12)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def build_loaders(cfg):
    ds_tr = ImageFolder(cfg["train_dir"], transform=train_transforms(cfg))
    ds_va = ImageFolder(cfg["val_dir"], transform=val_transforms(cfg))
    nw = int(cfg.get("num_workers", 8))
    common = dict(
        num_workers=nw,
        pin_memory=bool(cfg.get("pin_memory", True)),
        persistent_workers=nw > 0,
    )
    if nw > 0:
        common["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
    dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, **common)
    dl_va = DataLoader(ds_va, batch_size=cfg["batch_size"], shuffle=False, drop_last=False, **common)
    return ds_tr, ds_va, dl_tr, dl_va


def lr_multiplier(epoch, cfg):
    warmup = int(cfg.get("warmup_epochs", 5))
    total = int(cfg["epochs"])
    min_ratio = float(cfg.get("cosine_min_lr_ratio", 0.01))
    if warmup > 0 and epoch <= warmup:
        return epoch / float(warmup)
    t = (epoch - warmup) / max(1, total - warmup)
    t = min(max(t, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))


def rand_bbox(size, lam):
    _, _, h, w = size
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    return x1, y1, x2, y2


def apply_mixup_cutmix(images, targets, cfg):
    mixup_alpha = float(cfg.get("mixup_alpha", 0.0))
    cutmix_alpha = float(cfg.get("cutmix_alpha", 0.0))
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return images, targets, None
    use_mixup = mixup_alpha > 0 and (cutmix_alpha <= 0 or random.random() < 0.5)
    bsz = images.size(0)
    index = torch.randperm(bsz, device=images.device)
    if use_mixup:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        mixed = lam * images + (1 - lam) * images[index]
        return mixed, (targets, targets[index], lam), "mixup"
    lam = np.random.beta(cutmix_alpha, cutmix_alpha)
    x1, y1, x2, y2 = rand_bbox(images.size(), lam)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(images.size(-1) * images.size(-2)))
    return mixed, (targets, targets[index], lam), "cutmix"


def supervised_loss(logits, target_or_mix, ce):
    if isinstance(target_or_mix, tuple):
        y1, y2, lam = target_or_mix
        return lam * F.cross_entropy(logits, y1) + (1 - lam) * F.cross_entropy(logits, y2)
    return ce(logits, target_or_mix)


@torch.no_grad()
def accuracy(logits, targets, topk=(1, 5)):
    maxk = min(max(topk), logits.size(1))
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    out = []
    for k in topk:
        k = min(k, logits.size(1))
        out.append(correct[:k].reshape(-1).float().sum().item() * 100.0 / targets.size(0))
    return out


def autocast_ctx(device, cfg):
    name = str(cfg.get("amp_dtype", "fp16")).lower()
    if device.type != "cuda" or name == "off":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.float16 if name == "fp16" else torch.bfloat16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def train_one_epoch(model, loader, optimizer, scaler, ce, device, epoch, cfg):
    model.train()
    total_loss = total_top1 = total_top5 = total_n = 0.0
    pbar = tqdm(loader, desc=f"Train E{epoch:03d}", dynamic_ncols=True)
    use_scaler = scaler is not None and scaler.is_enabled()
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if cfg.get("channels_last", False):
            images = images.contiguous(memory_format=torch.channels_last)
        images, target_or_mix, aug_name = apply_mixup_cutmix(images, targets, cfg)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, cfg):
            logits = model(images)
            loss = supervised_loss(logits, target_or_mix, ce)
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        top1, top5 = accuracy(logits.detach(), targets)
        bsz = targets.size(0)
        total_loss += loss.item() * bsz
        total_top1 += top1 * bsz
        total_top5 += top5 * bsz
        total_n += bsz
        pbar.set_postfix(loss=f"{total_loss/total_n:.4f}", top1=f"{total_top1/total_n:.2f}%",
                         top5=f"{total_top5/total_n:.2f}%", aug=aug_name or "none",
                         lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    return total_loss / total_n, total_top1 / total_n, total_top5 / total_n


@torch.no_grad()
def validate(model, loader, ce, device, cfg):
    model.eval()
    total_loss = total_top1 = total_top5 = total_n = 0.0
    all_true, all_pred = [], []
    pbar = tqdm(loader, desc="Validate", dynamic_ncols=True)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if cfg.get("channels_last", False):
            images = images.contiguous(memory_format=torch.channels_last)
        with autocast_ctx(device, cfg):
            logits = model(images)
            loss = ce(logits, targets)
        top1, top5 = accuracy(logits, targets)
        pred = logits.argmax(dim=1)
        all_true.append(targets.cpu())
        all_pred.append(pred.cpu())
        bsz = targets.size(0)
        total_loss += loss.item() * bsz
        total_top1 += top1 * bsz
        total_top5 += top5 * bsz
        total_n += bsz
        pbar.set_postfix(loss=f"{total_loss/total_n:.4f}", top1=f"{total_top1/total_n:.2f}%",
                         top5=f"{total_top5/total_n:.2f}%")
    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    macro_f1, weighted_f1 = f1_from_predictions(y_true, y_pred, cfg["num_classes"])
    return total_loss / total_n, total_top1 / total_n, total_top5 / total_n, macro_f1, weighted_f1


def f1_from_predictions(y_true, y_pred, num_classes):
    support = np.bincount(y_true, minlength=num_classes).astype(float)
    pred_count = np.bincount(y_pred, minlength=num_classes).astype(float)
    correct = y_true == y_pred
    tp = np.bincount(y_true[correct], minlength=num_classes).astype(float)
    fp = pred_count - tp
    fn = support - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    macro_f1 = float(np.mean(f1) * 100.0)
    weighted_f1 = float(np.sum(f1 * support) / np.sum(support) * 100.0)
    return macro_f1, weighted_f1


def save_checkpoint(path, model, optimizer, scaler, epoch, metrics, cfg):
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({
        "model": state,
        "opt": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "epoch": epoch,
        "metrics": metrics,
        "best_top1": metrics["top1"],
        "config": cfg,
    }, path)


def main():
    args = parse_args()
    cfg = apply_overrides(CONFIG, args)
    set_seed(int(cfg["seed"]))
    os.makedirs(cfg["out_dir"], exist_ok=True)
    with open(Path(cfg["out_dir"]) / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"CUDA: True GPUs={torch.cuda.device_count()} Name={torch.cuda.get_device_name(0)}")
    print("CONFIG:", cfg)

    _, _, dl_tr, dl_va = build_loaders(cfg)
    model = timm.create_model(
        cfg["model_name"],
        pretrained=bool(cfg.get("pretrained", True)),
        num_classes=int(cfg["num_classes"]),
        img_size=int(cfg["img_size"]),
    )
    model.to(device)
    if cfg.get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    if cfg.get("compile", False):
        model = torch.compile(model)
    if cfg.get("dataparallel", True) and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ce = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.get("amp_dtype") == "fp16"))

    start_epoch = 1
    best_top1 = -1.0
    if cfg.get("resume_ckpt"):
        ckpt = torch.load(cfg["resume_ckpt"], map_location="cpu")
        target = model.module if isinstance(model, nn.DataParallel) else model
        target.load_state_dict(ckpt["model"], strict=False)
        if "opt" in ckpt:
            optimizer.load_state_dict(ckpt["opt"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_top1 = float(ckpt.get("best_top1", -1.0))

    history = []
    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        lr = cfg["lr"] * lr_multiplier(epoch, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        tr_loss, tr_top1, tr_top5 = train_one_epoch(model, dl_tr, optimizer, scaler, ce, device, epoch, cfg)
        va_loss, va_top1, va_top5, va_macro_f1, va_weighted_f1 = validate(model, dl_va, ce, device, cfg)
        metrics = {
            "loss": va_loss,
            "top1": va_top1,
            "top5": va_top5,
            "macro_f1": va_macro_f1,
            "weighted_f1": va_weighted_f1,
        }
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": tr_loss,
            "train_top1": tr_top1,
            "train_top5": tr_top5,
            **{f"val_{k}": v for k, v in metrics.items()},
        }
        history.append(row)
        print(f"[VAL] E{epoch:03d} top1={va_top1:.2f}% top5={va_top5:.2f}% "
              f"macro_f1={va_macro_f1:.2f}% weighted_f1={va_weighted_f1:.2f}% loss={va_loss:.4f}")
        with open(Path(cfg["out_dir"]) / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        save_checkpoint(Path(cfg["out_dir"]) / "last.pt", model, optimizer, scaler, epoch, metrics, cfg)
        if va_top1 > best_top1:
            best_top1 = va_top1
            save_checkpoint(Path(cfg["out_dir"]) / "best.pt", model, optimizer, scaler, epoch, metrics, cfg)
            with open(Path(cfg["out_dir"]) / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            print(f"[SAVE] best.pt top1={best_top1:.2f}%")
    print(f"[DONE] Best Top-1: {best_top1:.2f}% -> {Path(cfg['out_dir']) / 'best.pt'}")


if __name__ == "__main__":
    main()
