# -*- coding: utf-8 -*-
"""
Evaluate the R_KDAT ArcFace checkpoint with the original training model.

Example:
  C:\Anaconda\envs\opencv_py38\python.exe eval\eval_rkdat_arcface.py ^
    --ckpt "best (3).pt" ^
    --val-dir "C:\CNFOOD-241\CNFOOD-241\val600x600"

Notes:
  - Uses R_KDAT.models.regnety32.build_model(use_arcface=True).
  - Uses R_KDAT.data.transforms.get_val_transforms, matching training validation.
  - Default metrics include both training-style batch-average Top-1/Top-5 and
    exact sample-average Top-1/Top-5.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
RKDAT_DIR = ROOT / "R_KDAT"
sys.path.insert(0, str(RKDAT_DIR))

from config import CONFIG  # noqa: E402
from data.transforms import get_val_transforms  # noqa: E402
from models.regnety32 import build_model  # noqa: E402
from utils.metrics import accuracy as batch_accuracy  # noqa: E402


def _pick_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "net", "module", "model_state", "ema_state", "model_ema"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value
    return ckpt


def _normalize_prefix_for_model(state: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    model_keys = set(model.state_dict().keys())
    state_keys = list(state.keys())

    if state_keys and all(k.startswith("module.") for k in state_keys):
        stripped = {k[len("module."):]: v for k, v in state.items()}
        stripped_hits = sum(1 for k in stripped if k in model_keys)
        original_hits = sum(1 for k in state if k in model_keys)
        if stripped_hits >= original_hits:
            return stripped

    return state


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, strict: bool = True) -> Tuple[dict, dict]:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    state = _pick_state_dict(ckpt)
    state = _normalize_prefix_for_model(state, model)
    msg = model.load_state_dict(state, strict=strict)
    return ckpt if isinstance(ckpt, dict) else {}, {
        "missing": list(getattr(msg, "missing_keys", [])),
        "unexpected": list(getattr(msg, "unexpected_keys", [])),
    }


@torch.no_grad()
def predict_logits(model: torch.nn.Module, images: torch.Tensor, hflip_tta: bool) -> torch.Tensor:
    logits = model(images)
    if not hflip_tta:
        return logits

    logits_flip = model(torch.flip(images, dims=[3]))
    probs = (F.softmax(logits, dim=1) + F.softmax(logits_flip, dim=1)) * 0.5
    return torch.log(probs.clamp_min(1e-12))


@torch.no_grad()
def evaluate(model, loader, device, hflip_tta: bool = False, use_amp: bool = True, max_batches: int = 0):
    model.eval()

    loss_sum = 0.0
    batch_top1_sum = 0.0
    batch_top5_sum = 0.0
    n_batches = 0

    correct1 = 0
    correct5 = 0
    n_samples = 0

    autocast_enabled = use_amp and device.type == "cuda"
    total = len(loader) if max_batches <= 0 else min(len(loader), max_batches)
    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="Validate", ncols=100, total=total)):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled):
            logits = predict_logits(model, images, hflip_tta=hflip_tta)
            loss = F.cross_entropy(logits, targets)

        top1, top5 = batch_accuracy(logits, targets, topk=(1, 5))
        loss_sum += float(loss.detach().item())
        batch_top1_sum += float(top1)
        batch_top5_sum += float(top5)
        n_batches += 1

        maxk = min(5, logits.size(1))
        pred = logits.topk(maxk, dim=1, largest=True, sorted=True).indices
        correct = pred.eq(targets.view(-1, 1))
        correct1 += int(correct[:, :1].any(dim=1).sum().item())
        correct5 += int(correct[:, :maxk].any(dim=1).sum().item())
        n_samples += int(targets.numel())

    return {
        "loss_batch_avg": loss_sum / max(1, n_batches),
        "top1_batch_avg_percent": batch_top1_sum / max(1, n_batches),
        "top5_batch_avg_percent": batch_top5_sum / max(1, n_batches),
        "top1_sample_avg": correct1 / max(1, n_samples),
        "top5_sample_avg": correct5 / max(1, n_samples),
        "num_samples": n_samples,
        "num_batches": n_batches,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate R_KDAT ArcFace checkpoint.")
    parser.add_argument("--ckpt", default=str(ROOT / "best (3).pt"), help="Checkpoint path.")
    parser.add_argument("--val-dir", default=CONFIG["val_dir"], help="Validation ImageFolder directory.")
    parser.add_argument("--num-classes", type=int, default=int(CONFIG["num_classes"]))
    parser.add_argument("--img-size", type=int, default=int(CONFIG["img_size"]))
    parser.add_argument("--batch-size", type=int, default=int(CONFIG["batch_size"]))
    parser.add_argument("--num-workers", type=int, default=int(CONFIG["num_workers"]))
    parser.add_argument("--arc-s", type=float, default=float((CONFIG.get("head") or {}).get("s", 30.0)))
    parser.add_argument("--arc-m", type=float, default=float((CONFIG.get("head") or {}).get("m", 0.35)))
    parser.add_argument("--hflip-tta", action="store_true", help="Optional hflip TTA. Training validation did not use this.")
    parser.add_argument("--max-batches", type=int, default=int(CONFIG.get("val_max_batches", 0)))
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA AMP during evaluation.")
    parser.add_argument("--non-strict", action="store_true", help="Use strict=False for checkpoint loading.")
    parser.add_argument("--out-json", default="", help="Optional path to save metrics JSON.")
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_path = str(Path(args.ckpt).expanduser())
    val_dir = str(Path(args.val_dir).expanduser())
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageFolder(val_dir, transform=get_val_transforms(args.img_size))
    if len(dataset.classes) != args.num_classes:
        print(f"[WARN] num_classes={args.num_classes}, dataset classes={len(dataset.classes)}; using dataset value.")
        args.num_classes = len(dataset.classes)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(
        num_classes=args.num_classes,
        pretrained=False,
        use_arcface=True,
        arc_s=args.arc_s,
        arc_m=args.arc_m,
    ).to(device)

    ckpt_meta, load_info = load_checkpoint(model, ckpt_path, strict=not args.non_strict)
    print(f"[LOAD] {ckpt_path}")
    print(f"[LOAD] missing={len(load_info['missing'])} unexpected={len(load_info['unexpected'])}")
    if load_info["missing"]:
        print("[LOAD] missing sample:", load_info["missing"][:10])
    if load_info["unexpected"]:
        print("[LOAD] unexpected sample:", load_info["unexpected"][:10])

    metrics = evaluate(
        model,
        loader,
        device,
        hflip_tta=args.hflip_tta,
        use_amp=not args.no_amp,
        max_batches=args.max_batches,
    )
    result = {
        "checkpoint": ckpt_path,
        "val_dir": val_dir,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "hflip_tta": bool(args.hflip_tta),
        "max_batches": args.max_batches,
        "checkpoint_epoch": ckpt_meta.get("epoch"),
        "checkpoint_best_top1_percent": ckpt_meta.get("best_top1"),
        **metrics,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        "[RESULT] train-style Top1={:.4f}% Top5={:.4f}% | sample Top1={:.6f} Top5={:.6f}".format(
            result["top1_batch_avg_percent"],
            result["top5_batch_avg_percent"],
            result["top1_sample_avg"],
            result["top5_sample_avg"],
        )
    )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    main()
