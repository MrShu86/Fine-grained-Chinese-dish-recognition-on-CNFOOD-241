# -*- coding: utf-8 -*-
"""
Main (RegNetY-32GF 单塔基线)
- 读取 data.cnfood_dataset.build_loaders 提供的 DataLoader
- 使用 models.regnety32.build_model 构建分类模型
- AMP + AdamW + warmup+cosine（由 utils.schedulers.lr_warmup_cosine 提供 LR 比例）
- 训练/验证循环调用 utils.train_loop.{train_one_epoch, validate}
- 记录 top1/top5 到 TensorBoard（可在 config 中开关）
- 支持断点恢复（resume_ckpt），默认保存 best.pt（以 val top1 为准）
"""
import argparse
import copy
import os, time, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

# AMP GradScaler 兼容导入（老/新 PyTorch 都能跑）
try:
    from torch.cuda.amp import GradScaler  # 常见于 PyTorch 1.x/2.0/2.1
except Exception:  # 极少数环境使用 torch.amp.GradScaler
    from torch.amp import GradScaler

from config import CONFIG
from models.regnety32 import build_model
from data.cnfood_dataset import build_loaders
from utils.losses import build_loss_fn
from utils.schedulers import lr_warmup_cosine
from utils.train_loop import train_one_epoch, validate


def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train RegNetY-32GF baseline.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "off"], default=None)
    parser.add_argument("--channels-last", dest="channels_last", action="store_true", default=None)
    parser.add_argument("--no-channels-last", dest="channels_last", action="store_false")
    parser.add_argument("--fused-adamw", dest="fused_adamw", action="store_true", default=None)
    parser.add_argument("--no-fused-adamw", dest="fused_adamw", action="store_false")
    parser.add_argument("--compile", dest="compile_model", action="store_true", default=None)
    parser.add_argument("--no-compile", dest="compile_model", action="store_false")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic CuDNN settings for repeatability.")
    return parser.parse_args()


def apply_overrides(cfg, args):
    cfg = copy.deepcopy(cfg)
    mapping = {
        "seed": args.seed,
        "out_dir": args.out_dir,
        "resume_ckpt": args.resume_ckpt,
        "train_dir": args.train_dir,
        "val_dir": args.val_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "amp_dtype": args.amp_dtype,
        "channels_last": args.channels_last,
        "fused_adamw": args.fused_adamw,
    }
    for key, value in mapping.items():
        if value is not None:
            cfg[key] = value
    if args.no_tensorboard:
        cfg["use_tensorboard"] = False
    if args.compile_model is not None:
        cfg["compile"] = args.compile_model
    return cfg


def make_optimizer(model, cfg, use_cuda):
    kwargs = dict(lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if cfg.get("channels_last", False) and cfg.get("fused_adamw", False):
        print("[WARN] fused AdamW is disabled because channels_last can create non-standard parameter layouts.")
        print("[WARN] Use regular AdamW for A100 stability.")
        return optim.AdamW(model.parameters(), **kwargs)
    if use_cuda and cfg.get("fused_adamw", False):
        try:
            return optim.AdamW(model.parameters(), fused=True, **kwargs)
        except TypeError:
            print("[WARN] fused AdamW is not available in this PyTorch; fallback to AdamW.")
    return optim.AdamW(model.parameters(), **kwargs)


def main():
    args = parse_args()
    cfg = apply_overrides(CONFIG, args)
    print("CONFIG OK.")
    set_seed(cfg.get("seed", 42))
    print("SEED =", cfg.get("seed", 42))
    print("OUT_DIR =", cfg.get("out_dir"))
    print("CWD =", os.getcwd())
    print("PYTHONPATH head =", sys.path[:3])
    print("CONFIG keys =", list(cfg.keys())[:5])

    # ==== 设备与 CUDNN ====
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cudnn.benchmark = not args.deterministic
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.deterministic:
        cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    if use_cuda:
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        print(f"CUDA: True  GPUs={torch.cuda.device_count()}  Name={torch.cuda.get_device_name(0)}")
    else:
        print("CUDA: False (use CPU)")

    # ==== 数据 ====
    ds_tr, ds_va, dl_tr, dl_va = build_loaders(cfg)
    num_classes = cfg["num_classes"]

    # ==== 模型 ====
    model = build_model(num_classes=num_classes, pretrained=True).to(device)
    if cfg.get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)
        print("[A100] channels_last enabled.")
    if cfg.get("compile", False):
        try:
            model = torch.compile(model)
            print("[A100] torch.compile enabled.")
        except Exception as exc:
            print(f"[WARN] torch.compile failed, continue without compile: {exc}")
    if cfg.get("dataparallel", False) and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # ==== 优化器 / 损失 ====
    optimizer = make_optimizer(model, cfg, use_cuda)
    loss_fn = build_loss_fn(cfg)

    # ==== AMP ====
    amp_dtype = str(cfg.get("amp_dtype", "bf16")).lower()
    scaler = GradScaler(enabled=(use_cuda and amp_dtype == "fp16")) if use_cuda else None
    print(f"[A100] amp_dtype={amp_dtype}  GradScaler={'on' if scaler is not None and scaler.is_enabled() else 'off'}")

    # ==== 日志 ====
    os.makedirs(cfg["out_dir"], exist_ok=True)
    writer = None
    if cfg.get("use_tensorboard", True):
        run_dir = os.path.join(cfg["out_dir"], "runs", time.strftime("%Y%m%d-%H%M%S"))
        writer = SummaryWriter(run_dir)
        print(f"[TB] logdir: {run_dir}")

    # ==== 断点恢复 ====
    start_epoch, best_top1 = 1, 0.0
    best_path = os.path.join(cfg["out_dir"], "best.pt")
    resume_path = cfg.get("resume_ckpt", "")
    if resume_path and os.path.isfile(resume_path):
        print(f"[INFO] Resume from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        # 兼容 {'model': state_dict} 或完整 checkpoint
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
        if "opt" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["opt"])
            except Exception:
                print("[WARN] Optimizer state 不兼容，已跳过。")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_top1 = ckpt.get("best_top1", 0.0)
        if "scaler" in ckpt and ckpt["scaler"] is not None and scaler is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                print("[WARN] AMP scaler 状态不兼容，已忽略。")

    # ==== 训练循环 ====
    try:
        for epoch in range(start_epoch, cfg["epochs"] + 1):
            # 设置当轮学习率：warmup + cosine（返回倍率，乘以 base lr）
            lr_mul = lr_warmup_cosine(
                epoch_idx=epoch - 1,
                base_lr=cfg["lr"],
                warmup_epochs=cfg["warmup_epochs"],
                total_epochs=cfg["epochs"],
                min_lr_ratio=cfg.get("cosine_min_lr_ratio", 0.05)
            )
            for pg in optimizer.param_groups:
                pg["lr"] = cfg["lr"] * lr_mul

            # 训练 1 轮
            train_loss, train_top1, train_top5 = train_one_epoch(
                model=model,
                dataloader=dl_tr,
                optimizer=optimizer,
                scaler=scaler,
                loss_fn=loss_fn,
                device=device,
                epoch=epoch,
                cfg=cfg,
                scheduler_fn=None  # 我们已在外部手动设置 lr
            )

            if writer:
                writer.add_scalar("train/loss", train_loss, epoch)
                writer.add_scalar("train/top1", train_top1, epoch)
                writer.add_scalar("train/top5", train_top5, epoch)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

            # 验证（按频率）
            do_val = (epoch % cfg["eval_every"] == 0) or (epoch == cfg["epochs"])
            if do_val:
                val_loss, val_top1, val_top5 = validate(
                    model=model,
                    dataloader=dl_va,
                    device=device,
                    max_batches=cfg.get("val_max_batches", 0)  # 0 表示全量
                )
                print(f"[VAL] E{epoch:03d}  top1={val_top1:.2f}%  top5={val_top5:.2f}%  loss={val_loss:.4f}")

                if writer:
                    writer.add_scalar("val/loss", val_loss, epoch)
                    writer.add_scalar("val/top1", val_top1, epoch)
                    writer.add_scalar("val/top5", val_top5, epoch)

                # 保存 best
                if val_top1 > best_top1:
                    best_top1 = val_top1
                    torch.save({
                        "model": model.state_dict(),
                        "opt": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_top1": best_top1,
                        "scaler": (scaler.state_dict() if scaler is not None else None)
                    }, best_path)
                    print(f"[SAVE] best.pt  (top1={best_top1:.2f}%)")

    except KeyboardInterrupt:
        # 支持 Ctrl+C 时保存一个 last.pt
        last_path = os.path.join(cfg["out_dir"], "last.pt")
        torch.save({
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "epoch": epoch,
            "best_top1": best_top1,
            "scaler": (scaler.state_dict() if scaler is not None else None)
        }, last_path)
        print(f"\n[INTERRUPT] 已保存 last.pt → {last_path}")

    if writer:
        writer.close()
    print(f"[DONE] Best Top-1: {best_top1:.2f}%  → {best_path}")


if __name__ == "__main__":
    main()
