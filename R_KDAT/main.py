# -*- coding: utf-8 -*-
"""
Main (RegNetY-32GF + ArcFace 头 + Triplet(可选) + 在线知识蒸馏)
- 学生：可选 ArcFace（cfg.head.arcface/s/m），与 KD/MixUp/CutMix 兼容
- 老师：仅前向冻结；自动把 checkpoint 的 backbone.fc.* → head.* 键名映射
- AMP + AdamW + warmup+cosine
- 训练/验证：utils.train_loop.{train_one_epoch, validate}
- TensorBoard 记录；保存 best.pt（以 val top1）
"""
import os, time, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

# AMP GradScaler 兼容导入
try:
    from torch.cuda.amp import GradScaler
except Exception:
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


# ------------------------ Teacher 加载辅助函数 ------------------------
def _resolve_ckpt_path(path_str, cfg):
    """把 ~ / 环境变量 / 相对路径解析成可用的绝对路径。"""
    if not path_str:
        return None
    p = os.path.expanduser(os.path.expandvars(path_str))
    cand = [p]
    if not os.path.isabs(p):
        cand.append(os.path.join(os.getcwd(), p))
        cand.append(os.path.join(cfg.get("out_dir", "."), p))
        cand.append(os.path.join(os.path.dirname(cfg.get("out_dir", ".")), p))
    for c in cand:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def _strip_module_prefix(state_dict):
    """去掉 DataParallel 保存时的 'module.' 前缀。"""
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        from collections import OrderedDict
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            new_sd[k[7:] if k.startswith("module.") else k] = v
        return new_sd
    return state_dict


def _remap_fc_to_head(state_dict):
    """
    兼容旧权重：把 checkpoint 中的 'backbone.fc.{weight,bias}'
    重命名为当前模型的 'head.{weight,bias}'。
    """
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("backbone.fc.") for k in state_dict.keys()):
        return state_dict
    from collections import OrderedDict
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("backbone.fc."):
            new_k = k.replace("backbone.fc.", "head.")
            new_sd[new_k] = v
        else:
            new_sd[k] = v
    print("[KD] remap keys: backbone.fc.* → head.*")
    return new_sd


def _get_head_out_features(model: nn.Module):
    """获取模型分类头的 out_features（ArcFace 用 weight 行数）。"""
    core = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(core, "head") and core.head is not None:
        head = core.head
        if hasattr(head, "out_features"):
            return int(head.out_features)
        if hasattr(head, "weight"):
            # ArcFace/AM-Softmax 头
            return int(head.weight.size(0))
    # 兜底：某些老模型可能还在 backbone.fc
    if hasattr(core, "backbone") and hasattr(core.backbone, "fc"):
        fc = core.backbone.fc
        if hasattr(fc, "out_features"):
            return int(fc.out_features)
        if hasattr(fc, "weight"):
            return int(fc.weight.size(0))
    return None


def _teacher_head_matches(teacher: nn.Module, num_classes: int) -> bool:
    out_features = _get_head_out_features(teacher)
    if out_features is None:
        print("[KD][WARN] 无法获取 teacher 头维度，将禁用 KD。")
        return False
    if out_features != num_classes:
        print(f"[KD][WARN] teacher head out_features={out_features} != num_classes={num_classes}。")
        print("[KD][WARN] 为避免 KL 维度不匹配，本轮将禁用 KD。")
        return False
    return True


def _build_teacher_if_needed(cfg, num_classes, device):
    """按需构建并加载 teacher；若未启用 KD 或失败，则返回 None。"""
    kd_cfg = cfg.get("kd", {})
    if not kd_cfg or not kd_cfg.get("use", False):
        print("[KD] disabled.")
        return None

    raw_path = kd_cfg.get("teacher_ckpt", "")
    ckpt_path = _resolve_ckpt_path(raw_path, cfg)
    if ckpt_path is None:
        print(f"[KD] teacher_ckpt 未设置或文件不存在：{raw_path!r}")
        print(f"[KD] CWD={os.getcwd()}  (建议改为绝对路径或保证相对路径基于 CWD)")
        return None

    print(f"[KD] Loading teacher from: {ckpt_path}")
    # Teacher 一律用“普通线性头”（use_arcface=False）
    teacher = build_model(num_classes=num_classes, pretrained=True,
                          use_arcface=False).to(device)

    # 如需多卡
    if kd_cfg.get("dp_teacher", False) and torch.cuda.device_count() > 1:
        teacher = nn.DataParallel(teacher)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    state = _strip_module_prefix(state)
    state = _remap_fc_to_head(state)  # 关键：把 backbone.fc.* → head.*

    missing, unexpected = teacher.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[KD][WARN] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        print("[KD][MISS ex]", list(missing)[:5] if isinstance(missing, (list, tuple)) else missing)
        print("[KD][UNEXP ex]", list(unexpected)[:5] if isinstance(unexpected, (list, tuple)) else unexpected)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # 冻结 BN 统计，确保推理稳定
    if kd_cfg.get("freeze_bn", True):
        def set_bn_eval(m):
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
        teacher.apply(set_bn_eval)

    if not _teacher_head_matches(teacher, num_classes):
        return None

    print("[KD] teacher ready (eval, frozen).")
    return teacher
# ---------------------------------------------------------------------------


def main():
    cfg = CONFIG
    print("CONFIG OK.")
    set_seed(cfg.get("seed", 42))
    print("CWD =", os.getcwd())
    print("PYTHONPATH head =", sys.path[:3])
    print("CONFIG keys =", list(cfg.keys())[:5])

    # ==== 设备 ====
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cudnn.benchmark = True
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

    # ==== 学生模型（ArcFace 参数从 cfg.head 读取） ====
    h = cfg.get("head", {}) or {}
    model = build_model(
        num_classes=num_classes,
        pretrained=True,
        use_arcface=h.get("arcface", False),
        arc_s=h.get("s", 30.0),
        arc_m=h.get("m", 0.35),
    ).to(device)

    if cfg.get("dataparallel", False) and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # ==== Teacher（按需） ====
    teacher = _build_teacher_if_needed(cfg, num_classes, device)

    # ==== 优化器 / 损失 ====
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = build_loss_fn(cfg)   # 基础 CE/Focal；KD/Triplet 在 train_loop 内部处理

    # ==== AMP ====
    scaler = GradScaler(enabled=use_cuda) if use_cuda else None

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
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
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
            # 学习率（warmup + cosine）
            lr_mul = lr_warmup_cosine(
                epoch_idx=epoch - 1,
                base_lr=cfg["lr"],
                warmup_epochs=cfg["warmup_epochs"],
                total_epochs=cfg["epochs"],
                min_lr_ratio=cfg.get("cosine_min_lr_ratio", 0.05)
            )
            for pg in optimizer.param_groups:
                pg["lr"] = cfg["lr"] * lr_mul

            # 训练 1 轮（KD/Triplet 在 train_one_epoch 内部处理）
            train_loss, train_top1, train_top5 = train_one_epoch(
                model=model,
                dataloader=dl_tr,
                optimizer=optimizer,
                scaler=scaler,
                loss_fn=loss_fn,
                device=device,
                epoch=epoch,
                cfg=cfg,
                scheduler_fn=None,
                teacher=teacher  # 若为 None 则自动不做 KD
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
                    max_batches=cfg.get("val_max_batches", 0)
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
