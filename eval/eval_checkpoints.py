# -*- coding: utf-8 -*-
"""
批量评估 RegNetY-32GF 在 CNFOOD-241 上的权重质量（Top1/Top5、P/R/F1、ROC-AUC），并输出可视化：
- 归一化混淆矩阵（PNG）
- 各类别 F1 柱状图（最好/最差 Top-K）
- 高置信错误样本拼图（Top 错判）
- 逐样本预测（含 Top-5）CSV、最混淆类对 CSV
- 可靠性图 + ECE、置信度直方图、决策边际直方图、低边际样本拼图
- t-SNE 特征可视化（最差 K 类）
- 大类（super-class）聚合混淆矩阵与 F1
- 新增：覆盖率–准确率曲线、长尾诊断、Top-5 召回（最差类）、每类 ECE、类中心相似度热图、逐类难例拼图

依赖：
  pip install torch torchvision timm scikit-learn matplotlib pandas pillow tqdm
"""

import argparse
import json
import os
import warnings
from glob import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from sklearn.metrics import (
    accuracy_score, top_k_accuracy_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve, auc, f1_score
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


# ============ 默认路径 & 常用开关（直接改这里最方便） ============
DEFAULTS = dict(
    VAL_DIR=r"C:\CNFOOD-241\CNFOOD-241\val600x600",
    CKPTS=[
        r"C:\Users\11138\Downloads\best (3).pt",
        # r"C:\Users\11138\Downloads\exp_regnety32\best*.pt",
    ],
    OUT_DIR=r".\eval_outputs",
    MODEL_NAME="tv_regnety32_wrapper",  # 使用你的自定义封装，避免 state_dict 对不上
    NUM_CLASSES=241,
    IMG_SIZE=300,
    BATCH_SIZE=128,
    NUM_WORKERS=4,
    HFLIP_TTA=True,
    SEED=42,
    F1_TOPK=50,
    MAX_ERROR_GRID=24,

    # —— 可视化 / 导出控制 ——
    MAKE_PRED_CSV=True,
    MAKE_TOP_CONFUSIONS=True,
    MAKE_WORST_F1=True,
    MAKE_CALIB=True,           # ECE + 可靠性图
    MAKE_CONF_HIST=True,       # 置信度直方图（对/错）
    MAKE_MARGIN_HIST=True,     # 决策边际直方图
    MAKE_LOW_MARGIN_GRID=True, # 低边际样本拼图

    # t-SNE（可选，开启可能耗时）
    MAKE_TSNE=True,
    TSNE_WORST_K=15,
    TSNE_MAX_PER_CLASS=120,
    TSNE_PERPLEXITY=30,

    # 可选：大类映射CSV（两列：class,super_class），提供后输出大类混淆矩阵与 F1
    SUPERCLASS_CSV=None,  # 例如 r"C:\CNFOOD-241\superclass_map.csv"
)
# ============================================================


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# === 与训练一致的模型封装（torchvision RegNetY-32GF + 分类头在 backbone.fc） ===
try:
    from torchvision.models import regnet_y_32gf, RegNet_Y_32GF_Weights
except Exception:
    from torchvision.models import regnet_y_32gf
    RegNet_Y_32GF_Weights = None


class RegNetY32(nn.Module):
    def __init__(self,
                 num_classes: int = 241,
                 pretrained: bool = False,
                 dropout: float = 0.0,
                 return_features: bool = False):
        super().__init__()
        if pretrained and RegNet_Y_32GF_Weights is not None:
            weights = RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1
        else:
            weights = None
        self.backbone = regnet_y_32gf(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)
        self.head_dropout = nn.Dropout(p=dropout) if (dropout and dropout > 0) else nn.Identity()
        self.num_classes = num_classes
        self.feature_dim = in_feats
        self.return_features_default = return_features

    def forward(self, x: torch.Tensor, return_features: bool = None):
        x = self.backbone.stem(x)
        x = self.backbone.trunk_output(x)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)
        feats = self.head_dropout(feats)
        logits = self.backbone.fc(feats)
        rf = self.return_features_default if return_features is None else return_features
        if rf:
            return logits, feats
        return logits


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name in ["tv_regnety32_wrapper", "regnet_y32_wrapper"]:
        return RegNetY32(num_classes=num_classes, pretrained=False)

    model = None
    try:
        import timm
        if model_name in timm.list_models(pretrained=False):
            model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        else:
            alt_names = [model_name.replace("-", "_"), model_name.replace("_", "-")]
            for n in alt_names:
                if n in timm.list_models(pretrained=False):
                    model = timm.create_model(n, pretrained=False, num_classes=num_classes)
                    break
    except Exception:
        pass
    if model is None:
        from torchvision.models import regnet_y_32gf
        model = regnet_y_32gf(weights=None, num_classes=num_classes)
    return model


def load_state_dict_safely(model: nn.Module, ckpt_path: str) -> Tuple[List[str], List[str]]:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    candidates = ["state_dict", "model", "net", "module", "model_state", "ema_state", "model_ema", "backbone"]
    sd = None
    if isinstance(ckpt, dict):
        for k in candidates:
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]; break
        if sd is None:
            for v in ckpt.values():
                if isinstance(v, dict):
                    sd = v; break
        if sd is None:
            sd = ckpt
    else:
        sd = ckpt

    strip_prefixes = ["module.", "model.", "network.", "net.", "encoder.", "model_ema."]
    if not hasattr(model, "backbone"):
        strip_prefixes.append("backbone.")

    def strip_pref(name: str) -> str:
        for p in strip_prefixes:
            if name.startswith(p):
                return name[len(p):]
        return name

    new_sd = {strip_pref(k): v for k, v in sd.items()}

    msg = model.load_state_dict(new_sd, strict=False)
    missing = list(msg.missing_keys)
    unexpected = list(msg.unexpected_keys)
    if missing or unexpected:
        print(f"[state_dict] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("[state_dict] missing(samples):", missing[:10])
        if unexpected:
            print("[state_dict] unexpected(samples):", unexpected[:10])
    return missing, unexpected


def build_transforms(img_size: int, center_crop: bool = True):
    resizer = transforms.Resize(int(img_size * 1.15))
    cropper = transforms.CenterCrop(img_size) if center_crop else transforms.Resize(img_size)
    return transforms.Compose([
        resizer, cropper, transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


@torch.no_grad()
def model_predict_logits(model: nn.Module, images: torch.Tensor, hflip_tta: bool = False, want_feats: bool = False):
    if want_feats and hasattr(model, "return_features_default"):
        out = model(images, return_features=True)
        if isinstance(out, tuple) and len(out) == 2:
            logits, feats = out
        else:
            logits, feats = out, None
    else:
        logits, feats = model(images), None

    if not hflip_tta:
        return logits, feats

    flipped = torch.flip(images, dims=[3])
    logits_flip = model(flipped) if not want_feats else model(flipped, return_features=False)
    probs = F.softmax(logits, dim=1) + F.softmax(logits_flip, dim=1)
    logits = torch.log(probs / 2.0 + 1e-12)
    return logits, feats


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], normalize: bool, out_png: str, figsize=(14, 12)):
    cm_ = cm.astype('float')
    if normalize:
        with np.errstate(all='ignore'):
            cm_ = cm_ / cm_.sum(axis=1, keepdims=True)
            cm_[np.isnan(cm_)] = 0.0

    plt.figure(figsize=figsize)
    im = plt.imshow(cm_, interpolation='nearest')
    plt.title('Confusion Matrix (normalized)' if normalize else 'Confusion Matrix (counts)')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel('Predicted'); plt.ylabel('True')
    step = max(1, len(classes) // 20)
    ticks = list(range(0, len(classes), step))
    plt.xticks(ticks, [classes[i] for i in ticks], rotation=90)
    plt.yticks(ticks, [classes[i] for i in ticks])
    plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()


def plot_class_f1(f1_dict: Dict[str, float], out_png: str, top_k: int = 50):
    items = sorted(f1_dict.items(), key=lambda x: x[1], reverse=True)
    if top_k is not None:
        items = items[:top_k]
    labels, vals = zip(*items) if items else ([], [])
    plt.figure(figsize=(max(10, len(labels) * 0.2), 5))
    plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylabel("F1"); plt.title(f"Top-{top_k} Class F1" if top_k else "Class F1")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def plot_class_f1_worst(f1_dict: Dict[str, float], out_png: str, bottom_k: int = 30):
    items = sorted(f1_dict.items(), key=lambda x: x[1])[:bottom_k]
    labels, vals = zip(*items) if items else ([], [])
    plt.figure(figsize=(max(10, len(labels)*0.25), 5))
    plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylabel("F1"); plt.title(f"Worst-{bottom_k} Class F1")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def save_predictions_csv(dataset, img_indices, y_true, y_pred, prob_pred, class_names, out_csv):
    rows = []
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    for i, ds_idx in enumerate(img_indices):
        if ds_idx >= len(dataset.samples):
            continue
        path = dataset.samples[ds_idx][0]
        pred_id = int(y_pred[i]); true_id = int(y_true[i])
        conf = float(prob_pred[i, pred_id])
        top5_idx = np.argsort(-prob_pred[i])[:5]
        top5_labels = [class_names[j] for j in top5_idx]
        top5_probs  = [float(prob_pred[i, j]) for j in top5_idx]
        rows.append({
            "path": path,
            "true_id": true_id, "true_label": class_names[true_id],
            "pred_id": pred_id, "pred_label": class_names[pred_id],
            "pred_conf": conf,
            "top5_ids": ";".join(map(str, top5_idx.tolist())),
            "top5_labels": ";".join(top5_labels),
            "top5_probs": ";".join([f"{p:.6f}" for p in top5_probs]),
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")


def export_top_confusions(cm, classes, out_csv, k=50):
    cm = np.asarray(cm)
    n = cm.shape[0]
    pairs = []
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums
    for t in range(n):
        for p in range(n):
            if t == p:
                continue
            cnt = int(cm[t, p])
            if cnt <= 0:
                continue
            pairs.append({
                "true_id": t, "true_label": classes[t],
                "pred_id": p, "pred_label": classes[p],
                "count": cnt,
                "true_class_total": int(row_sums[t, 0]),
                "rate_in_true": float(cm_norm[t, p]),
            })
    df = pd.DataFrame(pairs)
    if df.empty:
        pd.DataFrame(columns=["true_id","true_label","pred_id","pred_label","count","true_class_total","rate_in_true"]).to_csv(out_csv, index=False, encoding="utf-8-sig")
        return
    df.sort_values(["count", "rate_in_true"], ascending=[False, False], inplace=True)
    df.head(k).to_csv(out_csv, index=False, encoding="utf-8-sig")


def plot_confidence_hist(prob_pred, y_true, out_png):
    y_true = np.asarray(y_true).astype(int)
    conf = prob_pred.max(axis=1)
    pred = prob_pred.argmax(axis=1)
    correct = (pred == y_true)
    plt.figure(figsize=(7,5))
    plt.hist(conf[correct], bins=20, alpha=0.7, label="Correct")
    plt.hist(conf[~correct], bins=20, alpha=0.7, label="Incorrect")
    plt.xlabel("Predicted confidence"); plt.ylabel("Count"); plt.title("Confidence Histogram")
    plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def plot_margin_hist(prob_pred, out_png):
    sorted_probs = -np.sort(-prob_pred, axis=1)
    margins = sorted_probs[:,0] - sorted_probs[:,1]
    plt.figure(figsize=(7,5))
    plt.hist(margins, bins=30)
    plt.xlabel("Top1 - Top2 margin"); plt.ylabel("Count"); plt.title("Decision Margin Histogram")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
    return margins


def save_low_margin_grid(dataset, img_indices, margins, max_show, out_png, img_size=224, grid_cols=6):
    from PIL import Image
    order = np.argsort(margins)  # 最小边际在前
    pick = order[:max_show]
    imgs = []
    for idx in pick:
        ds_idx = img_indices[idx]
        if ds_idx >= len(dataset.samples):
            continue
        p = dataset.samples[ds_idx][0]
        try:
            img = Image.open(p).convert("RGB").resize((img_size, img_size))
            imgs.append(transforms.ToTensor()(img))
        except Exception:
            continue
    if not imgs:
        return
    grid = make_grid(torch.stack(imgs, dim=0), nrow=grid_cols, padding=2)
    save_image(grid, out_png)


def compute_ece(prob_pred, y_true, n_bins=15):
    y_true = np.asarray(y_true).astype(int)
    conf = prob_pred.max(axis=1)
    pred = prob_pred.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b+1]
        mask = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece += np.abs(acc_bin - conf_bin) * (mask.mean())
    return float(ece)


def plot_reliability(prob_pred, y_true, out_png, n_bins=15):
    y_true = np.asarray(y_true).astype(int)
    conf = prob_pred.max(axis=1)
    pred = prob_pred.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    xs, accs, confs = [], [], []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b+1]
        mask = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        xs.append((lo+hi)/2)
        accs.append(correct[mask].mean())
        confs.append(conf[mask].mean())
    plt.figure(figsize=(6,5))
    plt.plot([0,1], [0,1], linestyle="--")
    plt.plot(confs, accs, marker="o")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title("Reliability Diagram")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def plot_roc_pr_micro(prob_pred, y_true, out_roc_png, out_pr_png):
    y_true = np.asarray(y_true).astype(int)
    n_classes = prob_pred.shape[1]
    Y = label_binarize(y_true, classes=list(range(n_classes)))
    fpr, tpr, _ = roc_curve(Y.ravel(), prob_pred.ravel())
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot([0,1], [0,1], linestyle="--")
    plt.plot(fpr, tpr)
    plt.title(f"Micro-averaged ROC (AUC={roc_auc:.4f})")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.tight_layout(); plt.savefig(out_roc_png, dpi=200); plt.close()
    precision, recall, _ = precision_recall_curve(Y.ravel(), prob_pred.ravel())
    ap = average_precision_score(Y, prob_pred, average="micro")
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision)
    plt.title(f"Micro-averaged PR (AP={ap:.4f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.tight_layout(); plt.savefig(out_pr_png, dpi=200); plt.close()


def plot_confusion_submatrix_worst(cm, class_names, report, out_png, worst_rows=20, top_pred_cols=20):
    cm = np.asarray(cm)
    f1_per_class = {k: v["f1-score"] for k, v in report.items() if k not in ["accuracy","macro avg","weighted avg"]}
    name_to_id = {name: i for i, name in enumerate(class_names)}
    worst_names = [k for k,_ in sorted(f1_per_class.items(), key=lambda x:x[1])[:worst_rows]]
    row_ids = [name_to_id[n] for n in worst_names]
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums
    col_scores = cm_norm[row_ids, :].sum(axis=0)
    col_ids = np.argsort(-col_scores)[:top_pred_cols]
    sub = cm_norm[np.ix_(row_ids, col_ids)]
    plt.figure(figsize=(max(10, top_pred_cols*0.5), max(6, worst_rows*0.35)))
    plt.imshow(sub, interpolation='nearest')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.yticks(range(len(row_ids)), [class_names[i] for i in row_ids])
    plt.xticks(range(len(col_ids)), [class_names[j] for j in col_ids], rotation=90)
    plt.title("Confusions on Worst-F1 Classes (row-normalized)")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def load_superclass_mapping(csv_path: Optional[str], class_names: List[str]):
    if not csv_path or not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if "class" not in cols or "super_class" not in cols:
        return None
    ccol, scol = cols["class"], cols["super_class"]
    name_to_super = dict(zip(df[ccol].astype(str), df[scol].astype(str)))
    super_names = sorted(list(set(name_to_super.values())))
    class_to_super_id = []
    used_other = False
    for cname in class_names:
        sname = name_to_super.get(cname, "__OTHER__")
        if sname not in super_names:
            super_names.append(sname)
        if sname == "__OTHER__":
            used_other = True
        class_to_super_id.append(super_names.index(sname))
    if used_other and "__OTHER__" not in super_names:
        super_names.append("__OTHER__")
    return super_names, np.array(class_to_super_id, dtype=int)


def aggregate_super_confusion(y_true, y_pred, class_to_super_id, super_names):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_true_s = class_to_super_id[y_true]
    y_pred_s = class_to_super_id[y_pred]
    n = len(super_names)
    cm_s = np.zeros((n,n), dtype=int)
    for t,p in zip(y_true_s, y_pred_s):
        cm_s[t,p] += 1
    return cm_s, y_true_s, y_pred_s


@torch.no_grad()
def save_top_errors_grid(dataset: datasets.ImageFolder,
                         img_indices: List[int],
                         y_true: List[int],
                         y_pred: List[int],
                         prob_pred: np.ndarray,
                         classes: List[str],
                         out_png: str,
                         grid_cols: int = 6,
                         max_show: int = 24,
                         img_size: int = 224):
    """
    从验证集里挑选“置信度最高但预测错误”的样本做拼图保存，便于排查难例/脏数据。
    """
    from PIL import Image

    if max_show <= 0:
        return

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    wrong_mask = (y_true_np != y_pred_np)
    wrong_idxs = np.where(wrong_mask)[0]
    if wrong_idxs.size == 0:
        return

    conf_for_wrong = prob_pred[wrong_idxs, y_pred_np[wrong_idxs]]
    order = np.argsort(-conf_for_wrong)
    pick = order[:max_show]
    chosen_wrong = wrong_idxs[pick]

    imgs = []
    for idx in chosen_wrong:
        ds_idx = img_indices[idx]
        if ds_idx >= len(dataset.samples):
            continue
        path = dataset.samples[ds_idx][0]
        try:
            img = Image.open(path).convert("RGB").resize((img_size, img_size))
            imgs.append(transforms.ToTensor()(img))
        except Exception:
            continue

    if not imgs:
        return

    grid = make_grid(torch.stack(imgs, dim=0), nrow=grid_cols, padding=2)
    save_image(grid, out_png)

    # 可选：另存一张带标题的图
    try:
        plt.figure(figsize=(12, 10))
        plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
        plt.axis('off')
        plt.title("Top-Confidence Misclassifications (T:true, P:pred)")
        plt.tight_layout()
        plt.savefig(out_png.replace(".png", "_with_text.png"), dpi=200)
        plt.close()
    except Exception:
        pass


# ====== 新增：部署阈值/长尾/Top5/类别ECE/类中心相似度/逐类难例 可视化 ======

def plot_coverage_accuracy_curve(prob_pred, y_true, out_png, out_csv, steps=200):
    """
    覆盖率-准确率曲线：阈值升高 -> 覆盖率下降、准确率上升。
    同时导出 CSV，并给出 90%/95% 覆盖的推荐阈值（JSON 附带）。
    """
    y_true = np.asarray(y_true).astype(int)
    conf = prob_pred.max(axis=1)
    pred = prob_pred.argmax(axis=1)
    correct = (pred == y_true).astype(int)

    thrs = np.linspace(0.0, 1.0, steps)
    cover, acc = [], []
    for t in thrs:
        mask = conf >= t
        cov = mask.mean()
        if mask.any():
            acc_t = correct[mask].mean()
        else:
            acc_t = np.nan
        cover.append(cov); acc.append(acc_t)

    def find_thr_for_cover(target_cover):
        idx = np.where(np.array(cover) >= target_cover)[0]
        if len(idx) == 0:
            return None
        i = idx[0]
        return float(thrs[i]), float(cover[i]), float(acc[i])

    rec_90 = find_thr_for_cover(0.90)
    rec_95 = find_thr_for_cover(0.95)

    df = pd.DataFrame({"threshold": thrs, "coverage": cover, "accuracy": acc})
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(7,5))
    plt.plot(cover, acc)
    plt.xlabel("Coverage (retain rate)")
    plt.ylabel("Accuracy")
    title = "Coverage–Accuracy Curve"
    if rec_90 or rec_95: title += "  (recs marked)"
    plt.title(title)
    for tag, rec in [("90%", rec_90), ("95%", rec_95)]:
        if rec:
            thr, cov, ac = rec
            plt.scatter([cov], [ac])
            plt.text(cov, ac, f" {tag}@t={thr:.3f}\nacc={ac:.3f}", va="bottom")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

    recs = {"rec_90": None, "rec_95": None}
    if rec_90: recs["rec_90"] = {"thr": rec_90[0], "coverage": rec_90[1], "accuracy": rec_90[2]}
    if rec_95: recs["rec_95"] = {"thr": rec_95[0], "coverage": rec_95[1], "accuracy": rec_95[2]}
    with open(out_png.replace(".png", ".json"), "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)


def plot_support_vs_f1(report, out_png):
    """类别样本量 vs F1 散点图（对数横轴），用于判断长尾是否主要由“样本量不足”导致。"""
    cls_items = [(k, v) for k, v in report.items() if k not in ["accuracy","macro avg","weighted avg"]]
    supports = np.array([v["support"] for _,v in cls_items], dtype=float)
    f1s = np.array([v["f1-score"] for _,v in cls_items], dtype=float)

    plt.figure(figsize=(7,5))
    plt.scatter(supports + 1e-6, f1s, s=12)
    plt.xscale("log")
    plt.xlabel("Class support (log scale)")
    plt.ylabel("F1")
    plt.title("Support vs F1 (per class)")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def plot_topk_recall_worst(prob_pred, y_true, class_names, out_png, top_k=5, bottom_k=30):
    """统计每类 Top-K Recall（真类是否在Top-K候选里），画最差 bottom_k 类柱状图。"""
    y_true = np.asarray(y_true).astype(int)
    n = prob_pred.shape[1]
    topk_idx = np.argpartition(-prob_pred, kth=top_k-1, axis=1)[:, :top_k]
    topk_hit = np.zeros(n, dtype=float)
    count = np.zeros(n, dtype=float)
    for i, t in enumerate(y_true):
        count[t] += 1
        if t in topk_idx[i]:
            topk_hit[t] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        topk_rec = np.where(count>0, topk_hit / count, 0.0)

    items = sorted([(class_names[i], topk_rec[i]) for i in range(n)], key=lambda x:x[1])[:bottom_k]
    labels, vals = zip(*items) if items else ([], [])
    plt.figure(figsize=(max(10, len(labels)*0.25), 5))
    plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylabel(f"Top-{top_k} Recall"); plt.title(f"Worst-{bottom_k} Top-{top_k} Recall (per class)")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def compute_ece_per_class(prob_pred, y_true, n_bins=15):
    """
    One-vs-Rest 每类 ECE：衡量 prob[:,c] 作为“属于类 c 的概率”是否校准良好。
    """
    y_true = np.asarray(y_true).astype(int)
    n = prob_pred.shape[1]
    eces = {}
    bins = np.linspace(0.0, 1.0, n_bins+1)
    for c in range(n):
        y_bin = (y_true == c).astype(float)
        conf_all = prob_pred[:, c]
        ece = 0.0
        for b in range(n_bins):
            lo, hi = bins[b], bins[b+1]
            m = (conf_all > lo) & (conf_all <= hi) if b>0 else (conf_all >= lo) & (conf_all <= hi)
            if not np.any(m):
                continue
            acc_bin = y_bin[m].mean()
            conf_bin = conf_all[m].mean()
            ece += abs(acc_bin - conf_bin) * (m.mean())
        eces[c] = float(ece)
    return eces


def plot_class_ece_worst(prob_pred, y_true, class_names, out_png, out_csv, n_bins=15, bottom_k=30):
    eces = compute_ece_per_class(prob_pred, y_true, n_bins=n_bins)
    pairs = [(class_names[i], eces[i]) for i in range(len(class_names)) if not np.isnan(eces[i])]
    pairs.sort(key=lambda x: x[1], reverse=True)  # ECE 越大越差
    sel = pairs[:bottom_k]
    labels = [p[0] for p in sel]; vals = [p[1] for p in sel]
    plt.figure(figsize=(max(10, len(labels)*0.25), 5))
    plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylabel("ECE"); plt.title(f"Worst-{bottom_k} Class ECE (OvR)")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
    pd.DataFrame({"class": [p[0] for p in pairs], "ece": [p[1] for p in pairs]}).to_csv(out_csv, index=False, encoding="utf-8-sig")


def plot_centroid_cosine_heatmap(feats_np, y_true, class_names, out_png, focus_class_ids=None, max_classes=40):
    """
    基于特征的“类中心”余弦相似度热图。默认对 focus_class_ids 画；若为 None，则选样本数最多的若干类。
    """
    if feats_np is None:
        return
    y_true = np.asarray(y_true).astype(int)
    n_cls = len(class_names)

    if focus_class_ids is None:
        counts = np.bincount(y_true, minlength=n_cls)
        focus_class_ids = np.argsort(-counts)[:max_classes].tolist()
    else:
        focus_class_ids = list(focus_class_ids)[:max_classes]

    centroids = []
    names = []
    for c in focus_class_ids:
        m = (y_true == c)
        if not np.any(m):
            continue
        mu = feats_np[m].mean(axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-12)
        centroids.append(mu)
        names.append(class_names[c])
    if len(centroids) < 2:
        return
    C = np.stack(centroids, axis=0)
    sim = C @ C.T

    plt.figure(figsize=(max(8, len(names)*0.4), max(6, len(names)*0.4)))
    plt.imshow(sim, vmin=-1, vmax=1, interpolation='nearest')
    plt.colorbar(fraction=0.046, pad=0.04)
    step = max(1, len(names)//20)
    idxs = list(range(0, len(names), step))
    plt.xticks(idxs, [names[i] for i in idxs], rotation=90)
    plt.yticks(idxs, [names[i] for i in idxs])
    plt.title("Centroid Cosine Similarity (selected classes)")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def save_per_class_error_grids(dataset, img_indices, y_true, y_pred, prob_pred,
                               class_names, out_dir, target_class_ids, k_per_class=8,
                               img_size=224, grid_cols=4):
    """
    为给定若干类，各导出一张“最难样本”的拼图（真类=该类，预测错，且错判置信度高）。
    """
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    for c in target_class_ids:
        m = (y_true == c) & (y_pred != c)
        idxs = np.where(m)[0]
        if idxs.size == 0:
            continue
        conf_wrong = prob_pred[idxs, y_pred[idxs]]
        order = np.argsort(-conf_wrong)[:k_per_class]
        sel = idxs[order]
        imgs = []
        for i in sel:
            ds_idx = img_indices[i]
            if ds_idx >= len(dataset.samples):
                continue
            p = dataset.samples[ds_idx][0]
            try:
                img = Image.open(p).convert("RGB").resize((img_size, img_size))
                imgs.append(transforms.ToTensor()(img))
            except Exception:
                continue
        if not imgs:
            continue
        grid = make_grid(torch.stack(imgs, dim=0), nrow=grid_cols, padding=2)
        save_image(grid, os.path.join(out_dir, f"class_{c:03d}_{class_names[c]}_hard.png"))


@torch.no_grad()
def evaluate_one(model: nn.Module,
                 loader: DataLoader,
                 device: torch.device,
                 class_names: List[str],
                 hflip_tta: bool = False,
                 use_amp: bool = True,
                 collect_errors: bool = True,
                 collect_feats: bool = False):
    model.eval()
    y_true, y_pred = [], []
    prob_pred_list = []
    img_indices = []
    feats_list = [] if collect_feats else None
    seen = 0

    if use_amp and device.type == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda")
    else:
        autocast_ctx = nullcontext()

    with autocast_ctx:
        for images, targets in tqdm(loader, desc="Evaluating", ncols=100):
            bs = images.size(0)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits, feats = model_predict_logits(model, images, hflip_tta=hflip_tta, want_feats=collect_feats)
            probs = F.softmax(logits, dim=1)

            pred = probs.argmax(dim=1)

            y_true.extend(targets.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            prob_pred_list.append(probs.cpu().numpy())

            if collect_feats and feats is not None:
                feats_list.append(feats.detach().float().cpu().numpy())

            img_indices.extend(range(seen, seen + bs))
            seen += bs

    prob_pred = np.concatenate(prob_pred_list, axis=0)
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    feats_np = np.concatenate(feats_list, axis=0) if (collect_feats and feats_list) else None

    top1 = accuracy_score(y_true_np, y_pred_np)
    try:
        top5 = top_k_accuracy_score(y_true_np, prob_pred, k=5, labels=list(range(len(class_names))))
    except Exception:
        top5 = float('nan')

    report = classification_report(y_true_np, y_pred_np, target_names=class_names,
                                   output_dict=True, zero_division=0)
    macro_f1 = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]
    macro_prec = report["macro avg"]["precision"]
    macro_rec = report["macro avg"]["recall"]

    try:
        roc_auc = roc_auc_score(y_true_np, prob_pred, multi_class="ovr")
    except Exception:
        roc_auc = float('nan')

    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(len(class_names))))

    errors = None
    if collect_errors:
        wrong = np.where(y_true_np != y_pred_np)[0].tolist()
        errors = {"num_errors": len(wrong), "error_indices": wrong}

    metrics = {
        "top1": top1, "top5": top5,
        "macro_precision": macro_prec, "macro_recall": macro_rec,
        "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "roc_auc_ovr": roc_auc
    }
    return metrics, cm, report, prob_pred, y_true_np, y_pred_np, img_indices, errors, feats_np


def main():
    parser = argparse.ArgumentParser(description="批量评估 RegNetY-32GF 权重（CNFOOD-241）")
    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--ckpts", type=str, nargs="+", default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--hflip-tta", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-error-grid", type=int, default=None)
    parser.add_argument("--f1-topk", type=int, default=None)
    args = parser.parse_args()

    # 应用默认值（命令行优先）
    val_dir = args.val_dir or DEFAULTS["VAL_DIR"]
    ckpts_in = args.ckpts or DEFAULTS["CKPTS"]
    out_dir = args.out_dir or DEFAULTS["OUT_DIR"]
    model_name = args.model_name or DEFAULTS["MODEL_NAME"]
    num_classes = args.num_classes or DEFAULTS["NUM_CLASSES"]
    img_size = args.img_size or DEFAULTS["IMG_SIZE"]
    batch_size = args.batch_size or DEFAULTS["BATCH_SIZE"]
    num_workers = args.num_workers or DEFAULTS["NUM_WORKERS"]
    hflip_tta = args.hflip_tta or DEFAULTS["HFLIP_TTA"]
    seed = args.seed or DEFAULTS["SEED"]
    max_error_grid = args.max_error_grid or DEFAULTS["MAX_ERROR_GRID"]
    f1_topk = args.f1_topk or DEFAULTS["F1_TOPK"]

    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # 展开通配
    ckpt_paths = []
    for item in ckpts_in:
        expanded = glob(item)
        if expanded:
            ckpt_paths.extend(expanded)
        else:
            ckpt_paths.append(item)
    uniq = []
    for p in ckpt_paths:
        if p not in uniq and os.path.isfile(p):
            uniq.append(p)
    ckpt_paths = uniq
    if not ckpt_paths:
        raise FileNotFoundError("未找到任何权重文件，请检查 DEFAULTS['CKPTS'] 或命令行 --ckpts。")

    # 数据与 loader
    tfm = build_transforms(img_size, center_crop=True)
    dataset = datasets.ImageFolder(root=val_dir, transform=tfm)
    class_names = dataset.classes
    if num_classes != len(class_names):
        warnings.warn(f"传入 num_classes={num_classes} 与数据集类别数 {len(class_names)} 不一致，改用数据集实际类别数。")
    num_classes = len(class_names)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 保存类别映射
    with open(os.path.join(out_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump({i: c for i, c in enumerate(class_names)}, f, ensure_ascii=False, indent=2)

    summary_rows = []

    # 读取可选大类映射
    super_map = load_superclass_mapping(DEFAULTS.get("SUPERCLASS_CSV"), class_names)

    for ckpt in ckpt_paths:
        tag = Path(ckpt).stem
        print(f"\n==== 评估权重：{ckpt} ====")
        model = build_model(model_name, num_classes=num_classes)
        missing, unexpected = load_state_dict_safely(model, ckpt)
        if missing or unexpected:
            print(f"[state_dict] missing={len(missing)} unexpected={len(unexpected)}")

        model.to(device)

        (metrics, cm, report, prob_pred,
         y_true, y_pred, img_indices, errors, feats_np) = evaluate_one(
            model, loader, device, class_names,
            hflip_tta=hflip_tta, use_amp=True, collect_errors=True,
            collect_feats=DEFAULTS["MAKE_TSNE"] or True  # 采集一次特征，供类中心/难例图等
        )

        out_sub = os.path.join(out_dir, tag)
        os.makedirs(out_sub, exist_ok=True)

        # 指标 & 分类报告
        with open(os.path.join(out_sub, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        pd.DataFrame(report).transpose().to_csv(os.path.join(out_sub, "classification_report.csv"), encoding="utf-8-sig")

        # 混淆矩阵（全量）
        plot_confusion_matrix(cm, class_names, normalize=False, out_png=os.path.join(out_sub, "confusion_matrix_counts.png"))
        plot_confusion_matrix(cm, class_names, normalize=True, out_png=os.path.join(out_sub, "confusion_matrix_normalized.png"))

        # F1（最好 & 最差）
        class_f1 = {k: float(v["f1-score"]) for k, v in report.items() if k not in ["accuracy","macro avg","weighted avg"]}
        plot_class_f1(class_f1, out_png=os.path.join(out_sub, f"class_f1_top{f1_topk}.png"), top_k=f1_topk)
        if DEFAULTS["MAKE_WORST_F1"]:
            plot_class_f1_worst(class_f1, out_png=os.path.join(out_sub, "class_f1_worst30.png"), bottom_k=30)

        # 逐样本预测
        if DEFAULTS["MAKE_PRED_CSV"]:
            save_predictions_csv(dataset, img_indices, y_true, y_pred, prob_pred, class_names, os.path.join(out_sub, "predictions.csv"))

        # 最混淆类对 + 最差类子矩阵
        if DEFAULTS["MAKE_TOP_CONFUSIONS"]:
            export_top_confusions(cm, class_names, os.path.join(out_sub, "top_confusions.csv"), k=50)
            plot_confusion_submatrix_worst(cm, class_names, report, out_png=os.path.join(out_sub, "confusion_submatrix_worst20.png"),
                                           worst_rows=20, top_pred_cols=20)

        # 错误样本拼图（高置信错判）
        save_top_errors_grid(dataset, img_indices, y_true.tolist(), y_pred.tolist(), prob_pred,
                             class_names, out_png=os.path.join(out_sub, "top_errors_grid.png"),
                             grid_cols=6, max_show=DEFAULTS["MAX_ERROR_GRID"], img_size=256)

        # 置信度/边际与校准
        if DEFAULTS["MAKE_CONF_HIST"]:
            plot_confidence_hist(prob_pred, y_true, out_png=os.path.join(out_sub, "confidence_hist.png"))
        if DEFAULTS["MAKE_MARGIN_HIST"]:
            margins = plot_margin_hist(prob_pred, out_png=os.path.join(out_sub, "margin_hist.png"))
            if DEFAULTS["MAKE_LOW_MARGIN_GRID"]:
                save_low_margin_grid(dataset, img_indices, margins, max_show=DEFAULTS["MAX_ERROR_GRID"],
                                     out_png=os.path.join(out_sub, "low_margin_grid.png"), img_size=256, grid_cols=6)
        if DEFAULTS["MAKE_CALIB"]:
            ece_val = compute_ece(prob_pred, y_true, n_bins=15)
            with open(os.path.join(out_sub, "calibration.json"), "w", encoding="utf-8") as f:
                json.dump({"ece": ece_val}, f, ensure_ascii=False, indent=2)
            plot_reliability(prob_pred, y_true, out_png=os.path.join(out_sub, "reliability_diagram.png"), n_bins=15)
            plot_roc_pr_micro(prob_pred, y_true, out_roc_png=os.path.join(out_sub, "roc_micro.png"),
                              out_pr_png=os.path.join(out_sub, "pr_micro.png"))

        # —— 新增：部署阈值曲线 / 长尾 / Top-5 召回 / 每类 ECE ——
        plot_coverage_accuracy_curve(
            prob_pred, y_true,
            out_png=os.path.join(out_sub, "coverage_accuracy.png"),
            out_csv=os.path.join(out_sub, "coverage_accuracy.csv"),
            steps=200
        )
        plot_support_vs_f1(report, out_png=os.path.join(out_sub, "support_vs_f1.png"))
        plot_topk_recall_worst(
            prob_pred, y_true, class_names,
            out_png=os.path.join(out_sub, "top5_recall_worst30.png"),
            top_k=5, bottom_k=30
        )
        plot_class_ece_worst(
            prob_pred, y_true, class_names,
            out_png=os.path.join(out_sub, "class_ece_worst30.png"),
            out_csv=os.path.join(out_sub, "class_ece_ovr.csv"),
            n_bins=15, bottom_k=30
        )

        # t-SNE（选最差K类，限每类样本数）
        if DEFAULTS["MAKE_TSNE"] and feats_np is not None:
            f1_per_class = {k: float(v["f1-score"]) for k, v in report.items() if k not in ["accuracy","macro avg","weighted avg"]}
            name_to_id = {name: i for i, name in enumerate(class_names)}
            worst_names = [k for k,_ in sorted(f1_per_class.items(), key=lambda x:x[1])[:DEFAULTS["TSNE_WORST_K"]]]
            worst_ids = set([name_to_id[n] for n in worst_names])
            y_true_np = np.asarray(y_true)
            keep_idx = []
            per_class_count = {}
            for i, c in enumerate(y_true_np):
                if c in worst_ids:
                    per_class_count.setdefault(int(c), 0)
                    if per_class_count[int(c)] < DEFAULTS["TSNE_MAX_PER_CLASS"]:
                        keep_idx.append(i)
                        per_class_count[int(c)] += 1
            if keep_idx:
                X = feats_np[keep_idx]
                y_sel = y_true_np[keep_idx]
                tsne = TSNE(n_components=2, perplexity=DEFAULTS["TSNE_PERPLEXITY"], init="pca", learning_rate="auto", random_state=42)
                emb = tsne.fit_transform(X)
                plt.figure(figsize=(8,6))
                for cid in sorted(list(worst_ids)):
                    mask = (y_sel == cid)
                    if np.any(mask):
                        plt.scatter(emb[mask,0], emb[mask,1], s=8, label=class_names[cid])
                plt.legend(markerscale=2, fontsize=8, ncol=2)
                plt.title(f"t-SNE on Worst-{len(worst_ids)} Classes (≤{DEFAULTS['TSNE_MAX_PER_CLASS']}/class)")
                plt.tight_layout(); plt.savefig(os.path.join(out_sub, "tsne_worstK.png"), dpi=200); plt.close()

        # —— 类中心相似度热图 + 逐类难例拼图（聚焦最差类，需要 feats_np） ——
        if feats_np is not None:
            f1_per_class = {k: float(v["f1-score"]) for k, v in report.items() if k not in ["accuracy","macro avg","weighted avg"]}
            name_to_id = {name: i for i, name in enumerate(class_names)}
            worst_names = [k for k,_ in sorted(f1_per_class.items(), key=lambda x:x[1])[:40]]
            worst_ids = [name_to_id[n] for n in worst_names]
            plot_centroid_cosine_heatmap(
                feats_np, y_true, class_names,
                out_png=os.path.join(out_sub, "centroid_cosine_worst40.png"),
                focus_class_ids=worst_ids, max_classes=40
            )
            save_per_class_error_grids(
                dataset, img_indices, y_true, y_pred, prob_pred, class_names,
                out_dir=os.path.join(out_sub, "per_class_hard"),
                target_class_ids=worst_ids, k_per_class=8, img_size=256, grid_cols=4
            )

        # —— 可选：大类聚合 ——
        super_map = load_superclass_mapping(DEFAULTS.get("SUPERCLASS_CSV"), class_names)
        if super_map is not None:
            super_names, class_to_super_id = super_map
            cm_s, y_true_s, y_pred_s = aggregate_super_confusion(y_true, y_pred, class_to_super_id, super_names)
            plot_confusion_matrix(cm_s, super_names, normalize=True, out_png=os.path.join(out_sub, "super_confusion_normalized.png"),
                                  figsize=(max(8, len(super_names)*0.6), max(6, len(super_names)*0.6)))
            f1_super = f1_score(y_true_s, y_pred_s, average=None, labels=list(range(len(super_names))))
            plt.figure(figsize=(max(8, len(super_names)*0.5), 4))
            plt.bar(range(len(super_names)), f1_super)
            plt.xticks(range(len(super_names)), super_names, rotation=45, ha='right')
            plt.ylabel("F1"); plt.title("Super-class F1")
            plt.tight_layout(); plt.savefig(os.path.join(out_sub, "super_f1.png"), dpi=200); plt.close()

        # 汇总行
        summary_rows.append({
            "ckpt": ckpt,
            "top1": metrics["top1"],
            "top5": metrics["top5"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "roc_auc_ovr": metrics["roc_auc_ovr"],
            "num_errors": (errors or {}).get("num_errors", None)
        })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.sort_values(by=["top1", "macro_f1"], ascending=[False, False], inplace=True)
    df_sum.to_csv(os.path.join(out_dir, "summary.csv"), index=False, encoding="utf-8-sig")

    print("\n评估完成。结果目录：", os.path.abspath(out_dir))
    print(df_sum.head(min(10, len(df_sum))))


if __name__ == "__main__":
    main()
