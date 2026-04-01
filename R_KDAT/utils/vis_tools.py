# utils/vis_tools.py
# -*- coding: utf-8 -*-
import os, math, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 后端无界面环境也能保存图
import matplotlib.pyplot as plt
from PIL import Image

def plot_confusion_matrix(cm: np.ndarray, out_png: str, class_names=None, normalize=True, figsize=(10,8)):
    """
    cm: (C, C) numpy array, 行是真实类，列是预测类
    normalize: True 时每行归一化到 0~1
    """
    cm = cm.astype(np.float64)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm = cm / row_sum

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest', aspect='auto')
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix (row-normalized)" if normalize else "Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if class_names is not None and len(class_names) == cm.shape[0] and cm.shape[0] <= 50:
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90)
        ax.set_yticklabels(class_names)

    # 不逐格写数值，避免 241 类时拥挤
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

def save_per_class_acc_csv(per_class_acc: np.ndarray, out_csv: str, class_names=None):
    """
    per_class_acc: (C,) 每类准确率（0~1）
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if class_names is not None:
            w.writerow(["class_id", "class_name", "acc"])
            for i, a in enumerate(per_class_acc):
                w.writerow([i, class_names[i], float(a)])
        else:
            w.writerow(["class_id", "acc"])
            for i, a in enumerate(per_class_acc):
                w.writerow([i, float(a)])

def save_topk_error_grid(paths, labels, preds, out_png: str, k=64, rows=8, cols=8, thumb=128):
    """
    从错误样本中挑前 k 个，按 rows x cols 拼图保存。
    paths: 图像路径列表（若未知可传空字符串；会跳过打不开的）
    labels/preds: np.ndarray of int
    """
    wrong_idx = np.where(labels != preds)[0]
    if wrong_idx.size == 0:
        # 没有错误样本也保存个空白图，便于汇报
        fig = plt.figure(figsize=(cols, rows))
        plt.text(0.5, 0.5, "No errors", ha="center", va="center", fontsize=18)
        plt.axis('off')
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    sel = wrong_idx[:k]
    imgs = []
    for i in sel:
        p = paths[i] if i < len(paths) else ""
        try:
            im = Image.open(p).convert("RGB")
        except:
            im = Image.new("RGB", (thumb, thumb), color=(200,200,200))
        im = im.resize((thumb, thumb))
        # 画上真/预测标签 id
        canvas = Image.new("RGB", (thumb, thumb+24), color=(255,255,255))
        canvas.paste(im, (0, 0))
        from PIL import ImageDraw, ImageFont
        dr = ImageDraw.Draw(canvas)
        txt = f"T:{int(labels[i])} P:{int(preds[i])}"
        dr.text((5, thumb+4), txt, fill=(0,0,0))
        imgs.append(canvas)

    # 拼图
    grid_w = cols * thumb
    grid_h = rows * (thumb+24)
    grid = Image.new("RGB", (grid_w, grid_h), color=(255,255,255))
    for idx, im in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        if r >= rows: break
        grid.paste(im, (c*thumb, r*(thumb+24)))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    grid.save(out_png)
