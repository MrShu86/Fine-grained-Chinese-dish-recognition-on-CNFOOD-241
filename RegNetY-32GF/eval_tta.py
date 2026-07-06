# eval_tta.py  (flex ckpt loader + rebuild val DataLoader + TTA)
# -*- coding: utf-8 -*-
import os, argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import CONFIG as BASE_CONFIG
from models.regnety32 import build_model
from data.cnfood_dataset import build_loaders

# ============== metrics ==============
@torch.no_grad()
def _topk(output: torch.Tensor, target: torch.Tensor, topk=(1,5)) -> Tuple[float, float]:
    maxk = max(topk); b = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)  # (B, maxk)
    pred = pred.t()                             # (maxk, B)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k * (100.0 / b)).item())
    if len(res) == 1: return res[0], res[0]
    return res[0], res[1]

# ============== TTA ==============
def _build_tta_ops(sizes: List[int], img_size: int, hflip: bool):
    ops = []
    for s in sizes:
        base = transforms.Compose([
            transforms.Resize(int(s*1.10)),
            transforms.CenterCrop(s),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])
        ops.append((f"s{s}", base))
        if hflip:
            ops.append((f"s{s}-hflip", transforms.Compose([base, transforms.RandomHorizontalFlip(p=1.0)])))
    return ops

@torch.no_grad()
def tta_forward(model, images: torch.Tensor, tta_ops, device):
    b = images.size(0)
    logits_accum = None
    for _, t in tta_ops:
        imgs = []
        for i in range(b):
            img = transforms.ToPILImage()(images[i].cpu())
            imgs.append(t(img))
        x = torch.stack(imgs, 0).to(device)
        out = model(x)
        logits_accum = out if logits_accum is None else (logits_accum + out)
    return logits_accum / float(len(tta_ops))

# ============== Flexible checkpoint loader ==============
def load_ckpt_flex(model: torch.nn.Module, ckpt_path: str, ignore_mismatch_head: bool = True):
    """
    - 识别 {"model": state_dict} 或直接 state_dict
    - 去掉 DataParallel 的 "module." 前缀
    - 若分类头形状不一致且 ignore_mismatch_head=True，则自动跳过该层（只加载形状匹配的键）
    """
    print(f"[CKPT] loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)

    # strip "module." if present
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    msd = model.state_dict()
    loadable = {}
    skipped = []
    missing = []
    for k, v in state.items():
        if k in msd:
            if msd[k].shape == v.shape:
                loadable[k] = v
            else:
                # 仅在分类头等维度不一致时跳过
                if ignore_mismatch_head and ("fc." in k or "classifier." in k):
                    skipped.append((k, tuple(v.shape), tuple(msd[k].shape)))
                else:
                    skipped.append((k, tuple(v.shape), tuple(msd[k].shape)))
        # 其他不存在的键忽略

    # 把可加载的子集灌进模型
    msg = model.load_state_dict(loadable, strict=False)
    # 收集真正缺失的键（strict=False 时不会抛异常）
    missing.extend(list(msg.missing_keys))

    print(f"[CKPT] loaded keys: {len(loadable)} / {len(msd)}")
    if skipped:
        print("[CKPT] skipped (shape mismatch):")
        for k, s_ckpt, s_model in skipped[:10]:
            print(f"  - {k}: ckpt{ s_ckpt } vs model{ s_model }")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped)-10} more")
    if missing:
        print(f"[CKPT] missing in ckpt but in model: {len(missing)} (ok if heads differ)")
    return ckpt

# ============== main ==============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="./exp_regnety32/best.pt", help="模型权重路径")
    ap.add_argument("--out", type=str, default="./exp_regnety32/eval_tta", help="输出目录")
    ap.add_argument("--sizes", type=str, default="256,300,336", help="TTA 多尺度，逗号分隔")
    ap.add_argument("--no_hflip", action="store_true", help="关闭水平翻转 TTA")
    ap.add_argument("--bs", type=int, default=64, help="评测 batch_size")
    ap.add_argument("--limit", type=int, default=0, help="仅评测前 N 批（0=全量）")
    ap.add_argument("--strict_head", action="store_true", help="分类头形状必须匹配（默认宽松跳过）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = dict(BASE_CONFIG)
    img_size = int(cfg.get("img_size", 300))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset / val loader（用 --bs 重建）
    _, ds_va, _, _ = build_loaders(cfg)
    dl_va = DataLoader(
        ds_va,
        batch_size=args.bs,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        prefetch_factor=int(cfg.get("prefetch_factor", 2)),
        persistent_workers=int(cfg.get("num_workers", 4)) > 0,
    )

    # model
    model = build_model(num_classes=int(cfg["num_classes"]), pretrained=False).to(device)
    load_ckpt_flex(model, args.ckpt, ignore_mismatch_head=not args.strict_head)
    model.eval()

    # TTA ops
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    tta_ops = _build_tta_ops(sizes, img_size=img_size, hflip=(not args.no_hflip))
    print(f"[TTA] ops={len(tta_ops)}  sizes={sizes}  hflip={not args.no_hflip}")

    # eval
    loss_sum = n_sum = 0
    top1_sum = top5_sum = 0.0
    pbar = tqdm(dl_va, ncols=120, desc="[TTA Eval]")
    for bi, batch in enumerate(pbar):
        (images, labels, paths) = batch if len(batch) == 3 else (*batch, [""]*len(batch[0]))
        images = images.to(device); labels = labels.to(device)

        with torch.no_grad():
            logits = tta_forward(model, images, tta_ops, device=device)
            loss = F.cross_entropy(logits, labels)
            t1, t5 = _topk(logits, labels, topk=(1,5))

        bs = labels.size(0)
        loss_sum += loss.item() * bs; n_sum += bs
        top1_sum += t1 * bs; top5_sum += t5 * bs
        pbar.set_postfix(loss=f"{loss_sum/max(n_sum,1):.4f}",
                         top1=f"{top1_sum/max(n_sum,1):.2f}%",
                         top5=f"{top5_sum/max(n_sum,1):.2f}%")
        if args.limit and (bi+1) >= args.limit:
            break

    top1 = top1_sum/max(n_sum,1); top5 = top5_sum/max(n_sum,1)
    avg_loss = loss_sum/max(n_sum,1)
    with open(os.path.join(args.out, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"top1={top1:.2f}%\n")
        f.write(f"top5={top5:.2f}%\n")
        f.write(f"loss={avg_loss:.4f}\n")
    print(f"[DONE] TTA top1={top1:.2f}%  top5={top5:.2f}%  loss={avg_loss:.4f}")
    # （若需要可在此处补充可视化，之前我给过 vis_tools.py 的示例接口）

if __name__ == "__main__":
    main()
