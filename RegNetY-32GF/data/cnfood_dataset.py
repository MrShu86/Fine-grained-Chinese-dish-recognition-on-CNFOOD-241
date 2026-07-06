# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from .transforms import get_train_transforms, get_val_transforms

def build_loaders(cfg):
    tr_tf = get_train_transforms(cfg["img_size"], cfg["random_erasing"])
    va_tf = get_val_transforms(cfg["img_size"])

    ds_tr = ImageFolder(cfg["train_dir"], transform=tr_tf)
    ds_va = ImageFolder(cfg["val_dir"],   transform=va_tf)

    num_workers = int(cfg.get("num_workers", 8))
    common_kwargs = dict(
        num_workers=num_workers,
        pin_memory=bool(cfg.get("pin_memory", True)),
        persistent_workers=True if num_workers > 0 else False,
    )
    if num_workers > 0 and "prefetch_factor" in cfg:
        common_kwargs["prefetch_factor"] = int(cfg["prefetch_factor"])
    dl_tr = DataLoader(
        ds_tr, batch_size=cfg["batch_size"], shuffle=True,
        **common_kwargs
    )
    dl_va = DataLoader(
        ds_va, batch_size=cfg["batch_size"], shuffle=False,
        **common_kwargs
    )
    dl_tr._codex_cfg = cfg
    dl_va._codex_cfg = cfg
    return ds_tr, ds_va, dl_tr, dl_va
