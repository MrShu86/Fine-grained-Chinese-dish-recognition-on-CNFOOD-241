# data/cnfood_dataset.py
# -*- coding: utf-8 -*-
"""
CNFOOD-241 DataLoaders (ImageFolder) + Sampler
支持三种训练采样策略：
  1) uniform / None：常规 shuffle=True
  2) weighted：WeightedRandomSampler（按类别频次倒数加权）
  3) balanced_pk：Balanced P-K Sampler（每个 batch 采 P 类、每类 K 张，batch=P*K），适配 batch-hard triplet

返回：
  ds_tr, ds_va, dl_tr, dl_va
"""

import os
import math
import random
from collections import defaultdict
from typing import Optional, Union, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Sampler
from torchvision.datasets import ImageFolder

from .transforms import get_train_transforms, get_val_transforms


def _seed_worker(worker_id: int):
    """让每个 DataLoader worker 的随机数可复现。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _get_labels_from_imagefolder(dataset: ImageFolder) -> np.ndarray:
    labels = getattr(dataset, "targets", None)
    if labels is None:
        labels = [c for _, c in dataset.samples]
    return np.asarray(labels, dtype=np.int64)


class BalancedPKSampler(Sampler[int]):
    """
    Balanced P-K Sampler:
      - 每个 batch 抽 P 个类别，每类抽 K 个样本（不足则放回采样）
      - batch_size = P * K
    说明：
      - 为了“每个 epoch 都有不同采样”但仍可复现：每次 __iter__ 会使用 seed + epoch 作为随机种子，
        并在 __iter__ 末尾 epoch 自增。
      - 适用于 DataParallel/单机训练；如果你使用 DistributedDataParallel，请用更严格的分布式采样器。
    """

    def __init__(self, labels: Union[np.ndarray, list], P: int, K: int, seed: int = 42):
        super().__init__()
        self.P = int(P)
        self.K = int(K)
        assert self.P > 0 and self.K > 0, "P 和 K 必须为正整数"
        self.seed = int(seed)

        self.labels = np.asarray(labels, dtype=np.int64)
        self.num_samples = int(len(self.labels))
        self.batch_size = self.P * self.K

        self.cls_to_indices = defaultdict(list)
        for i, y in enumerate(self.labels):
            self.cls_to_indices[int(y)].append(i)
        self.classes = sorted(list(self.cls_to_indices.keys()))
        if len(self.classes) == 0:
            raise ValueError("BalancedPKSampler: labels 为空，无法采样。")

        # 估计每个 epoch 的 batch 数：尽量覆盖全数据（近似）
        self.num_batches = int(math.ceil(self.num_samples / float(self.batch_size)))

        # epoch 计数（用于每轮采样不同，但仍可复现）
        self._epoch = 0

    def __len__(self) -> int:
        # 让长度是 batch_size 的整数倍，避免 DataLoader 拼 batch 出现尾巴
        return self.num_batches * self.batch_size

    def set_epoch(self, epoch: int):
        """可选：外部手动设置 epoch（如果你想更可控）。"""
        self._epoch = int(epoch)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self._epoch)

        for _ in range(self.num_batches):
            # 选 P 个类（类数不足则放回）
            replace_cls = len(self.classes) < self.P
            chosen_classes = rng.choice(self.classes, size=self.P, replace=replace_cls)

            batch = []
            for c in chosen_classes:
                idxs = self.cls_to_indices[int(c)]
                replace = len(idxs) < self.K
                picked = rng.choice(idxs, size=self.K, replace=replace)
                batch.extend(picked.tolist())

            rng.shuffle(batch)
            for i in batch:
                yield int(i)

        self._epoch += 1


def _make_weighted_sampler(labels: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    class_counts = np.bincount(labels, minlength=int(num_classes))
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )


def _make_sampler_if_needed(dataset: ImageFolder, cfg: dict):
    """
    sampler 允许三种形态：
      - "uniform"/None：不启用 sampler（shuffle=True）
      - "weighted"：WeightedRandomSampler
      - {"type":"balanced_pk","P":32,"K":4}：BalancedPKSampler
      - {"type":"weighted"}：WeightedRandomSampler（等价于字符串）
    """
    sampler_cfg = cfg.get("sampler", "uniform")

    labels = _get_labels_from_imagefolder(dataset)
    num_classes = int(cfg.get("num_classes", int(labels.max() + 1)))

    # dict 形式
    if isinstance(sampler_cfg, dict):
        stype = sampler_cfg.get("type", "uniform")
        if stype == "balanced_pk":
            P = int(sampler_cfg.get("P", 32))
            K = int(sampler_cfg.get("K", 4))
            return BalancedPKSampler(labels, P=P, K=K, seed=int(cfg.get("seed", 42)))
        if stype == "weighted":
            return _make_weighted_sampler(labels, num_classes=num_classes)
        # 其它类型视为不启用
        return None

    # 字符串形式
    if isinstance(sampler_cfg, str):
        stype = sampler_cfg.lower().strip()
        if stype == "weighted":
            return _make_weighted_sampler(labels, num_classes=num_classes)
        # "uniform" / 其它
        return None

    # 其它类型：不启用
    return None


def build_loaders(cfg: Dict[str, Any]):
    """
    返回：
      ds_tr, ds_va, dl_tr, dl_va

    说明：
      - 若启用 sampler（weighted / balanced_pk），训练集 DataLoader 使用 sampler 且 shuffle=False
      - 否则训练集使用 shuffle=True
      - prefetch_factor 仅当 num_workers>0 时生效
    """
    img_size = int(cfg["img_size"])
    tr_tf = get_train_transforms(img_size, cfg.get("random_erasing", False))
    va_tf = get_val_transforms(img_size)

    assert os.path.isdir(cfg["train_dir"]), f"train_dir 不存在：{cfg['train_dir']}"
    assert os.path.isdir(cfg["val_dir"]), f"val_dir 不存在：{cfg['val_dir']}"

    ds_tr = ImageFolder(cfg["train_dir"], transform=tr_tf)
    ds_va = ImageFolder(cfg["val_dir"], transform=va_tf)

    n_tr_cls = len(ds_tr.classes)
    n_va_cls = len(ds_va.classes)
    if int(cfg["num_classes"]) != n_tr_cls:
        print(f"[WARN] 训练集类别数({n_tr_cls}) ≠ CONFIG.num_classes({cfg['num_classes']})")
    if int(cfg["num_classes"]) != n_va_cls:
        print(f"[WARN] 验证集类别数({n_va_cls}) ≠ CONFIG.num_classes({cfg['num_classes']})")

    # sampler
    tr_sampler = _make_sampler_if_needed(ds_tr, cfg)

    # 如果是 balanced_pk，建议 batch_size == P*K
    sampler_cfg = cfg.get("sampler", None)
    if isinstance(tr_sampler, BalancedPKSampler):
        bs = int(cfg["batch_size"])
        if bs != tr_sampler.batch_size:
            print(f"[WARN] balanced_pk 要求 batch_size=P*K={tr_sampler.batch_size}，"
                  f"但当前 batch_size={bs}。建议把 batch_size 改为 {tr_sampler.batch_size}。")

    num_workers = int(cfg.get("num_workers", 8))
    common_kwargs = dict(
        num_workers=num_workers,
        pin_memory=bool(cfg.get("pin_memory", True)),
        persistent_workers=True if num_workers > 0 else False,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(int(cfg.get("seed", 42))),
    )
    if num_workers > 0 and "prefetch_factor" in cfg:
        common_kwargs["prefetch_factor"] = int(cfg["prefetch_factor"])

    dl_tr = DataLoader(
        ds_tr,
        batch_size=int(cfg["batch_size"]),
        shuffle=(tr_sampler is None),
        sampler=tr_sampler,
        drop_last=True,
        **common_kwargs,
    )

    dl_va = DataLoader(
        ds_va,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )

    return ds_tr, ds_va, dl_tr, dl_va