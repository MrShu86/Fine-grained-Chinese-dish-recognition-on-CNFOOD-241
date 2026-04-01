# -*- coding: utf-8 -*-

CONFIG = {
    # ====== 数据路径（按你的机器修改）======
    "train_dir": "/food_data/food_data/CNFOOD-241/CNFOOD-241/train600x600",
    "val_dir":   "/food_data/food_data/CNFOOD-241/CNFOOD-241/val600x600",
    "class_file": "/food_data/food_data/CNFOOD-241/CNFOOD-241/class_name.xls",

    # ====== 训练超参 ======
    "num_classes": 241,
    "img_size": 300,                # 224 或 300；RegNetY-32GF 建议 300 起步
    "batch_size": 128,              # 24G GPU: 64~96；12G GPU: 32~48
    "epochs": 40,
    "lr": 5e-4,
    "weight_decay": 1e-4,

    # ====== 正则 & 增强 ======
    "label_smoothing": 0.1,
    "use_focal_loss": True,         # 长尾/难例时可 True
    "focal_gamma": 2.0,
    "mixup_alpha": 0.2,             # 先关；如需开启设为 0.2
    "cutmix_alpha": 1.0,            # 先关；如需开启设为 1.0
    "random_erasing": True,

    # ====== 采样器（长尾更稳）======
    "sampler": "uniform",           # "uniform" 或 "weighted"

    # ====== 学习率调度 ======
    "warmup_epochs": 5,
    "cosine_min_lr_ratio": 0.01,    # 余弦最低 lr = base_lr * ratio

    # ====== 评估/日志/存档 ======
    "eval_every": 1,                # 每 N 个 epoch 验证
    "val_max_batches": 0,           # 0 表示全量验证
    "out_dir": "./exp_regnety32",
    "use_tensorboard": True,
    "resume_ckpt": "/food_data/R_KDAT/exp_regnety32/best.pt",              # 需要续训时填入 best.pt 路径

    # ====== 运行环境 ======
    "seed": 42,
    "num_workers": 8,
    "prefetch_factor": 2,
    "pin_memory": True,
    "dataparallel": True,           # 多卡可设 True（nn.DataParallel）
    #ArcFace（m=0.35）+ Triplet(0.2)
    # 采样（建议配合 Triplet）
    "sampler": {"type": "balanced_pk", "P": 32, "K": 4},  # 若 batch_size=128
    
    # 分类头：ArcFace
    "head": {"arcface": True, "s": 30.0, "m": 0.35},      # 先用 m=0.35
    
    # Triplet（batch-hard）
    "metric": {"use_triplet": True, "margin": 0.3, "weight": 0.2, "normalize": True},
    
    # （可选，已在你这有）收尾把 MixUp/CutMix 衰减到 0
    "mix_sched": {"final_zero_epochs": 3},

    # ====== 知识蒸馏（与 main/train_loop 匹配）======
    "kd": {
        "use": True,                                # 开关：True 启动 KD
        "alpha": 0.6,                               # 总损失 = (1-α)*CE + α*KL
        "T": 3.0,                                   # 温度
        "teacher_ckpt": "/food_data/RegNetY-32GF/exp_regnety32/best.pt",  # ← 改为你的教师权重
        "freeze_bn": True,                          # 冻结 teacher BN 统计
        "dp_teacher": False                         # 多卡下是否对 teacher 用 DataParallel
    },
}
