# -*- coding: utf-8 -*-

CONFIG = {
    # ====== 数据路径（按你的机器修改）======
    "train_dir": "/food_data/food_data/CNFOOD-241/CNFOOD-241/train600x600",
    "val_dir":   "/food_data/food_data/CNFOOD-241/CNFOOD-241/val600x600",
    "class_file": "/food_data/food_data/CNFOOD-241/CNFOOD-241/class_name.xls",

    # ====== 训练超参 ======
    "num_classes": 241,
    "img_size": 300,                 # 224 或 300；RegNetY-32GF 建议 300 起步
    "batch_size": 128,                # 24G GPU: 64~96；12G GPU: 32~48
    "epochs": 30,
    "lr": 5e-4,
    "weight_decay": 1e-4,

    # ====== 正则 & 增强 ======
    "label_smoothing": 0.0,
    "use_focal_loss": False,        # Pure baseline: standard cross-entropy
    "focal_gamma": 2.0,
    "mixup_alpha": 0.0,              # Pure baseline: disabled
    "cutmix_alpha": 0.0,             # Pure baseline: disabled
    "random_erasing": False,

    # ====== 学习率调度 ======
    "warmup_epochs": 5,
    "cosine_min_lr_ratio": 0.01,     # 余弦最低 lr = lr * ratio

    # ====== 评估/日志/存档 ======
    "eval_every": 1,                 # 每 N 个 epoch 验证
    "val_max_batches": 0,            # 0 表示全量验证
    "out_dir": "./exp_regnety32",
    "resume_ckpt": "",               # 断点恢复路径；留空不恢复
    "use_tensorboard": True,

    # ====== 运行环境 ======
    "seed": 42,
    "num_workers": 8,
    "prefetch_factor": 2,
    "pin_memory": True,
    "amp_dtype": "bf16",             # A100 recommended: "bf16"; alternatives: "fp16", "off"
    "channels_last": True,
    "fused_adamw": True,
    "compile": False,
    "dataparallel": True ,           # 多卡可设 True（nn.DataParallel）
    #
    "use_tensorboard": True,
    "resume_ckpt": "",   # 需要续训时填入 best.pt 路径

}

