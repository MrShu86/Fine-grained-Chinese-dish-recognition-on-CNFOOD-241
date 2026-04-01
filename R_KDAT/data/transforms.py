# -*- coding: utf-8 -*-
from torchvision import transforms as T

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int,
                         use_random_erasing: bool = False) -> T.Compose:
    """
    训练增强：温和但有效，适合细粒度食物分类。
    """
    train_tf = [
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
    if use_random_erasing:
        train_tf.append(
            T.RandomErasing(p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.3), inplace=False)
        )
    return T.Compose(train_tf)


def get_val_transforms(img_size: int) -> T.Compose:
    """验证增强：标准的 Resize+CenterCrop。"""
    resize_size = int(img_size / 0.875)
    return T.Compose([
        T.Resize(resize_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
