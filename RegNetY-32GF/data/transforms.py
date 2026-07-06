# -*- coding: utf-8 -*-
from torchvision import transforms

def get_train_transforms(img_size=300, random_erasing=False):
    aug = [
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ]
    if random_erasing:
        aug.append(transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.3)))
    return transforms.Compose(aug)

def get_val_transforms(img_size=300):
    return transforms.Compose([
        transforms.Resize(int(img_size*1.12)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
