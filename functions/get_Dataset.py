import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
from PIL import Image
import re
import random
from natsort import natsorted  # 排序


class Restruction:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Lambda(lambda x: (x - 0.5) * 2),
            ]
        )

    def get_loaders(self):
        print("=> Utilizing the RestructionDataset() for data loading...")
        train_dataset = RestructionDataset(
            dir=os.path.join(self.config.data.data_dir, "train"),
            transforms=self.transforms,
            phase="val",
        )

        val_dataset = RestructionDataset(
            dir=os.path.join(self.config.data.data_dir, "val"),
            transforms=self.transforms,
            phase="val",
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.sampling.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        return train_loader, val_loader

    def get_test_loaders(self):
        print("=> Utilizing the RestructionDataset() for test data loading...")
        test_dataset = RestructionDataset(
            dir=os.path.join(self.config.data.data_dir, "test"),
            transforms=self.transforms,
            phase="test",
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.testing.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
        )
        return test_loader


class RestructionDataset(torch.utils.data.Dataset):
    def __init__(self, dir, transforms, phase="train"):
        super().__init__()
        source_dir = dir
        self.phase = phase
        # if data_type == "coco":
        dir_deg1 = os.path.join(source_dir, "ImageA")
        dir_deg2 = os.path.join(source_dir, "ImageB")
        dir_gt = os.path.join(source_dir, "ImageS")
        # elif data_type == ""

        deg1_names, deg2_names, gt_names = [], [], []
        file_list = natsorted(os.listdir(dir_gt))
        for item in file_list:
            if item.endswith(".jpg") or item.endswith(".png") or item.endswith(".bmp"):
                deg1_names.append(os.path.join(dir_deg1, item))
                deg2_names.append(os.path.join(dir_deg2, item))
                gt_names.append(os.path.join(dir_gt, item))
        # print("The number of the training dataset is: {}".format(len(gt_names)))
        if self.phase == "train":
            x = list(enumerate(deg1_names))
            random.shuffle(x)
            indices, deg1_names = zip(*x)
            deg2_names = [deg2_names[idx] for idx in indices]
            gt_names = [gt_names[idx] for idx in indices]

        self.deg1_names = deg1_names
        self.deg2_names = deg2_names
        self.gt_names = gt_names
        self.transforms = transforms

    def __len__(self):
        return len(self.gt_names)

    def __getitem__(self, idx):
        deg1_name = self.deg1_names[idx]
        deg2_name = self.deg2_names[idx]
        gt_name = self.gt_names[idx]
        img_id = re.split("/", deg1_name)[-1]
        deg1_img = Image.open(deg1_name).convert("RGB")
        deg2_img = Image.open(deg2_name).convert("RGB")
        gt_img = Image.open(gt_name).convert("RGB")

        res = torch.cat(
            [
                self.transforms(deg1_img),
                self.transforms(deg2_img),
                self.transforms(gt_img),
            ],
            dim=0,
        )
        return (res, img_id)


# 融合数据集
class Fusion:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Lambda(lambda x: (x - 0.5) * 2),
            ]
        )

    def get_fusion_loaders(self, parent_dir, data_type, batch_size, num_works=24):
        print("=> Utilizing the RestructionDataset() for fusion data loading...")
        fusion_dataset = FusionDataset(
            dir=parent_dir,
            transforms=self.transforms,
            data_type=data_type,
        )

        fusion_loader = torch.utils.data.DataLoader(
            fusion_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_works,
            pin_memory=True,
            drop_last=True,
        )
        return fusion_loader


class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, dir, transforms, data_type="coco"):
        super().__init__()
        # if data_type == "coco":
        ir_path = os.path.join(dir, "ir")
        vi_path = os.path.join(dir, "vi")
        # elif data_type == ""

        ir_names, vi_names = [], []
        file_list = natsorted(os.listdir(ir_path))
        for item in file_list:
            if item.endswith(".jpg") or item.endswith(".png") or item.endswith(".bmp"):
                ir_names.append(os.path.join(ir_path, item))
                vi_names.append(os.path.join(vi_path, item))
        # print("The number of the training dataset is: {}".format(len(gt_names)))

        self.ir_names = ir_names
        self.vi_names = vi_names
        self.transforms = transforms

    def __len__(self):
        return len(self.ir_names)

    def __getitem__(self, idx):
        ir_name = self.ir_names[idx]
        vi_name = self.vi_names[idx]
        img_id = re.split("/", ir_name)[-1]
        ir_img = Image.open(ir_name).convert("RGB")
        vi_img = Image.open(vi_name).convert("RGB")
        res = torch.cat(
            [
                self.transforms(ir_img),
                self.transforms(vi_img),
            ],
            dim=0,
        )
        return (res, img_id)
