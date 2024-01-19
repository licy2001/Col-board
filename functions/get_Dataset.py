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
            data_type=self.config.data.data_type,
        )

        val_dataset = RestructionDataset(
            dir=os.path.join(self.config.data.data_dir, "val"),
            transforms=self.transforms,
            phase="val",
            data_type=self.config.data.data_type,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.sampling.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    def get_test_loaders(self):
        print("=> Utilizing the RestructionDataset() for test data loading...")
        test_dataset = RestructionDataset(
            dir=os.path.join(self.config.data.data_dir, "test"),
            transforms=self.transforms,
            phase="test",
            data_type=self.config.data.data_type,
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
    def __init__(self, dir, transforms, phase="train", data_type="coco"):
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
