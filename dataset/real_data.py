import torch
import torch.nn as nn
import cv2
from torch.utils.data import Dataset
from glob import glob
import os
from torchvision import transforms


class RealDataset(Dataset):
    def __init__(self, images_path, masks_path, augment=False):
        self.masks_extension = '.png'
        self.images_extension = '.jpg'

        self.images_path = images_path
        self.masks_path = masks_path

        self.images = glob(os.path.join(images_path, f'*{self.images_extension}'))
        self.masks = glob(os.path.join(masks_path, f'*{self.masks_extension}'))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((1024, 1600)),
        ])

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, item):
        base_name = os.path.basename(self.masks[item]).split('.')[0]

        mask = self.transform(
            cv2.imread(os.path.join(self.masks_path, base_name + self.masks_extension))
        )
        img = self.transform(
            cv2.imread(os.path.join(self.images_path, base_name + self.images_extension))
        )

        return img, mask