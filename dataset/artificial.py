import os
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .image_utils import extract_random_roi


# class FlagOnGroundDataset(Dataset):
#     def __init__(self, flags_path: str, desert_path: str, device=None, apply_augmentations=False):
#         """
#         Creates a dataset of flags randomly positioned in a random desert image in a random position
#         :param flags_path: path to the flags folder
#         :param desert_path: path to the images folder
#         :param device: device to work on, leave None to find best
#         :param apply_augmentations: Whether to augment the desert or not
#         """
#         if device is None:
#             device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#         self.device = device
#
#         self.min_flag_percentage = 0.025
#         self.max_flag_percentage = 0.035
#
#         self.flags = glob(os.path.join(flags_path, '*.png'))
#         self.deserts = glob(os.path.join(desert_path, '*.jpeg'))
#         self.apply_augmentations = apply_augmentations
#
#         # Basic transformations including resizing, cropping, and normalization
#         self.preprocess = transforms.ToTensor()
#
#         # Other augmentations (excluding rotation)
#         self.augmentation_pipeline = transforms.Compose([
#             transforms.RandomApply([
#                 transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.3),
#             transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
#         ])
#
#     def __len__(self):
#         return len(min(self.flags, self.deserts))
#
#     def __getitem__(self, bg_idx, flag_idx=-1):
#         # Load desert and flag images
#         desert_path = self.deserts[bg_idx]
#         desert = cv2.imread(desert_path)
#         desert = cv2.cvtColor(desert, cv2.COLOR_BGR2RGB)
#
#         flag_path = np.random.choice(self.flags) if flag_idx < 0 else self.flags[flag_idx]
#         flag = cv2.imread(flag_path)
#         flag = cv2.cvtColor(flag, cv2.COLOR_BGR2RGB)
#
#         # Generate random ROI for flag placement
#         flag_x_min, flag_y_min, flag_w, flag_h = extract_random_roi(desert.shape[:2], self.min_flag_percentage,
#                                                                     self.max_flag_percentage, ratio=2,
#                                                                     )
#         flag_y_max = flag_y_min + flag_h
#         flag_x_max = flag_x_min + flag_w
#
#         # Resize and place flag on desert image
#         flag = cv2.resize(flag, (flag_w, flag_h))
#         desert[flag_y_min: flag_y_max, flag_x_min: flag_x_max] = flag
#
#         # Convert to PIL for other augmentations if enabled
#         if self.apply_augmentations:
#             # Convert desert back to array for OpenCV rotation
#             desert = np.array(self.augmentation_pipeline(Image.fromarray(desert)))
#
#         # Final preprocessing
#         desert = self.preprocess(Image.fromarray(desert))
#
#         # Adjust bounding box if rotation is applied (optional step)
#         return desert.to(self.device), torch.tensor([flag_x_min, flag_y_min, flag_x_max, flag_y_max],
#                                                     device=self.device)



class FlagOnGroundWithMask(Dataset):
    def __init__(self, flags_path: str, desert_path: str, device=None, apply_augmentations=False):
        """
        Creates a dataset of flags randomly positioned in a random desert image in a random position
        :param flags_path: path to the flags folder
        :param desert_path: path to the images folder
        :param device: device to work on, leave None to find best
        :param apply_augmentations: Whether to augment the desert or not
        """
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device

        self.min_flag_percentage = 0.025
        self.max_flag_percentage = 0.035

        self.flags = glob(os.path.join(flags_path, '*.png'))
        self.deserts = glob(os.path.join(desert_path, '*.jp*g'))
        self.apply_augmentations = apply_augmentations

        # Basic transformations including resizing, cropping, and normalization
        self.preprocess = transforms.ToTensor()

        # Other augmentations (excluding rotation)
        self.augmentation_pipeline = transforms.Compose([
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.3),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ])

    def __len__(self):
        return min(len(self.flags), len(self.deserts))

    def __getitem__(self, bg_idx, flag_idx=-1):
        # Load desert and flag images
        desert_path = self.deserts[bg_idx]
        desert = cv2.imread(desert_path)
        desert = cv2.cvtColor(desert, cv2.COLOR_BGR2RGB)

        flag_path = np.random.choice(self.flags) if flag_idx < 0 else self.flags[flag_idx]
        flag = cv2.imread(flag_path)
        flag = cv2.cvtColor(flag, cv2.COLOR_BGR2RGB)

        # Generate random ROI for flag placement
        flag_x_min, flag_y_min, flag_w, flag_h = extract_random_roi(
            desert.shape[:2], self.min_flag_percentage, self.max_flag_percentage, ratio=2
        )
        flag_y_max = flag_y_min + flag_h
        flag_x_max = flag_x_min + flag_w

        # Resize and place flag on desert image
        flag = cv2.resize(flag, (flag_w, flag_h))
        desert[flag_y_min: flag_y_max, flag_x_min: flag_x_max] = flag

        # Create binary mask
        mask = np.zeros(desert.shape[:2], dtype=np.uint8)
        mask[flag_y_min: flag_y_max, flag_x_min: flag_x_max] = 1

        # Convert to PIL for augmentations
        desert_pil = Image.fromarray(desert)
        mask_pil = Image.fromarray(mask)

        if self.apply_augmentations:
            augmented = self.augmentation_pipeline(desert_pil)
            desert = np.array(augmented)
            mask = np.array(mask_pil)  # Ensure mask augmentation syncs with image

        # Final preprocessing
        desert = self.preprocess(desert_pil)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return desert.to(self.device), mask.to(self.device)