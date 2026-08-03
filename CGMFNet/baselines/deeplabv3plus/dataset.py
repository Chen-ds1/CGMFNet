import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        image_suffix='.png',
        mask_suffix='.png',
        sample_ids=None,
        ignore_index=None,
        augment=False,
        input_size=None,
        binary_mask=True,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix
        self.ignore_index = ignore_index
        self.augment = augment
        self.input_size = tuple(input_size) if input_size is not None else None
        self.binary_mask = binary_mask

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f'Image directory not found: {image_dir}')
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(f'Mask directory not found: {mask_dir}')

        image_ids = {
            os.path.splitext(name)[0]
            for name in os.listdir(image_dir)
            if name.lower().endswith(image_suffix.lower())
        }
        mask_ids = {
            os.path.splitext(name)[0]
            for name in os.listdir(mask_dir)
            if name.lower().endswith(mask_suffix.lower())
        }
        available_ids = sorted(image_ids & mask_ids)

        if sample_ids is None:
            self.sample_ids = available_ids
        else:
            available = set(available_ids)
            missing = [sample_id for sample_id in sample_ids if sample_id not in available]
            if missing:
                raise FileNotFoundError(
                    f'{len(missing)} requested image-mask pairs are missing. '
                    f'First missing ID: {missing[0]}'
                )
            self.sample_ids = list(sample_ids)

        if not self.sample_ids:
            raise RuntimeError('No matching image-mask pairs were found.')

    def __len__(self):
        return len(self.sample_ids)

    def _augment_pair(self, image, mask):
        if np.random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if self.input_size is None:
            return image, mask

        target_height, target_width = self.input_size
        scale = np.random.uniform(0.75, 1.5)
        scaled_height = max(1, int(target_height * scale))
        scaled_width = max(1, int(target_width * scale))

        image = TF.resize(
            image,
            (scaled_height, scaled_width),
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask,
            (scaled_height, scaled_width),
            interpolation=TF.InterpolationMode.NEAREST,
        )

        pad_right = max(0, target_width - scaled_width)
        pad_bottom = max(0, target_height - scaled_height)
        if pad_right or pad_bottom:
            image = TF.pad(image, [0, 0, pad_right, pad_bottom], fill=0)
            mask_fill = self.ignore_index if self.ignore_index is not None else 0
            mask = TF.pad(mask, [0, 0, pad_right, pad_bottom], fill=mask_fill)

        current_width, current_height = image.size
        top = np.random.randint(0, current_height - target_height + 1)
        left = np.random.randint(0, current_width - target_width + 1)
        image = TF.crop(image, top, left, target_height, target_width)
        mask = TF.crop(mask, top, left, target_height, target_width)
        return image, mask

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        image_path = os.path.join(self.image_dir, sample_id + self.image_suffix)
        mask_path = os.path.join(self.mask_dir, sample_id + self.mask_suffix)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.augment:
            image, mask = self._augment_pair(image, mask)
        elif self.input_size is not None:
            image = TF.resize(
                image,
                self.input_size,
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            mask = TF.resize(
                mask,
                self.input_size,
                interpolation=TF.InterpolationMode.NEAREST,
            )

        image = TF.to_tensor(image)
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        mask = np.asarray(mask, dtype=np.int64)
        if self.binary_mask and mask.max(initial=0) > 1:
            mask = (mask > 127).astype(np.int64)
        return image, torch.from_numpy(mask.copy())
