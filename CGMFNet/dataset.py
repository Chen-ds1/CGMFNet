import os

import cv2
import numpy as np
import torch.utils.data

class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        img_ids,
        rgb_dir,
        depth_dir,
        rgb_mask_dir,
        depth_mask_dir,
        ext,
        num_classes,
        transform=None,
    ):
        self.img_ids = img_ids
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.rgb_mask_dir = rgb_mask_dir
        self.depth_mask_dir = depth_mask_dir
        self.ext = ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    @staticmethod
    def _read_image(path, flags=cv2.IMREAD_COLOR):
        image = cv2.imread(path, flags)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return image

    def _read_mask(self, mask_dir, img_id):
        channels = []
        for class_index in range(self.num_classes):
            path = os.path.join(
                mask_dir, str(class_index), img_id + self.ext
            )
            mask = self._read_image(path, cv2.IMREAD_GRAYSCALE)
            channels.append(mask[..., None])
        return np.dstack(channels)

    @staticmethod
    def _image_to_chw(image):

        image = image.astype(np.float32) / 255.0
        return image.transpose(2, 0, 1)

    @staticmethod
    def _mask_to_chw(mask):
        mask = (mask > 127).astype(np.float32)
        return mask.transpose(2, 0, 1)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        rgb_img = self._read_image(
            os.path.join(self.rgb_dir, img_id + self.ext)
        )
        depth_img = self._read_image(
            os.path.join(self.depth_dir, img_id + self.ext)
        )
        rgb_mask = self._read_mask(self.rgb_mask_dir, img_id)
        depth_mask = self._read_mask(self.depth_mask_dir, img_id)

        if self.transform is not None:
            augmented = self.transform(
                image=rgb_img,
                depth_image=depth_img,
                mask=rgb_mask,
                depth_mask=depth_mask,
            )
            rgb_img = augmented["image"]
            depth_img = augmented["depth_image"]
            rgb_mask = augmented["mask"]
            depth_mask = augmented["depth_mask"]

        rgb_img = self._image_to_chw(rgb_img)
        depth_img = self._image_to_chw(depth_img)
        rgb_mask = self._mask_to_chw(rgb_mask)
        depth_mask = self._mask_to_chw(depth_mask)

        return (
            rgb_img,
            rgb_mask,
            depth_img,
            depth_mask,
            {"img_id": img_id},
        )
