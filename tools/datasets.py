import os
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


class CocoClaspDataset(Dataset):
    def __init__(self, img_dir: str, ann_file, patch_size: int):
        self.patch_size = patch_size
        if ann_file is not None:
            self.coco = COCO(ann_file)
        else:
            self.coco = ann_file
        self.img_dir = img_dir
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]

        f_name = img_info.get("file_name") or os.path.basename(
            img_info.get("coco_url", "")
        )

        if not f_name:
            raise KeyError(
                f"Не нашел ни file_name, ни coco_url в img_info: {img_info}"
            )

        path = os.path.join(self.img_dir, f_name)
        image_pil = Image.open(path).convert("RGB")
        w, h = image_pil.size
        new_w = (w // self.patch_size) * self.patch_size
        new_h = (h // self.patch_size) * self.patch_size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        gt_mask = np.zeros((h, w), dtype=np.int32)
        for i, ann in enumerate(anns):
            if isinstance(ann["segmentation"], dict):
                return self.__getitem__((index + 1) % len(self))
            mask = self.coco.annToMask(ann)
            gt_mask[mask > 0] = i + 1

        transform = transforms.Compose(
            [
                transforms.Resize((new_h, new_w)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image_tensor = transform(image_pil)

        return {
            "img_tensor": image_tensor,
            "img_orig": np.array(image_pil),
            "gt_mask": gt_mask,
            "img_id": img_id,
            "w": w,
            "h": h,
            "new_w": new_w,
            "new_h": new_h,
        }

    def __len__(self):
        return len(self.ids)


class SimpleDataset(Dataset):
    def __init__(self, img_dir: str, patch_size: int, size_img: int):
        self.img_dir = img_dir
        self.patch_size = patch_size
        self.size_img = size_img
        self.img_names = []
        for f in os.listdir(img_dir):
            if f.lower().endswith(("png", "jpg", "jpeg")):
                self.img_names.append(f)

    def __getitem__(self, index):
        path = os.path.join(self.img_dir, self.img_names[index])
        image_pil = Image.open(path).convert("RGB")
        w, h = image_pil.size
        new_w = (int(w * self.size_img) // self.patch_size) * self.patch_size
        new_h = (int(h * self.size_img) // self.patch_size) * self.patch_size
        transform = transforms.Compose(
            [
                transforms.Resize((new_h, new_w)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image_tensor = transform(image_pil)
        return {
            "img_tensor": image_tensor,
            "img_pil": image_pil,
            "img_orig": np.array(image_pil),
            "gt_mask": None,
            "img_id": self.img_names[index],
            "w": w,
            "h": h,
            "new_w": new_w,
            "new_h": new_h,
        }

    def __len__(self):
        return len(self.img_names)
