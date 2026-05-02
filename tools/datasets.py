import os
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


def get_high_quality_coco_ids(ann_file, num_imgs=100):
    coco = COCO(ann_file)
    # Выбираем изображения, где суммарная площадь объектов > 50% кадра
    img_ids = []
    k_gt = []
    for img_id in coco.getImgIds():
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        
        total_area = sum([ann['area'] for ann in anns])
        img_area = img_info['width'] * img_info['height']
        
        # Фильтр: покрытие > 50% и не слишком много мелких объектов (от 2 до 6)
        # if 0.5 < (total_area / img_area) < 0.9 and 2 <= len(anns) <= 6:
        #     img_ids.append(img_id)
        if len(img_ids) >= num_imgs:
            break
        img_ids.append(img_id)
        k_gt.append(len(anns))
    return img_ids, k_gt


class CocoClaspDataset(Dataset):
    def __init__(
        self, img_dir: str, ann_file, patch_size: int, mapping_level=None
    ):
        self.patch_size = patch_size
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.count_img = len(os.listdir(img_dir))
        self.ids = list(self.coco.imgs.keys())
        self.cat_id_to_name = {
            cat["id"]: cat["name"]
            for cat in self.coco.loadCats(self.coco.getCatIds())
        }
        self.name_to_cat_id = {
            cat["name"]: cat["id"]
            for cat in self.coco.loadCats(self.coco.getCatIds())
        }

        # Создаем итоговый цифровой маппинг ID -> ID
        self.mapping_level = {}
        for child_name, parent_name in mapping_level.items():
            if (
                child_name in self.name_to_cat_id
                and parent_name in self.name_to_cat_id
            ):
                child_id = self.name_to_cat_id[child_name]
                parent_id = self.name_to_cat_id[parent_name]
                self.mapping_level[child_id] = parent_id
            else:
                print(
                    f"Warning: Category names not found: {child_name} or {parent_name}"
                )
        print(self.mapping_level)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        f_name = img_info.get("file_name")

        path = os.path.join(self.img_dir, f_name)
        image_pil = Image.open(path).convert("RGB")
        w, h = image_pil.size

        new_w = (w // self.patch_size) * self.patch_size
        new_h = (h // self.patch_size) * self.patch_size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        anns = sorted(anns, key=lambda x: x["area"], reverse=True)

        gt_mask_l1 = np.zeros((h, w), dtype=np.int32)
        gt_mask_l2 = np.zeros((h, w), dtype=np.int32)
        l2_map_value = list(self.mapping_level.values())
        for ann in anns:
            mask = self.coco.annToMask(ann)
            cat_id = ann["category_id"]
            if cat_id not in l2_map_value:
                gt_mask_l2[mask > 0] = cat_id
            cat_id_l1 = self.mapping_level.get(cat_id, cat_id)
            gt_mask_l1[mask > 0] = cat_id_l1

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

        gt_mask_pil_l1 = Image.fromarray(gt_mask_l1.astype(np.int32))
        gt_mask_pil_l2 = Image.fromarray(gt_mask_l2.astype(np.int32))

        # gt_mask_resized = np.array(
        #     gt_mask_pil.resize((new_w, new_h), resample=Image.NEAREST)
        # )

        return {
            "img_tensor": image_tensor,
            "img_orig": np.array(
                image_pil
            ),  # np.array(image_pil.resize((new_w, new_h))),
            "gt_mask_l1": np.array(gt_mask_pil_l1),
            "gt_mask_l2": np.array(gt_mask_pil_l2),
            "img_id": img_id,
            "category_ids": [ann["category_id"] for ann in anns],
            "new_w": new_w,
            "new_h": new_h,
        }

    def __len__(self):
        return self.count_img


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
