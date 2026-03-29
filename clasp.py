import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import spectral_clustering
from sklearn.metrics import silhouette_score


def load_dinov2_model(
    repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14"
):
    print(f"Загрузка модели {model_name} из torch.hub...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load(repo_name, model_name)
    model.eval()
    model.to(device)
    print(f"Модель загружена на {device}.")
    return model, device


@torch.no_grad()
def extract_dino_features(model, device, img_tensor):
    img_tensor = img_tensor.to(device)
    features_dict = model.forward_features(img_tensor)
    patch_tokens = features_dict[
        "x_norm_patchtokens"
    ]  # [Batch=1, Num_Patches, Dim]
    return patch_tokens.squeeze(0)  # [Num_Patches, Dim]


@torch.no_grad()
def compute_affinity_matrix(patch_descriptors, gamma=1, treshold=0):

    features = torch.nn.functional.normalize(patch_descriptors, p=2, dim=1)
    A = torch.mm(features, features.t())
    A = torch.relu(A)
    A = torch.where(A > treshold, A, torch.zeros_like(A))
    A = torch.pow(A, gamma)
    A.fill_diagonal_(0)

    return A


@torch.no_grad()
def compute_eigengaps(A, max_k=50):
    """
    Выполняет спектральное разложение и 
    считает зазоры между собственными значениями.
    A: Матрица сходства [N, N]
    max_k: Сколько первых значений проверяем (обычно 50 достаточно для COCO)
    """
    eigenvalues, _ = torch.linalg.eigh(A)
    eigenvalues = torch.flip(eigenvalues, dims=[0])
    # eigenvectors = torch.flip(eigenvectors, dims=[1])
    vals = eigenvalues[:max_k]
    # vecs = eigenvectors[:, :max_k]
    delta = vals[:-1] - vals[1:]
    return vals, delta


def find_elbow_point(delts):
    """
    Находит точку 'локтя' на кривой собственных значений.
    """
    n = len(delts)
    p1 = np.array([0, delts[0].cpu().item()])
    p2 = np.array([n - 1, delts[-1].cpu().item()])
    distances = []
    for i in range(n):
        p0 = np.array([i, delts[i].cpu().item()])
        dist = np.abs(np.cross(p2 - p1, p1 - p0)) / np.linalg.norm(p2 - p1)
        distances.append(dist)
    k_optimal = np.argmax(distances) + 1
    return k_optimal


def find_best_k(A, embeddings, k_opt, beta=0.2):
    """
    embeddings: собственные векторы [N_patches, k_opt]
    k_opt: найденный ранее "локоть"
    beta: ширина поиска
    """
    start_k = int(np.floor(k_opt * (1 - beta)))
    end_k = int(np.ceil(k_opt * (1 + beta)))
    ks = range(max(2, start_k), end_k + 1)

    best_s = -1
    best_k = k_opt
    best_clast = None
    X = embeddings.detach().cpu().numpy()
    A = A.detach().cpu().numpy()

    for k in ks:
        labels = spectral_clustering(
            A, n_clusters=k, assign_labels="discretize"
        )
        score = silhouette_score(X, labels, metric="cosine")
        if score > best_s:
            best_s = score
            best_k = k
            best_clast = labels

    return best_k, best_clast


def visualize_segmentation(
    orig_img, gt_mask, pred_mask, num_clusters, img_id, save_dir
):

    fig = plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(orig_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(orig_img)
    plt.imshow(gt_mask, alpha=0.5, cmap="tab20")
    plt.title("COCO Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(orig_img)
    plt.imshow(pred_mask, alpha=0.5, cmap="tab20")
    plt.title(f"CLASP Prediction (K={num_clusters})")
    plt.axis("off")

    plt.tight_layout()
    # plt.show()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"result_{img_id}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


class CocoClaspDataset(Dataset):
    def __init__(self, img_dir, ann_file, patch_size):
        self.patch_size = patch_size
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]

        path = os.path.join(self.img_dir, img_info["file_name"])
        image_pil = Image.open(path).convert("RGB")
        w, h = image_pil.size
        new_w = (w // self.patch_size) * self.patch_size
        new_h = (h // self.patch_size) * self.patch_size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        gt_mask = np.zeros((h, w), dtype=np.int32)
        for i, ann in enumerate(anns):
            mask = self.coco.annToMask(ann)
            gt_mask[mask > 0] = i + 1

        # 3. Трансформация для модели (DINOv2)
        # Важно: здесь image_pil превращается в тензор,
        # но для визуализации нам понадобится и исходный массив
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


if __name__ == "__main__":
    PATCH_SIZE = 14
    ann_file = "datasets/coco/annotations.jsons"
    img_dir = "datasets/coco/images"
    dataset = CocoClaspDataset(img_dir, ann_file, PATCH_SIZE)
    model, device = load_dinov2_model(model_name="dinov2_vitb14")

    for i in range(len(dataset)):
        img = dataset[i]
        new_w = img["new_w"]
        new_h = img["new_h"]
        num_patches_w = new_w // PATCH_SIZE
        num_patches_h = new_h // PATCH_SIZE
        num_patches = num_patches_w * num_patches_h

        patch_descriptors = extract_dino_features(
            model, device, img["img_tensor"].unsqueeze(0)
        )
        A = compute_affinity_matrix(patch_descriptors, gamma=1, treshold=0)
        _, delts = compute_eigengaps(A, max_k=50)
        k_opt = find_elbow_point(delts)
        best_k, labels = find_best_k(A, patch_descriptors, k_opt, beta=0.2)

        mask_small = labels.reshape(num_patches_h, num_patches_w)
        mask_tensor = torch.from_numpy(mask_small).float()[None, None, :, :]
        full_mask = F.interpolate(
            mask_tensor, size=(new_h, new_w), mode="nearest"
        )
        full_mask = full_mask.squeeze().byte().cpu().numpy()
        visualize_segmentation(
            img["img_orig"],
            img["gt_mask"],
            full_mask,
            best_k,
            img["img_id"],
            "result_img",
        )
