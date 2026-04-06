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
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from transformers import AutoModel, AutoImageProcessor
import timm


# def load_dinov2_model(
#     repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14"
# ):
#     print(f"Загрузка модели {model_name} из torch.hub...")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = torch.hub.load(repo_name, model_name)
#     model.eval()
#     model.to(device)
#     print(f"Модель загружена на {device}.")
#     return model, device


def load_dinov2_model(model_name="dinov2_vitb14"):

    print(f"Загрузка модели {model_name} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timm_names = {
        "dinov2_vits14": "vit_small_patch14_dinov2",
        "dinov2_vitb14": "vit_base_patch14_dinov2",
        "dinov2_vitl14": "vit_large_patch14_dinov2",
    }
    dinov3_names = {
        "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m"
    }

    if model_name in timm_names:
        timm_name = timm_names.get(model_name, "vit_base_patch14_dinov2")
        model = timm.create_model(
            timm_name,
            pretrained=True,
            num_classes=0,
            global_pool="",
            dynamic_img_size=True,
        )

        if not hasattr(model, "mask_token"):
            model.mask_token = torch.nn.Parameter(
                torch.zeros(1, 1, model.embed_dim)
            )

        model = model.to(device)
        model.eval()
        
    else:
        processor = AutoImageProcessor.from_pretrained(dinov3_names[model_name])# , token=''
        model = AutoModel.from_pretrained(dinov3_names[model_name]).to(device)
        model.eval()
        model = (model, processor)
    
    return model, device


@torch.no_grad()
def extract_dinov2_features(model, device, img_tensor):
    img_tensor = img_tensor
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # Модель теперь выдает [1, 1370, 768]
    out = model(img_tensor)

    # Отрезаем CLS-токен (первый), оставляем только патчи [1369, 768]
    if out.ndim == 3 and out.shape[1] > 1:
        out = out[:, 1:, :]
    out = out.squeeze(0)
    return  out #[N_patches, Dim]

def extract_dinov3_features(model_bundle, img_tensor, device):
    model, _ = model_bundle 
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)    
    img_tensor = img_tensor.to(device)
    outputs = model(img_tensor) 
    features = outputs.last_hidden_state.squeeze(0)    
    patch_features = features[5:, :] 
    
    return patch_features


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

    return best_k, best_clast, best_s


def dense_crf(img, probs, sxy, compat):
    """
    img: исходное изображение (H, W, 3) тип uint8
    probs: вероятности классов от модели (C, H, W) тип float32
    """

    img = np.ascontiguousarray(img)
    probs = np.ascontiguousarray(probs)

    c, h, w = probs.shape

    d = dcrf.DenseCRF2D(w, h, c)
    unary = unary_from_softmax(probs.astype(np.float32))
    d.setUnaryEnergy(unary)

    d.addPairwiseGaussian(sxy=(sxy, sxy), compat=3)

    d.addPairwiseBilateral(
        sxy=(80, 80), srgb=(13, 13, 13), rgbim=img, compat=10
    )

    q = d.inference(10)

    return np.array(q).reshape((c, h, w))


def visualize_segmentation(
    count_img,
    parameters,
    orig_img,
    gt_mask,
    masks_without_crf,
    pred_mask,
    num_clusters,
    best_s,
    img_id,
    save_dir,
):

    count_img
    param = "".join([f"{key}:{val}  " for key, val in parameters.items()])
    fig = plt.figure(figsize=(15, 4 * count_img))

    for i in range(count_img):
        plt.subplot(count_img, 3, 3 * i + 1)
        plt.imshow(orig_img[i])
        plt.title("Original Image")
        plt.axis("off")

        # plt.subplot(1, 2, 2)
        # plt.imshow(orig_img)
        # plt.imshow(gt_mask, alpha=0.7, cmap="tab20")
        # plt.title("COCO Ground Truth")
        # plt.axis("off")

        plt.subplot(count_img, 3, 3 * i + 2)
        plt.imshow(orig_img[i])
        plt.imshow(masks_without_crf[i], alpha=0.8, cmap="tab10")
        plt.title(f"CLASP Prediction (K={num_clusters[i]}) sil_score={best_s[i]:.2f}")

        plt.subplot(count_img, 3, 3 * i + 3)
        plt.imshow(orig_img[i])
        plt.imshow(pred_mask[i], alpha=0.8, cmap="tab10")
        plt.title(f"CLASP Prediction with CRF (K={num_clusters[i]}) sil_score={best_s[i]:.2f}")
        plt.axis("off")
    plt.suptitle(f"{param}", y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{param}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=250)
    plt.close(fig)


class CocoClaspDataset(Dataset):
    def __init__(self, img_dir, ann_file, patch_size):
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
    def __init__(self, img_dir, patch_size):
        self.img_dir = img_dir
        self.patch_size = patch_size
        self.img_names = []
        for f in os.listdir(img_dir):
            if f.lower().endswith(("png", "jpg", "jpeg")):
                self.img_names.append(f)

    def __getitem__(self, index):
        path = os.path.join(self.img_dir, self.img_names[index])
        image_pil = Image.open(path).convert("RGB")
        w, h = image_pil.size
        new_w = (w // self.patch_size) * self.patch_size 
        new_h = (h // self.patch_size) * self.patch_size
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
            "img_pil":image_pil,
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


if __name__ == "__main__":
    parameters = {
        "treshold": 0,
        "gamma": 5,
        "beta": 0.4,
        "max_k": 150,
        "sxy_crf": 1,
        "compat_crf": 45,
        "encoder": "dinov3_vitl16",
    }
    PATCH_SIZE = 16
    ann_file = "datasets/assets"
    img_dir = "examples"
    # dataset = CocoClaspDataset(img_dir, ann_file, PATCH_SIZE)
    dataset = SimpleDataset(img_dir, PATCH_SIZE)
    model, device = load_dinov2_model(model_name=parameters["encoder"])

    n = min(4, len(dataset))

    mask_preds = []
    img_origs = []
    count_classes = []
    masks_without_crf = []
    best_ss = []

    for i in range(n):
        img = dataset[i]
        new_w = img["new_w"]
        new_h = img["new_h"]
        num_patches_w = new_w // PATCH_SIZE
        num_patches_h = new_h // PATCH_SIZE
        num_patches = num_patches_w * num_patches_h

        # patch_descriptors = extract_dino_features(
        #     model, device, img["img_tensor"].unsqueeze(0)
        # )
        patch_descriptors = extract_dinov3_features(
            model, img["img_tensor"], device
        )
        A = compute_affinity_matrix(
            patch_descriptors, parameters["gamma"], parameters["treshold"]
        )
        _, delts = compute_eigengaps(A, parameters["max_k"])
        k_opt = find_elbow_point(delts)
        best_k, labels, best_s = find_best_k(
            A, patch_descriptors, k_opt, parameters["beta"]
        )
        mask_small = labels.reshape(num_patches_h, num_patches_w)

        mask_tensor = torch.from_numpy(mask_small).float()[None, None, :, :]
        full_mask = F.interpolate(
            mask_tensor, size=(new_h, new_w), mode="nearest"
        )
        full_mask = full_mask.squeeze().byte().cpu().numpy()

        orig_h, orig_w = img["img_orig"].shape[:2]
        mask_one_hot = np.zeros(
            (best_k, num_patches_h, num_patches_w), dtype=np.float32
        )
        for k in range(best_k):
            mask_one_hot[k] = (mask_small == k).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_one_hot).unsqueeze(0)
        full_mask_probs = (
            F.interpolate(
                mask_tensor,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .numpy()
        )
        original_image = np.ascontiguousarray(img["img_orig"])
        refined_probs = dense_crf(
            original_image,
            full_mask_probs,
            parameters["sxy_crf"],
            parameters["compat_crf"],
        )
        mask_pred = np.argmax(refined_probs, axis=0)

        img_origs.append(img["img_orig"])
        mask_preds.append(mask_pred)
        count_classes.append(best_k)
        masks_without_crf.append(full_mask)
        best_ss.append(best_s)

    visualize_segmentation(
        n,
        parameters,
        img_origs,
        None,
        masks_without_crf,
        mask_preds,
        count_classes,
        best_ss,
        None,
        "result_img",
    )

    # visualize_segmentation(
    #     parameters,
    #     img["img_orig"],
    #     img["gt_mask"],
    #     mask_pred,
    #     best_k,
    #     img["img_id"],
    #     "result_img",
    # )
