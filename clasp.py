import torch
import torch.nn.functional as F
import numpy as np


from tools import (
    clustering_methods,
    datasets,
    models,
    post_processing,
    visualize,
)


@torch.no_grad()
def compute_affinity_matrix(patch_descriptors, gamma=1, threshold=0):

    features = torch.nn.functional.normalize(patch_descriptors, p=2, dim=1)
    A = torch.mm(features, features.t())
    A = torch.relu(A)
    A = torch.where(A > threshold, A, torch.zeros_like(A))
    A = torch.pow(A, gamma)
    A.fill_diagonal_(0)

    return A


@torch.no_grad()
def compute_eigengaps(A, max_k=50):
    """
    Выполняет спектральное разложение и
    считает зазоры между собственными значениями.
    Args:
        A: Матрица сходства [N, N]
        max_k: Сколько первых значений проверяем (обычно 50 достаточно для COCO)
    Returns:
        eigenvalues, delta
    """
    eigenvalues, _ = torch.linalg.eigh(A)
    eigenvalues = torch.flip(eigenvalues, dims=[0])
    # eigenvectors = torch.flip(eigenvectors, dims=[1])
    eigenvalues = eigenvalues[:max_k]
    # vecs = eigenvectors[:, :max_k]
    delta = eigenvalues[:-1] - eigenvalues[1:]
    return eigenvalues, delta


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


if __name__ == "__main__":
    parameters = {
        "size_img": 1.5,
        "threshold": 0.0,
        "gamma": 5,
        "beta": 0.5,
        "max_k": 150,
        "sxy_crf": 3,
        "compat_crf": 15,
        "encoder": "dinov2_vitb14",
    }
    PATCH_SIZE = 14
    ann_file = "datasets/assets"
    img_dir = "examples"

    model, device = models.load_dino_model(model_name=parameters["encoder"])
    dataset = datasets.SimpleDataset(
        img_dir=img_dir, patch_size=PATCH_SIZE, size_img=parameters["size_img"]
    )

    n = min(4, len(dataset))

    mask_preds = []
    img_origs = []
    count_classes = []
    masks_without_crf = []
    best_ss = []
    best_dbcvs = []
    best_k_dbcvs = []

    for i in range(n):
        img = dataset[i]
        new_w = img["new_w"]
        new_h = img["new_h"]
        num_patches_w = new_w // PATCH_SIZE
        num_patches_h = new_h // PATCH_SIZE
        num_patches = num_patches_w * num_patches_h

        patch_features = models.extract_dino_features(
            model, device, img["img_tensor"]
        )
        A = compute_affinity_matrix(
            patch_features, parameters["gamma"], parameters["threshold"]
        )
        _, delts = compute_eigengaps(A, parameters["max_k"])
        k_opt = find_elbow_point(delts)
        labels, best_k, best_k_dbcv, best_s, best_dbcv = (
            clustering_methods.find_best_k(
                A, patch_features, k_opt, parameters["beta"]
            )
        )

        # best_k, labels = find_clusters_hdbscan(patch_descriptors, min_cluster_size=3, min_samples=25)
        # best_s = 0

        mask_small = labels.reshape(num_patches_h, num_patches_w)
        mask_tensor = torch.from_numpy(mask_small).float()[None, None, :, :]
        orig_h, orig_w = img["img_orig"].shape[:2]
        full_mask = F.interpolate(
            mask_tensor, size=(orig_h, orig_w), mode="nearest"
        )
        full_mask = full_mask.squeeze().byte().cpu().numpy()

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
        refined_probs = post_processing.dense_crf(
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
        best_dbcvs.append(best_dbcv)
        best_k_dbcvs.append(best_k_dbcv)

    visualize.visualize_segmentation(
        n,
        parameters,
        img_origs,
        None,
        masks_without_crf,
        mask_preds,
        count_classes,
        best_k_dbcvs,
        best_ss,
        best_dbcvs,
        None,
        "result_img",
    )
