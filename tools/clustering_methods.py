import hdbscan
from sklearn.decomposition import PCA
import warnings
import numpy as np
import torch
from sklearn.cluster import spectral_clustering, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    jaccard_score,
)
from typing import Tuple, Any
import torch.nn.functional as F
from . import post_processing
import time
import umap
from sklearn.preprocessing import StandardScaler, normalize
from .diffcut import DiffCut


@torch.no_grad()
def calculate_miou(pred_mask, gt_mask):
    """
    return iou, remapped_pred, len(gt_ids)
    """
    pred_ids = np.unique(pred_mask)
    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[gt_ids != 0]  # убираем фон из GT если он 0

    remapped_pred = np.zeros_like(pred_mask)

    for p_id in pred_ids:
        # Находим, с каким классом GT этот кластер пересекается больше всего
        if p_id == 0:
            continue

        intersect_counts = []
        for g_id in gt_ids:
            intersect = np.logical_and(
                pred_mask == p_id, gt_mask == g_id
            ).sum()
            intersect_counts.append((intersect, g_id))

        if intersect_counts:
            best_gt_id = max(intersect_counts, key=lambda x: x[0])[1]
            remapped_pred[pred_mask == p_id] = best_gt_id

    iou = jaccard_score(
        gt_mask.flatten(), remapped_pred.flatten(), average="macro"
    )

    return iou, remapped_pred, len(gt_ids)


@torch.no_grad()
def pre_miou(
    labels, best_k_sil, num_patches_h, num_patches_w, img_data, parameters
):
    mask_small = labels.reshape(num_patches_h, num_patches_w)
    mask_tensor = torch.from_numpy(mask_small).float()[None, None, :, :]
    orig_h, orig_w = img_data["img_orig"].shape[:2]
    full_mask = F.interpolate(
        mask_tensor,
        size=(orig_h, orig_w),
        mode="nearest",
    )
    full_mask = full_mask.squeeze().byte().cpu().numpy()
    mask_one_hot = np.zeros(
        (best_k_sil, num_patches_h, num_patches_w),
        dtype=np.float32,
    )
    for k in range(best_k_sil):
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
    original_image = np.ascontiguousarray(img_data["img_orig"])
    refined_probs = post_processing.dense_crf(
        original_image,
        full_mask_probs,
        parameters["sxy_crf"],
        parameters["compat_crf"],
    )
    mask_pred = np.argmax(refined_probs, axis=0)
    return mask_pred, full_mask


@torch.no_grad()
def find_clusters_hdbscan(
    patch_descriptors, min_cluster_size=40, min_samples=5
) -> Tuple[int, Any]:
    """
    min_cluster_size: Минимальное количество патчей, чтобы считать их объектом.
    min_samples: Насколько консервативен алгоритм (чем меньше, тем больше шума
                превращается в кластеры).

    return num_clusters, labels
    """
    X_norm = normalize(patch_descriptors)

    reducer = umap.UMAP(n_components=64, metric="cosine")
    X_reduced = reducer.fit_transform(X_norm)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_epsilon=0.05,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X_reduced)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if -1 in labels:
        labels[labels == -1] = num_clusters
        num_clusters += 1

    return num_clusters, labels


@torch.no_grad()
def evaluate_with_dbcv(X, labels) -> float:
    """
    Псевдометрика dbcv
    X: embeddings

    return score
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    X = X.astype(np.float64)

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / (norms + 1e-8)

    unique_labels = np.unique(labels)
    if len(unique_labels[unique_labels != -1]) <= 1:
        return -1.0

    try:
        pca = PCA(n_components=50)
        X_for_score = pca.fit_transform(X_normalized)
        score = hdbscan.validity.validity_index(X_for_score, labels)
        if np.isnan(score):
            return -1.0

    except Exception as e:
        return -1.0

    return score


@torch.no_grad()
def find_best_k_for_SAM(A, embeddings, k_opt, beta):
    start_k = int(np.floor(k_opt * (1 - beta)))
    end_k = int(np.ceil(k_opt * (1 + beta)))
    # ks = range(max(2, start_k), end_k + 1)
    ks = range(k_opt - 1, k_opt + 1)

    best_sil = -1
    best_dbcv = -1
    best_k_sil = k_opt
    best_k_dbcv = k_opt
    best_clast = None

    X = embeddings.detach().cpu().numpy()
    A = A.detach().cpu().numpy()

    for k in ks:
        labels = spectral_clustering(
            A, n_clusters=k, assign_labels="discretize"
        )
        score_sil = silhouette_score(X, labels, metric="cosine")
        score_dbcv = evaluate_with_dbcv(X, labels)

        if score_dbcv > best_dbcv:
            best_dbcv = score_dbcv
            best_k_dbcv = k
        if score_sil > best_sil:
            best_sil = score_sil
            best_k_sil = k
            best_clast = labels

    return (
        best_clast,
        best_k_sil,
        best_k_dbcv,
        best_sil,
        best_dbcv,
    )


@torch.no_grad()
def find_best_k(
    A,
    embeddings,
    k_opt,
    k_gt,
    beta,
    num_patches_h,
    num_patches_w,
    img_data,
    parameters,
):
    """
    Args:
        embeddings: собственные векторы [N_patches, k_opt]
        k_opt: найденный ранее "локоть"
        beta: ширина поиска
        num_patches_h,
        num_patches_w,
        img_data,
        parameters,

    Returns:
        tuple: (
        best_clast,
        best_k_sil,
        best_k_dbcv,
        best_sil,
        best_dbcv,
        best_mask_pred,
        best_mask_crf,
        all_stats,
    )
    """

    start_k = int(np.floor(k_opt * (1 - beta)))
    end_k = int(np.ceil(k_opt * (1 + beta)))
    # ks = range(max(2, start_k), end_k + 1)
    ks = range(max(2, k_opt - 5), k_opt + 5)
    # ks = [20,40,60]
    # taus = [0.3, 0.5, 0.7, 0.9]

    best_sil = -1
    best_dbcv = -1
    best_k_sil = k_opt
    best_k_dbcv = k_opt
    best_clast = None
    best_mask_pred = None
    best_mask_crf = None

    X = embeddings.detach().cpu().numpy()
    X_norm = normalize(X)
    A = A.detach().cpu().numpy()
    all_stats = []
    for k in ks:
        start_time_method = time.time()

        # k, labels = find_clusters_hdbscan(X, min_cluster_size=min_cluster_size, min_samples=5)

        # labels = DiffCut().generate_masks(
        #     features=embeddings,
        #     tau=tau,
        #     patch_size=parameters["patch_size"],
        #     alpha=parameters["gamma"],
        #     num_patches_h=num_patches_h,
        #     num_patches_w=num_patches_w,
        # )
        # k = len(np.unique(labels))
        if k <= 1:
            continue
        
        # labels = spectral_clustering(
        #     A, n_clusters=k, assign_labels="discretize"
        # )

        # model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        # labels = model.fit_predict(X_norm)

        kmeans = KMeans(n_clusters=k, n_init='auto')
        labels = kmeans.fit_predict(X)

        method_time = time.time() - start_time_method
        start_time = time.time()
        score_sil = silhouette_score(X, labels, metric="cosine")
        sil_time = time.time() - start_time
        start_time = time.time()
        score_dbcv = evaluate_with_dbcv(X, labels)
        dbcv_time = time.time() - start_time
        score_dbcv = (score_dbcv + 1) / 2
        start_time = time.time()
        dbi_score = davies_bouldin_score(X, labels)
        dbi_time = time.time() - start_time
        dbi_score = 1.0 / (1.0 + dbi_score)

        mask_pred, mask_crf = pre_miou(
            labels=labels,
            best_k_sil=k,
            num_patches_h=num_patches_h,
            num_patches_w=num_patches_w,
            img_data=img_data,
            parameters=parameters,
        )
        miou_l1, _, count_gt_class = calculate_miou(
            pred_mask=mask_pred, gt_mask=img_data["gt_mask_l1"]
        )
        # miou_l2, _ = calculate_miou(
        #     pred_mask=mask_pred, gt_mask=img_data["gt_mask_l2"]
        # )
        if score_dbcv > best_dbcv:
            best_dbcv = score_dbcv
            best_k_dbcv = k
        if score_sil > best_sil:
            best_sil = score_sil
            best_k_sil = k
            best_clast = labels
            best_mask_pred = mask_pred
            best_mask_crf = mask_crf

        all_stats.append(
            {
                "img_id": img_data["img_id"],
                "beta": beta,
                "gamma": parameters.get("gamma"),
                # "tau": tau,
                # "min_cluster_size":min_cluster_size,
                "k": k,
                'k_gt':k_gt,
                "count_gt_class": count_gt_class,
                "silhouette": score_sil,
                "dbcv": score_dbcv,
                "sd_8": score_sil * 0.8 + (1 - 0.8) * score_dbcv,
                "sd_7": score_sil * 0.7 + (1 - 0.7) * score_dbcv,
                "sd_6": score_sil * 0.6 + (1 - 0.6) * score_dbcv,
                "sd_5": score_sil * 0.5 + (1 - 0.5) * score_dbcv,
                "sd_4": score_sil * 0.4 + (1 - 0.4) * score_dbcv,
                "sd_3": score_sil * 0.3 + (1 - 0.3) * score_dbcv,
                "sd_2": score_sil * 0.2 + (1 - 0.2) * score_dbcv,
                "dbi": dbi_score,
                "sil_dbcv_geom": (score_sil * (1 - score_dbcv)) ** 0.5,
                "sil_dbi_geom": (score_sil * (1 - dbi_score)) ** 0.5,
                "dbcv_dbi_geom": (score_dbcv * (1 - dbi_score)) ** 0.5,
                "sil_time": sil_time,
                "dbcv_time": dbcv_time,
                "dbi_time": dbi_time,
                "method_time": method_time,
                "miou_l1": miou_l1,
            }
        )

    return (
        best_clast,
        best_k_sil,
        best_k_dbcv,
        best_sil,
        best_dbcv,
        best_mask_pred,
        best_mask_crf,
        all_stats,
    )
