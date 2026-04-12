import hdbscan
from sklearn.decomposition import PCA
import warnings
import numpy as np
import torch
from sklearn.cluster import spectral_clustering
from sklearn.metrics import silhouette_score
from typing import Tuple, Any


def find_clusters_hdbscan(
    patch_descriptors, min_cluster_size=5, min_samples=5
) -> Tuple[int, Any]:
    """
    min_cluster_size: Минимальное количество патчей, чтобы считать их объектом.
    min_samples: Насколько консервативен алгоритм (чем меньше, тем больше шума
                превращается в кластеры).

    return num_clusters, labels
    """
    X = patch_descriptors.detach().cpu().numpy()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if -1 in labels:
        labels[labels == -1] = num_clusters
        num_clusters += 1

    return num_clusters, labels


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


def find_best_k(A, embeddings, k_opt, beta=0.2):
    """
    Args:
        embeddings: собственные векторы [N_patches, k_opt]
        k_opt: найденный ранее "локоть"
        beta: ширина поиска

    Returns:
        tuple: (best_clast, best_k_sil, best_k_dbcv, best_sil, best_dbcv_for_sil, best_dbcv)
    """
    
    start_k = int(np.floor(k_opt * (1 - beta)))
    end_k = int(np.ceil(k_opt * (1 + beta)))
    ks = range(max(2, start_k), end_k + 1)

    best_sil = -1
    best_dbcv = -1
    best_dbcv_for_sil = -1
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
            best_dbcv_for_sil = score_dbcv
            best_k_sil = k
            best_clast = labels

    return (
        best_clast,
        best_k_sil,
        best_k_dbcv,
        best_sil,
        best_dbcv_for_sil,
        best_dbcv,
    )
