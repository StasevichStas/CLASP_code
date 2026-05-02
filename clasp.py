import torch
import torch.nn.functional as F
import numpy as np
import time
import pandas as pd
import os
import random

# from sam2.build_sam import build_sam2
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import json

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
    # eigenvalues = eigenvalues[:max_k]
    # vecs = eigenvectors[:, :max_k]
    delta = eigenvalues[:-1] - eigenvalues[1:]
    return eigenvalues, delta


@torch.no_grad()
def find_elbow_point(patch_features, gamma=None):
    """
    Находит точку 'локтя' на кривой собственных значений.
    """
    max_norm_dist = 0
    best_dist = None
    best_A = None
    best_gamma = None
    gamma_range = [gamma] if gamma else np.arange(1, 17, 3)
    for gamma in gamma_range:
        A = compute_affinity_matrix(patch_features, gamma, 0)
        _, delts = compute_eigengaps(A)
        delts = delts.cpu().numpy()
        n = len(delts)
        distances = []
        delts = (delts - delts.min()) / (delts.max() - delts.min())
        y = np.arange(0, n, 1)
        y = (y - y.min()) / (y.max() - y.min())
        p1 = np.array([y[0], delts[0]])
        p2 = np.array([y[-1], delts[-1]])
        for i, y_i in enumerate(y):
            p0 = np.array([y_i, delts[i]])
            dist = np.abs(np.cross(p2 - p1, p1 - p0)) / np.linalg.norm(p2 - p1)
            distances.append(dist)
        k_optimal = np.argmax(distances)
        while k_optimal >= 15:
            distances[k_optimal] = -1
            k_optimal = np.argmax(distances)
        if max(distances) > max_norm_dist:
            max_norm_dist = max(distances)
            best_dist = distances
            best_A = A
            best_gamma = gamma
    k_optimal = np.argmax(best_dist) + 1
    return k_optimal, best_A, best_gamma


@torch.no_grad()
def BF(
    model, device, ann_file, img_dir, patch_size, parameters, mapping_level
):
    # range_gamma = [(7, 13)]
    # range_beta = [(0.1, 0.4)]
    range_size_img = [1]
    time_model_list = []
    start_time = time.time()
    counter = 0
    for size_img in range_size_img:
        parameters["size_img"] = size_img
        dataset = datasets.CocoClaspDataset(
            img_dir=img_dir,
            ann_file=ann_file,
            patch_size=patch_size,
            mapping_level=mapping_level,
        )
        print(f"len(dataset) = {len(dataset)}")
        n = min(20, len(dataset))
        n = 100
        index_choice, k_gt = datasets.get_high_quality_coco_ids(
            ann_file=ann_file, num_imgs=n
        )
        with open("choice_idx_100.json", "w") as f:
            json.dump(index_choice, f)

        # for start_gamma, end_gamma in range_gamma:
        # for gamma in np.arange(start_gamma, end_gamma, 1):
        # parameters["gamma"] = gamma
        # for start_beta, end_beta in range_beta:
        # for beta in np.arange(start_beta, end_beta, 0.1):
        # parameters["beta"] = beta
        # mask_preds = []
        # img_origs = []
        # count_classes = []
        # masks_without_crf = []
        # best_ss = []
        # best_k_dbcvs = []
        # best_dbcvs = []
        # list_parameters = []
        with open("choice_idx_100.json", "r") as f:
            index_choice = json.load(f)
        result_stats = []
        i = 0
        for idx in index_choice:
            for j in range(1000000):
                img_data = dataset[j]
                if img_data["img_id"] == idx:
                    break
            new_w = img_data["new_w"]
            new_h = img_data["new_h"]
            num_patches_w = new_w // patch_size
            num_patches_h = new_h // patch_size
            num_patches = num_patches_w * num_patches_h
            start_time_model = time.time()
            patch_features = models.extract_dino_features(
                model,
                device,
                img_data["img_tensor"],
                model_type=parameters["encoder"],
            )
            time_model = time.time() - start_time_model
            time_model_list.append(time_model)
            k_opt, A, best_gamma = find_elbow_point(patch_features)
            parameters["gamma"] = best_gamma
            (
                best_clast,
                best_k_sil,
                best_k_dbcv,
                best_sil,
                best_dbcv,
                best_mask_pred,
                best_mask_crf,
                all_stats,
            ) = clustering_methods.find_best_k(
                A,
                patch_features,
                k_opt,
                k_gt[i],
                parameters["beta"],
                num_patches_h,
                num_patches_w,
                img_data,
                parameters,
            )
            # best_k, labels = find_clusters_hdbscan(patch_descriptors, min_cluster_size=3, min_samples=25)
            # best_s = 0
            # img_origs.append(img_data["img_orig"])
            # mask_preds.append(best_mask_pred)
            # count_classes.append(best_k_sil)
            # masks_without_crf.append(best_mask_crf)
            # best_ss.append(best_sil)
            # best_k_dbcvs.append(best_k_dbcv)
            # best_dbcvs.append(best_dbcv)
            # list_parameters.append(parameters.copy())
            result_stats.extend(all_stats)
            # miou_l1 = clustering_methods.calculate_miou(
            #     pred_mask=mask_pred,
            #     gt_mask=img_data["gt_mask_l1"],
            # )
            # miou_l2 = clustering_methods.calculate_miou(
            #     pred_mask=mask_pred,
            #     gt_mask=img_data["gt_mask_l2"],
            # )
            if random.randint(1, 10) == 2:
                visualize.visualize_debug_plot(
                    img_data,
                    all_stats,
                    best_mask_pred,
                    best_mask_crf,
                    parameters,
                    save_dir="results",
                )
            print(f"{i}({counter})")
            print("-" * 30)
            i += 1
        # visualize.visualize_segmentation(
        #     count_img=n,
        #     parameters=list_parameters,
        #     orig_img=img_origs,
        #     gt_mask=None,
        #     masks_without_crf=masks_without_crf,
        #     pred_mask=mask_preds,
        #     count_classes=count_classes,
        #     best_k_dbcvs=best_k_dbcvs,
        #     best_s=best_ss,
        #     best_dbcvs_for_sil=best_dbcvs_for_sil,
        #     best_dbcvs=best_dbcvs,
        #     img_id=None,
        #     save_dir="result_img_dinov2l_4",
        # )
        print("=" * 50)
        counter += 1
        print(f"{counter}/18")
    print(
        f"Time: {(time.time() - start_time) / 60:.1f} минут  model_time = {(sum(time_model_list) / len(time_model_list)):.4f}"
    )
    df = pd.DataFrame(result_stats)
    output_path = "dinov3L_diffcut_experiment_city100_results.csv"
    if not os.path.isfile(output_path):
        df.to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, mode="a", header=False, index=False)


def fust_result_SAM(
    model, device, parameters, img_dir="single", patch_size=16
):
    device = torch.device("cuda")
    dataset = datasets.SimpleDataset(
        img_dir=img_dir, patch_size=patch_size, size_img=1
    )
    sam2_model = build_sam2(
        "sam2_hiera_s.yaml", "checkpoints/sam2_hiera_small.pt", device=device
    )
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=128,
        points_per_batch=128,  # Ускорит обработку на 5090
        pred_iou_thresh=0.5,  # Снижаем порог: берем даже "слабые" маски
        stability_score_thresh=0.5,  # Снижаем порог стабильности
        min_mask_region_area=1,  # Разрешаем даже крошечные маски
    )

    for i in range(len(dataset)):
        img_data = dataset[i]
        sam_result = mask_generator.generate(img_data["img_orig"])
        # sam_result — это список словарей: [{'segmentation': bool_mask, 'area': 123, ...}, ...]
        # Извлекаем фичи (допустим, они имеют размер [1, 1024, H/16, W/16])
        features = models.extract_dino_features(
            model, device, img_data["img_tensor"], parameters["encoder"]
        )
        # Интерполируем фичи до исходного размера изображения (например, 1024x1024),
        # чтобы они точно совпадали с масками SAM
        h_p = img_data["new_h"] // patch_size
        w_p = img_data["new_w"] // patch_size
        dim = features.shape[-1]  # Обычно 1024 для Large

        # 2. Решейпим: [N_patches, Dim] -> [1, Dim, H_p, W_p]
        # Сначала в 4D с каналами в конце, потом переставляем каналы вперед для интерполяции
        features = features.view(1, h_p, w_p, dim).permute(0, 3, 1, 2)
        features_resized = F.interpolate(
            features,
            size=(img_data["h"], img_data["w"]),
            mode="bilinear",
            align_corners=False,
        )
        features_resized = features_resized.squeeze(0).permute(
            1, 2, 0
        )  # [1024, 1024, 1024]
        object_embeddings = []
        valid_masks = []

        for mask_data in sam_result:
            mask = torch.from_numpy(mask_data["segmentation"]).to(
                device
            )  # [1024, 1024]

            # Извлекаем фичи, которые попали под маску
            # Masking: выбираем пиксели, где mask == True
            mask_features = features_resized[mask]  # [N_pixels_in_mask, 1024]

            if mask_features.shape[0] > 0:
                # Средний вектор объекта (Average Pooling)
                mean_embedding = mask_features.mean(dim=0)
                object_embeddings.append(mean_embedding)
                valid_masks.append(mask_data)
        # Превращаем в один тензор [Num_Objects, 1024]
        object_descriptors = torch.stack(object_embeddings)

        k_opt, A = find_elbow_point(patch_features=object_descriptors)

        # 2. Кластеризуем объекты (например, через Spectral Clustering или K-Means)
        # Теперь матрица сходства будет всего лишь [200 x 200]! Это мгновенно.
        (
            best_clast,
            best_k_sil,
            best_k_dbcv,
            best_sil,
            best_dbcv,
        ) = clustering_methods.find_best_k_for_SAM(
            A, object_descriptors, k_opt, parameters["beta"]
        )
        final_semantic_mask = np.zeros(
            (img_data["h"], img_data["w"]), dtype=np.int32
        )

        for i, mask_data in enumerate(valid_masks):
            mask = mask_data["segmentation"]
            cluster_id = best_clast[i]
            final_semantic_mask[mask] = cluster_id + 1  # +1 чтобы фон был 0
        visualize.visualize_sam_clustering(
            img_orig=img_data["img_orig"],
            final_mask=final_semantic_mask,
            sam_result=sam_result,
            save_path="result_img",
        )


if __name__ == "__main__":
    parameters = {
        "size_img": 1.5,
        "threshold": 0.0,
        "gamma": 5,
        "beta": 0.5,
        "max_k": 150,
        "sxy_crf": 3,
        "compat_crf": 3,
        "encoder": "dinov3_vitl16",
        "patch_size": 16,
    }
    PATCH_SIZE = 16
    ann_file = "/app/cityscapes_coco_full.json"
    img_dir = "/app/leftImg8bit/val"
    level_1_map = {
        # "front_house": "house",
        # "door_house": "house",
        # "roof_house": "house",
        # "chimneys": "house",
        # "egg": "plate_of_food",
        # "tomato": "plate_of_food",
        # "cucumber": "plate_of_food",
        # "porridge": "plate_of_food",
        # "plates_small": "plate_of_food",
        # "legs_people_on_plane": "people_on_plane",
        # "hands_people_on plane": "people_on_plane",
        # "head_people_on_plane": "people_on_plane",
        # "body_people_on_plane": "people_on_plane",
        # "window_water": "home",
        # "airplane wing": "plane",
        # "car_body": "car",
        # "window_car": "car",
        # "car wheels": "car",
        # "body_horse": "horse",
        # "head_horse": "horse",
        # "legs_horse": "horse",
        # "mane_horse": "horse",
        # "horse_tail": "horse",
        # "hands_people": "people",
        # "head_people": "people",
        # "legs_people": "people",
        # "body_people": "people",
        # "body_sheep": "sheep",
        # "head_sheep": "sheep",
        # "legs_sheep": "sheep",
    }

    model, device = models.load_dino_model(model_name=parameters["encoder"])
    BF(
        model=model,
        device=device,
        ann_file=ann_file,
        img_dir=img_dir,
        patch_size=PATCH_SIZE,
        parameters=parameters,
        mapping_level=level_1_map,
    )
    # fust_result_SAM(
    #     model, device, parameters=parameters, img_dir=img_dir, patch_size=16
    # )

    # datasetl1 = datasets.CocoClaspDataset(
    #     img_dir=img_dir,
    #     ann_file=ann_file,
    #     patch_size=PATCH_SIZE,
    #     mapping_level=level_1_map,
    # )
