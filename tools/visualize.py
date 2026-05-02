import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
import seaborn as sns

# Твои цвета...
custom_colors = [
    "#FF0000",
    "#00FF00",
    "#0000FF",
    "#FFFF00",
    "#FF00FF",
    "#00FFFF",
    "#FF8000",
    "#8000FF",
    "#0080FF",
    "#FF0080",
    "#808080",
    "#A20025",
    "#60A917",
    "#0050EF",
    "#D80073",
    "#A4C400",
    "#6A00FF",
    "#F0A30A",
    "#76608A",
    "#00ABA9",
]
my_cmap = ListedColormap(custom_colors)


def visualize_debug_plot(
    img_data, all_stats, best_mask_sil, best_mask_crf, parameters, save_dir
):
    """
    Создает диагностический график: маски + графики метрик.
    """

    # Находим все уникальные ID, которые реально есть в масках
    # all_ids = np.unique(
    #     np.hstack(
    #         [
    #             img_data["gt_mask_l1"].flatten(),
    #             img_data["gt_mask_l2"].flatten(),
    #             best_mask_sil.flatten(),
    #         ]
    #     )
    # )

    # palette = sns.color_palette("husl", len(all_ids))
    # my_cmap = ListedColormap(palette)

    # # 3. Настроили границы (bins)
    # boundaries = np.append(all_ids, all_ids[-1] + 1)
    # norm = BoundaryNorm(boundaries, ncolors=len(all_ids))

    df = pd.DataFrame(all_stats)
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

    # 1. Оригинал
    axes[0, 0].imshow(img_data["img_orig"])
    axes[0, 0].set_title("Original Image")

    # # 2. Ground Truth L2
    # axes[0, 2].imshow(img_data["gt_mask_l2"], cmap=my_cmap, norm=norm)
    # axes[0, 2].set_title("GT Level 2")

    # 3. Лучшая маска по мнению Силуэта
    axes[0, 1].imshow(best_mask_crf, cmap=my_cmap)
    axes[0, 1].set_title(f"Best by Silhouette without CRF")

    # 3. Лучшая маска по мнению Силуэта
    axes[0, 2].imshow(best_mask_sil, cmap=my_cmap)
    axes[0, 2].set_title(
        f"Best by Silhouette gamma={parameters['gamma']} (K={df.loc[df['silhouette'].idxmax(), 'k']})"
    )
    # 2. Ground Truth L1
    axes[0, 3].imshow(img_data["gt_mask_l1"], cmap=my_cmap)
    axes[0, 3].set_title("GT Level 1")

    # 4. График метрик
    ax_plot = axes[1, 0]
    ax_miou = ax_plot.twinx()
    ln1 = ax_plot.plot(df["k"], df["silhouette"], "g-", label="Silhouette")
    ln2 = ax_miou.plot(df["k"], df["miou_l1"], "r--", label="mIoU L1")
    ax_plot.set_xlabel("Number of clusters (K)")
    ax_plot.set_ylabel("Silhouette Score", color="g")
    ax_miou.set_ylabel("mIoU Score", color="r")
    # Объединяем легенду
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax_plot.legend(lns, labs, loc=0)
    ax_plot.set_title("Pseudo-metric vs Real mIoU L1")

    # 4. График метрик
    ax_plot = axes[1, 1]
    ax_miou = ax_plot.twinx()
    ln1 = ax_plot.plot(df["k"], df["dbcv"], "b-", label="dbcv")
    ln2 = ax_miou.plot(df["k"], df["miou_l1"], "r--", label="mIoU L1")
    ax_plot.set_xlabel("Number of clusters (K)")
    ax_plot.set_ylabel("dbcv Score", color="b")
    ax_miou.set_ylabel("mIoU Score", color="r")
    # Объединяем легенду
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax_plot.legend(lns, labs, loc=0)
    ax_plot.set_title("Pseudo-metric vs Real mIoU L1")

    # 4. График метрик
    ax_plot = axes[1, 2]
    ax_miou = ax_plot.twinx()
    ln1 = ax_plot.plot(df["k"], df["dbi"], "b-", label="dbi")
    ln2 = ax_miou.plot(df["k"], df["miou_l1"], "r--", label="mIoU L1")
    ax_plot.set_xlabel("Number of clusters (K)")
    ax_plot.set_ylabel("dbi Score", color="#000000")
    ax_miou.set_ylabel("mIoU Score", color="r")
    # Объединяем легенду
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax_plot.legend(lns, labs, loc=0)
    ax_plot.set_title("Pseudo-metric vs Real mIoU L1")

    # # 4. График метрик
    # ax_plot = axes[1, 2]
    # ax_miou = ax_plot.twinx()
    # ln1 = ax_plot.plot(df["k"], df["silhouette"], "g-", label="Silhouette")
    # ln2 = ax_miou.plot(df["k"], df["miou_l2"], "r--", label="mIoU L2")
    # ax_plot.set_xlabel("Number of clusters (K)")
    # ax_plot.set_ylabel("Silhouette Score", color="g")
    # ax_miou.set_ylabel("mIoU Score", color="r")
    # # Объединяем легенду
    # lns = ln1 + ln2
    # labs = [l.get_label() for l in lns]
    # ax_plot.legend(lns, labs, loc=0)
    # ax_plot.set_title("Pseudo-metric vs Real mIoU L2")

    # # 4. График метрик
    # ax_plot = axes[1, 3]
    # ax_miou = ax_plot.twinx()
    # ln1 = ax_plot.plot(df["k"], df["dbcv"], "b-", label="dbcv")
    # ln2 = ax_miou.plot(df["k"], df["miou_l2"], "r--", label="mIoU L2")
    # ax_plot.set_xlabel("Number of clusters (K)")
    # ax_plot.set_ylabel("dbcv Score", color="b")
    # ax_miou.set_ylabel("mIoU Score", color="r")
    # # Объединяем легенду
    # lns = ln1 + ln2
    # labs = [l.get_label() for l in lns]
    # ax_plot.legend(lns, labs, loc=0)
    # ax_plot.set_title("Pseudo-metric vs Real mIoU L2")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/dinov3L_diffcut{img_data['img_id']}.png")
    plt.close()


def visualize_segmentation_search_best(
    count_img,
    parameters,
    orig_img,
    gt_mask,
    masks_without_crf,
    pred_mask,
    count_classes,
    best_k_dbcvs,
    best_s,
    best_dbcvs_for_sil,
    best_dbcvs,
    img_id,
    save_dir,
):
    # Сетка 3x5 = 15 ячеек (1 оригинал + 7 пар масок)
    rows = 3
    cols = 5
    fig = plt.figure(figsize=(25, 15))

    plt.subplot(rows, cols, 1)
    plt.imshow(orig_img[0])
    plt.title("ORIGINAL IMAGE", fontsize=12, fontweight="bold", pad=15)
    plt.axis("off")

    max_params = 7  # min(len(parameters), 7)

    for idx in range(max_params):
        no_crf_pos = idx * 2 + 2
        with_crf_pos = idx * 2 + 3

        p = parameters[idx]
        info_text = (
            f"G={p['gamma']} | K={count_classes[idx]}\n"
            f"Sil={best_s[idx]:.2f} | DBCV={best_dbcvs[idx]:.2f}(K={best_k_dbcvs[idx]})"
        )

        # Отрисовка БЕЗ CRF
        plt.subplot(rows, cols, no_crf_pos)
        plt.imshow(orig_img[idx])
        if masks_without_crf is not None:
            plt.imshow(masks_without_crf[idx], alpha=0.95, cmap=my_cmap)
        plt.title(f"NO CRF\n{info_text}", fontsize=9)
        plt.axis("off")

        # Отрисовка С CRF
        plt.subplot(rows, cols, with_crf_pos)
        plt.imshow(orig_img[idx])
        plt.imshow(pred_mask[idx], alpha=0.95, cmap=my_cmap)
        plt.title(
            f"WITH CRF\n{info_text}",
            fontsize=9,
            color="blue",
            fontweight="bold",
        )
        plt.axis("off")

    plt.suptitle(
        f"MODEL: {parameters[0]['encoder']} | SIZE: {parameters[0]['size_img']} | BETA: {parameters[0]['beta']:.2f}",
        fontsize=18,
        y=0.98,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(
        save_dir,
        f"{parameters[0]['encoder']}_size{parameters[0]['size_img']}_beta{parameters[0]['beta']:.2f}_gamma{parameters[0]['gamma']}.png",
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved comparison to {save_path}")


def visualize_sam_clustering(img_orig, final_mask, sam_result, save_path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # Твои цвета из visualize.py
    cmap = ListedColormap(custom_colors)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # 1. Оригинал
    axes[0].imshow(img_orig)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 2. Только сетка SAM (границы объектов)
    axes[1].imshow(img_orig)
    for mask in sam_result:
        m = mask["segmentation"]
        # Рисуем контуры (опционально)
        axes[1].contour(m, levels=[0.5], colors="white", linewidths=0.5)
    axes[1].set_title(f"SAM Masks ({len(sam_result)} objects)")
    axes[1].axis("off")

    # 3. Финальная семантика
    # Маскируем 0 (фон), чтобы он был прозрачным или черным
    masked_final = np.ma.masked_where(final_mask == 0, final_mask)
    axes[2].imshow(img_orig)
    axes[2].imshow(masked_final, cmap=cmap, alpha=0.8, interpolation="nearest")
    axes[2].set_title("Object-Based Semantic Map")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    # plt.show()
