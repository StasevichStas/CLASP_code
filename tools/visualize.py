import os
import matplotlib.pyplot as plt


def visualize_segmentation(
    count_img,
    parameters,
    orig_img,
    gt_mask,
    masks_without_crf,
    pred_mask,
    num_clusters,
    best_k_dbcvs,
    best_s,
    best_dbcvs,
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
        plt.imshow(masks_without_crf[i], alpha=0.85, cmap="tab20")
        plt.title(f"CLASP Prediction without CRF(K={num_clusters[i]}) ")
        plt.axis("off")

        plt.subplot(count_img, 3, 3 * i + 3)
        plt.imshow(orig_img[i])
        plt.imshow(pred_mask[i], alpha=0.85, cmap="tab20")
        plt.title(
            f"CLASP Prediction (K={num_clusters[i]}) sil_score={best_s[i]:.3f} dbcv(K={best_k_dbcvs[i]}) = {best_dbcvs[i]:.3f}"
        )
        plt.axis("off")
    plt.suptitle(f"{param}", y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{param}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=350)
    plt.close(fig)
