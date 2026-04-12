import torch
from transformers import AutoModel, AutoImageProcessor
import timm
from sklearn.decomposition import PCA
from typing import Tuple, Any
from torch import nn


def load_dino_model(
    model_name: str = "dinov2_vitb14",
) -> Tuple[nn.Module, torch.device]:
    """
    dinov2_vits14, dinov2_vitb14, dinov2_vitl14,
    dinov3_vits16, "dinov3_vitb16, "dinov3_vitl16

    return model, device
    """
    print(f"Загрузка модели {model_name} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timm_names = {
        "dinov2_vits14": "vit_small_patch14_dinov2",
        "dinov2_vitb14": "vit_base_patch14_dinov2",
        "dinov2_vitl14": "vit_large_patch14_dinov2",
    }
    dinov3_names = {
        "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
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
    else:
        # processor = AutoImageProcessor.from_pretrained(
        #     dinov3_names[model_name]
        # )  # , token=''
        model = AutoModel.from_pretrained(dinov3_names[model_name])
        # model = (model, processor)
    model = model.to(device)
    model.eval()
    return model, device


@torch.no_grad()
def extract_dino_features(
    model: nn.Module,
    device: torch.device,
    img_tensor: torch.Tensor,
    model_type: str,
):
    """
    return patch_features
    """
    img_tensor = img_tensor.to(device)
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)
    outputs = model(img_tensor)
    if "dinov3" in model_type:
        features = outputs.last_hidden_state
        patch_features = features[:, 5:, :]
    else:
        patch_features = outputs[:, 1:, :]
    return patch_features.squeeze(0)


# @torch.no_grad()
# def extract_dinov2_features(model, device, img_tensor):
#     """
#     model: timm ViT модель
#     n_components: до скольки измерений сжимаем (16, 32, 64)
#     use_layer: какой слой использовать (0-23 для Large).
#                Слои 18-20 обычно лучше для текстур, чем самый последний.
#     """
#     n_components = 32
#     use_layer = 20
#     if img_tensor.ndim == 3:
#         img_tensor = img_tensor.unsqueeze(0).to(device)
#     else:
#         img_tensor = img_tensor.to(device)

#     # Достаем промежуточные слои, если нужно, или просто forward
#     # В timm можно достать фичи через forward_intermediates
#     # Но для простоты возьмем стандартный выход:
#     out = model.forward_features(img_tensor)

#     # Отрезаем CLS-токен [1, N_patches+1, 1024] -> [N_patches, 1024]
#     features = out[:, 1:, :].squeeze(0)

#     # Переводим в CPU для PCA (sklearn работает на CPU)
#     features_np = features.cpu().numpy()

#     # --- ШАГ С PCA ---
#     # Мы обучаем PCA на патчах ОДНОЙ картинки.
#     # Это выделит главные отличия именно внутри этого кадра.
#     pca = PCA(n_components=n_components)
#     features_reduced = pca.fit_transform(features_np)

#     # Возвращаем обратно в тензор на GPU для дальнейших расчетов (Affinity Matrix)
#     return torch.from_numpy(features_reduced).to(device)
