from collections.abc import Callable

import torch
import torchvision
from transformers import AutoImageProcessor

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

_TRANSFORM_224_224 = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
        torchvision.transforms.Normalize(mean=_MEAN, std=_STD),
    ]
)

_TRANSFORM_256_224 = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
        torchvision.transforms.Normalize(mean=_MEAN, std=_STD),
    ]
)


def get_transform(model_id: str) -> Callable:
    """Get the appropriate image transform for the given model_id."""
    if (
        # torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        model_id == "vit_b_16"
        # torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms(),
        # torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
        or model_id in ["resnet50|imagenet", "resnet18|imagenet"]
        # https://github.com/Trustworthy-ML-Lab/CLIP-dissect/blob/main/data_utils.py#L45, https://github.com/lkopf/cosy/blob/main/src/utils.py#L32
        or model_id in ["resnet18", "resnet50"]
        # https://arxiv.org/pdf/2007.08489
        or model_id == "resnet50|robust"
    ):
        return lambda imgs: torch.stack([_TRANSFORM_256_224(img.convert("RGB")) for img in imgs])
    elif model_id == "dinov3-vitb16":
        processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
        return lambda imgs: torch.cat(
            [processor(img.convert("RGB"), return_tensors="pt")["pixel_values"] for img in imgs]
        )
    else:
        raise ValueError(f"Unsupported model_id {model_id} for image transformation.")
