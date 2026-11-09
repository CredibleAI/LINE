import os
import urllib.request
from typing import Any

import torch
import torchvision
from transformers import AutoModel

from line.models.base import ExplainedModel

_RESNET_MODEL_IDS = (
    "resnet18",
    "resnet18|imagenet",
    "resnet50",
    "resnet50|imagenet",
    "resnet50|robust",
)
_VIT_MODEL_IDS = ("vit_b_16",)
_DINOV3_MODEL_IDS = ("dinov3-vitb16",)

_DINOV3_HF_IDS = {
    "dinov3-vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
}

_WEIGHTS = {
    "resnet18|imagenet": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
    "resnet18": None,  # Uses Places365 weights loaded separately
    "resnet50|imagenet": torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
    "resnet50": None,  # Uses Places365 weights loaded separately
    "resnet50|robust": None,  # Uses robust ResNet50 weights loaded separately
    "vit_b_16": torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1,
}

_OPTIONAL_KWARGS = {"resnet50": {"num_classes": 365}, "resnet18": {"num_classes": 365}}

_PLACES_365_MODELS = (
    "resnet50",
    "resnet18",
)
_PLACES_365_WEIGHTS_URL = (
    "http://places2.csail.mit.edu/models_places365/{model_id}_places365.pth.tar"
)

_ROBUST_MODELS = {"resnet50|robust"}
_ROBUST_RESNET50_WEIGHTS_PATH = "https://www.dropbox.com/s/knf4uimlqsi1yz8/imagenet_l2_3_0.pt?dl=1"


def _load_places_365_state_dict(model_id: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Load Places365 pretrained weights from remote URL.

    Downloads and loads the state dict for Places365-pretrained models,
    removing the 'module.' prefix added by DataParallel training.

    Args:
        model_id: Identifier of the model (e.g., 'resnet50').
        device: Device to map the loaded state dict to.

    Returns:
        State dictionary with cleaned keys suitable for loading into a model.
    """
    checkpoint = torch.hub.load_state_dict_from_url(
        _PLACES_365_WEIGHTS_URL.format(model_id=model_id),
        map_location=device,
    )
    state_dict = checkpoint["state_dict"]
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def _load_robust_resnet50_state_dict(device: torch.device) -> dict[str, torch.Tensor]:
    """Load robust ResNet50 pretrained weights from remote URL.

    Downloads and loads the state dict for the robust ResNet50 model,
    removing the 'module.' prefix added by DataParallel training.

    Args:
        device: Device to map the loaded state dict to.

    Returns:
        State dictionary with cleaned keys suitable for loading into a model.
    """
    data_dir = os.getenv("DATA_DIR", "./data")
    robust_destination = os.path.join(data_dir, "models", "robust_resnet50.pth")
    os.makedirs(os.path.dirname(robust_destination), exist_ok=True)
    if not os.path.exists(robust_destination):
        print(f"Downloading robust ResNet50 weights to {robust_destination}...")
        urllib.request.urlretrieve(_ROBUST_RESNET50_WEIGHTS_PATH, robust_destination)

    checkpoint = torch.load(robust_destination, map_location=device)
    state_dict = checkpoint["model"]
    return {
        k[len("module.model.") :]: v for k, v in state_dict.items() if k.startswith("module.model")
    }


class ResNet(ExplainedModel):
    """Explained model for ResNet architectures (ImageNet and Places365 weights).

    Supports resnet18, resnet50 (Places365), and resnet50_imagenet.
    Pools 4D CNN activations via spatial mean pooling.
    """

    def _load(self, **kwargs):
        """Load ResNet with ImageNet or Places365 weights.

        Returns:
            Loaded ResNet model on the specified device in evaluation mode.
        """
        weights = _WEIGHTS[self._model_id]
        arch_name = self._model_id.split("|")[0]

        model = torchvision.models.get_model(
            arch_name,
            weights=weights,
            **_OPTIONAL_KWARGS.get(self._model_id, dict()),
        )

        if self._model_id in _PLACES_365_MODELS:
            model.load_state_dict(_load_places_365_state_dict(self._model_id, self._device))
        elif self._model_id in _ROBUST_MODELS:
            model.load_state_dict(_load_robust_resnet50_state_dict(self._device))

        return model.to(self._device).eval()

    def _pool_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """Spatial mean pooling over H and W dimensions for CNN activations.

        Args:
            activations: Raw hook output of shape (N, C, H, W) for conv layers,
                or (N, C) for fully-connected layers.

        Returns:
            Pooled activations of shape (N, C).
        """
        if activations.dim() == 4:
            return activations.mean(dim=[2, 3])
        return activations


class ViT(ExplainedModel):
    """Explained model for Vision Transformer architectures.

    Supports vit_b_16. Extracts the CLS token from the sequence output.
    """

    def _load(self, **kwargs):
        """Load ViT with ImageNet weights.

        Returns:
            Loaded ViT model on the specified device in evaluation mode.
        """
        model = torchvision.models.get_model(
            self._model_id,
            weights=_WEIGHTS[self._model_id],
        )
        return model.to(self._device).eval()

    def _pool_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """Extract CLS token from ViT sequence output, or pass through linear outputs.

        Args:
            activations: Raw hook output of shape (N, seq_len, embed_dim) for encoder
                layers, or (N, num_classes) for the final heads.head linear layer.

        Returns:
            Activations of shape (N, embed_dim) or (N, num_classes).
        """
        if activations.dim() == 3:
            return activations[:, 0]
        return activations


class DINOv3(ExplainedModel):
    """Explained model for DINOv3 architectures loaded from HuggingFace.

    Supports dinov3-vitb16. The norm layer output has shape (N, seq_len, embed_dim)
    where index 0 is the CLS token.
    """

    def _load(self, **kwargs):
        """Load DINOv3 from HuggingFace.

        Returns:
            Loaded DINOv3 model on the specified device in evaluation mode.
        """
        hf_id = _DINOV3_HF_IDS[self._model_id]
        model = AutoModel.from_pretrained(hf_id)
        return model.to(self._device).eval()

    def _pool_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """Extract CLS token from norm layer output.

        Args:
            activations: Raw hook output of shape (N, seq_len, embed_dim).

        Returns:
            CLS token activations of shape (N, embed_dim).
        """
        return activations[:, 0]


MODELS_TO_EXPLAIN = _RESNET_MODEL_IDS + _VIT_MODEL_IDS + _DINOV3_MODEL_IDS


def get_model(
    model_id: str,
    layers: list[str],
    device: torch.device,
    model_swapping: bool,
    transforms: Any = None,
):
    if model_id in _RESNET_MODEL_IDS:
        return ResNet(model_id, layers, device, model_swapping, transforms)
    elif model_id in _VIT_MODEL_IDS:
        return ViT(model_id, layers, device, model_swapping, transforms)
    elif model_id in _DINOV3_MODEL_IDS:
        return DINOv3(model_id, layers, device, model_swapping, transforms)
    else:
        raise ValueError(
            f"Model {model_id} is not supported. Supported models: {MODELS_TO_EXPLAIN}"
        )
