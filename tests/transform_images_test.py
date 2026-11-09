"""Tests for transform_images.get_transform coverage across supported models."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

import transform_images
from explained_models import MODELS_TO_EXPLAIN
from transform_images import get_transform


class _DummyDinoProcessor:
    def __call__(self, image: Image.Image, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros((1, 3, 224, 224), dtype=torch.float32)}


@pytest.mark.parametrize("model_id", MODELS_TO_EXPLAIN)
def test_get_transform_supported_for_all_models(model_id: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        transform_images.AutoImageProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: _DummyDinoProcessor(),
    )

    transform = get_transform(model_id)
    assert callable(transform)

    test_images = [Image.new("L", (300, 300))]
    output = transform(test_images)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3, 224, 224)
