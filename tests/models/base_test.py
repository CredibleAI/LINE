"""Tests for line.models.base — BaseModel and ExplainedModel."""

import pytest
import pytest_mock
import torch

from line.models.base import ExplainedModel

# ---------------------------------------------------------------------------
# Minimal concrete implementations used across tests
# ---------------------------------------------------------------------------


class _MockCNNTorch(torch.nn.Module):
    """Two-layer CNN: conv (hook target) + avgpool."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=2, stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv.weight = torch.nn.Parameter(torch.ones((2, 3, 2, 2)), requires_grad=False)
        self.conv.bias = torch.nn.Parameter(torch.zeros(2), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avgpool(self.conv(x))


class _MockViTTorch(torch.nn.Module):
    """Identity model that returns a 3D sequence tensor (N, seq_len, embed_dim)."""

    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.encoder = torch.nn.Identity()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        seq = torch.ones(n, 5, self.embed_dim)  # (N, seq_len, embed_dim)
        return self.encoder(seq)  # pass through so the forward hook fires


class _CNNExplainedModel(ExplainedModel):
    def _load(self, **kwargs):
        return _MockCNNTorch().eval()

    def _pool_activations(self, activations: torch.Tensor) -> torch.Tensor:
        return activations.mean(dim=[2, 3])


class _ViTExplainedModel(ExplainedModel):
    def _load(self, **kwargs):
        return _MockViTTorch().eval()

    def _pool_activations(self, activations: torch.Tensor) -> torch.Tensor:
        return activations[:, 0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEVICE = torch.device("cpu")


def _make_cnn_model(mocker: pytest_mock.MockerFixture) -> _CNNExplainedModel:
    mocker.patch("line.models.base.BaseModel._send_to_device")
    return _CNNExplainedModel("resnet-test", ["conv"], _DEVICE, model_swapping=False)


def _make_vit_model(mocker: pytest_mock.MockerFixture) -> _ViTExplainedModel:
    mocker.patch("line.models.base.BaseModel._send_to_device")
    return _ViTExplainedModel("vit-test", ["encoder"], _DEVICE, model_swapping=False)


# ---------------------------------------------------------------------------
# BaseModel tests
# ---------------------------------------------------------------------------


def test_model_id_property(mocker: pytest_mock.MockerFixture) -> None:
    mocker.patch("line.models.base.BaseModel._send_to_device")
    model = _CNNExplainedModel("my-model-id", ["conv"], _DEVICE, model_swapping=False)
    assert model.model_id == "my-model-id"


def test_send_to_device_calls_model_to(mocker: pytest_mock.MockerFixture) -> None:
    mocker.patch("line.models.base.BaseModel._send_to_device")
    model = _CNNExplainedModel("m", ["conv"], _DEVICE, model_swapping=False)

    fake_inner = mocker.MagicMock()
    model._model = fake_inner

    # Restore real _send_to_device so we can test it
    mocker.stopall()
    model._send_to_device(torch.device("cpu"))

    fake_inner.to.assert_called_once_with(torch.device("cpu"))


def test_send_to_device_updates_device_attribute(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """_send_to_device keeps self._device in sync with the model's actual location."""
    model = _CNNExplainedModel("m", ["conv"], _DEVICE, model_swapping=False)
    model._model = mocker.MagicMock()  # prevent real tensor moves

    model._send_to_device(torch.device("cuda"))
    assert model._device == torch.device("cuda")

    model._send_to_device(torch.device("cpu"))
    assert model._device == torch.device("cpu")


# ---------------------------------------------------------------------------
# ExplainedModel — hook registration
# ---------------------------------------------------------------------------


def test_invalid_layer_raises(mocker: pytest_mock.MockerFixture) -> None:
    mocker.patch("line.models.base.BaseModel._send_to_device")
    with pytest.raises(ValueError, match="Layer nonexistent not found"):
        _CNNExplainedModel("m", ["nonexistent"], _DEVICE, model_swapping=False)


def test_valid_layer_does_not_raise(mocker: pytest_mock.MockerFixture) -> None:
    mocker.patch("line.models.base.BaseModel._send_to_device")
    _CNNExplainedModel("m", ["conv"], _DEVICE, model_swapping=False)  # no exception


# ---------------------------------------------------------------------------
# ExplainedModel — get_activations
# ---------------------------------------------------------------------------


def test_get_activations_cnn_shape(mocker: pytest_mock.MockerFixture) -> None:
    """CNN activations are pooled from (N, C, H, W) → (N, C)."""

    model = _make_cnn_model(mocker)

    n_images, n_channels = 3, 3
    input_batch = torch.ones(n_images, n_channels, 4, 4)
    activations = model.get_activations(input_batch)

    assert set(activations.keys()) == {"conv"}
    assert activations["conv"].shape == (n_images, 2)  # 2 output channels from MockCNNTorch


def test_get_activations_cnn_values(mocker: pytest_mock.MockerFixture) -> None:
    """Activations match expected conv output: sum of ones kernel over input."""

    model = _make_cnn_model(mocker)

    n_images, n_channels = 2, 3
    input_batch = torch.ones(n_images, n_channels, 4, 4)
    activations = model.get_activations(input_batch)

    # conv weight=1, kernel 2x2, 3 input channels → each output = 3*2*2 = 12
    expected = 4.0 * n_channels * torch.ones(n_images, 2)
    assert torch.allclose(activations["conv"], expected)


def test_get_activations_cnn_linearity(mocker: pytest_mock.MockerFixture) -> None:
    """Doubling input doubles activations (conv is linear)."""

    model = _make_cnn_model(mocker)

    input_batch = torch.ones(2, 3, 4, 4)
    act1 = model.get_activations(input_batch)
    act2 = model.get_activations(2 * input_batch)

    assert torch.allclose(act2["conv"], 2 * act1["conv"])


def test_get_activations_vit_shape(mocker: pytest_mock.MockerFixture) -> None:
    """ViT activations extract CLS token: (N, seq_len, embed_dim) → (N, embed_dim)."""

    model = _make_vit_model(mocker)

    input_batch = torch.ones(3, 3, 4, 4)
    activations = model.get_activations(input_batch)

    assert set(activations.keys()) == {"encoder"}
    assert activations["encoder"].shape == (3, 4)  # embed_dim=4 from _MockViTTorch


def test_get_activations_multiple_layers(mocker: pytest_mock.MockerFixture) -> None:
    """Multiple layers each produce their own entry in the returned dict."""

    mocker.patch("line.models.base.BaseModel._send_to_device")
    model = _CNNExplainedModel("m", ["conv", "avgpool"], _DEVICE, model_swapping=False)

    activations = model.get_activations(torch.ones(2, 3, 4, 4))

    assert set(activations.keys()) == {"conv", "avgpool"}
    assert activations["conv"].shape == (2, 2)  # pooled CNN: (N, C)
    assert activations["avgpool"].shape == (2, 2)  # pooled CNN: (N, C)


# ---------------------------------------------------------------------------
# ExplainedModel — model swapping
# ---------------------------------------------------------------------------


def test_model_swapping_sends_to_cuda_then_cpu(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """With model_swapping=True, model is sent to cuda before and cpu after inference."""

    mocker.patch("line.models.base.torch.cuda.is_available", return_value=True)
    send_spy = mocker.patch("line.models.base.BaseModel._send_to_device")
    model = _CNNExplainedModel("m", ["conv"], _DEVICE, model_swapping=True)
    send_spy.reset_mock()  # ignore calls from __init__

    model.get_activations(torch.ones(1, 3, 4, 4))

    calls = [c.args[0] for c in send_spy.call_args_list]
    assert calls == [torch.device("cuda"), torch.device("cpu")]


def test_model_swapping_raises_without_cuda(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """model_swapping=True raises RuntimeError immediately when CUDA is unavailable."""

    mocker.patch("line.models.base.torch.cuda.is_available", return_value=False)

    with pytest.raises(RuntimeError, match="model_swapping=True requires a CUDA-capable GPU"):
        _CNNExplainedModel("m", ["conv"], _DEVICE, model_swapping=True)


def test_no_model_swapping_does_not_send_to_device(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """With model_swapping=False, _send_to_device is never called during inference."""

    send_spy = mocker.patch("line.models.base.BaseModel._send_to_device")
    model = _CNNExplainedModel("m", ["conv"], _DEVICE, model_swapping=False)
    send_spy.reset_mock()

    model.get_activations(torch.ones(1, 3, 4, 4))

    send_spy.assert_not_called()


# ---------------------------------------------------------------------------
# ExplainedModel — _activations cleared between calls
# ---------------------------------------------------------------------------


def test_get_activations_clears_stale_keys(mocker: pytest_mock.MockerFixture) -> None:
    """Stale entries injected into _activations before a call must not appear in the result."""

    model = _make_cnn_model(mocker)
    model._activations["stale_layer"] = torch.zeros(1, 4)

    result = model.get_activations(torch.ones(1, 3, 4, 4))

    assert "stale_layer" not in result


def test_get_activations_no_stale_state_between_calls(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Second call returns fresh activations, not stale values from the first call."""

    model = _make_cnn_model(mocker)

    input1 = torch.ones(2, 3, 4, 4)
    input2 = 2 * input1

    act1 = model.get_activations(input1)
    act2 = model.get_activations(input2)

    # conv is linear: doubling input doubles activations
    assert torch.allclose(act2["conv"], 2 * act1["conv"])


def test_get_activations_batched_clears_stale_keys(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Stale entries injected before a batched call must not appear in the result."""

    model = _make_cnn_model(mocker)
    model._activations["stale_layer"] = torch.zeros(1, 4)

    result = model.get_activations(torch.ones(4, 3, 4, 4), batch_size=2)

    assert "stale_layer" not in result


def test_get_activations_batch_size_zero_uses_single_forward_pass(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """batch_size=0 means all samples are processed in a single forward pass."""

    model = _make_cnn_model(mocker)
    forward_spy = mocker.patch.object(model._model, "forward", wraps=model._model.forward)

    model.get_activations(torch.ones(4, 3, 4, 4), batch_size=0)

    assert forward_spy.call_count == 1
