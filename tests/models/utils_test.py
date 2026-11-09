"""Tests for line.models.utils — gpu_inference_wrapper and convert_input."""

from unittest.mock import MagicMock, call

import pytest
import torch

from line.models.utils import gpu_inference_wrapper

# ---------------------------------------------------------------------------
# Minimal stub to exercise gpu_inference_wrapper
# ---------------------------------------------------------------------------


class _FakeModel:
    """Minimal stub providing the interface expected by gpu_inference_wrapper."""

    def __init__(self, model_swapping: bool) -> None:
        self._model_swapping = model_swapping
        self._send_to_device = MagicMock()

    @gpu_inference_wrapper
    def infer(self, value: int) -> int:
        return value * 2


# ---------------------------------------------------------------------------
# gpu_inference_wrapper tests
# ---------------------------------------------------------------------------


def test_wrapper_returns_inference_result() -> None:
    model = _FakeModel(model_swapping=False)
    assert model.infer(3) == 6


def test_no_swapping_skips_device_transfer() -> None:
    model = _FakeModel(model_swapping=False)
    model.infer(1)
    model._send_to_device.assert_not_called()


def test_swapping_sends_to_cuda_before_and_cpu_after() -> None:
    model = _FakeModel(model_swapping=True)
    model.infer(1)
    assert model._send_to_device.call_args_list == [
        call(torch.device("cuda")),
        call(torch.device("cpu")),
    ]


def test_swapping_cuda_called_before_inference() -> None:
    """Verify that the cuda send happens before the inference function executes."""
    log = []
    model = _FakeModel(model_swapping=True)
    model._send_to_device = MagicMock(side_effect=lambda d: log.append(("send", d)))

    @gpu_inference_wrapper
    def _infer(self):
        log.append(("infer",))
        return "done"

    _infer(model)
    assert log == [("send", torch.device("cuda")), ("infer",), ("send", torch.device("cpu"))]


def test_swapping_cpu_called_even_on_exception() -> None:
    """cpu transfer is guaranteed via try/finally even when inference raises."""
    model = _FakeModel(model_swapping=True)

    @gpu_inference_wrapper
    def _bad_infer(self):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _bad_infer(model)

    assert model._send_to_device.call_args_list == [
        call(torch.device("cuda")),
        call(torch.device("cpu")),
    ]


def test_wrapper_passes_args_and_kwargs() -> None:
    class _M:
        _model_swapping = False
        _send_to_device = MagicMock()

        @gpu_inference_wrapper
        def infer(self, a, b, *, c=0):
            return a + b + c

    assert _M().infer(1, 2, c=10) == 13
