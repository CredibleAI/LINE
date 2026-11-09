"""Shared utilities for model management."""

from collections.abc import Callable
from functools import wraps
from typing import Any

import torch


def gpu_inference_wrapper(inference_func: Callable[..., Any]):
    """Decorator to manage GPU memory by swapping models between CPU and GPU.

    Sends the model to GPU before inference and returns it to CPU after,
    allowing efficient memory management. Only activates if model_swapping
    is enabled in the Model instance.

    Args:
        inference_func: The inference function to wrap.

    Returns:
        Wrapped function that manages device placement.
    """

    @wraps(inference_func)
    def wrapper(self, *args, **kwargs):
        if self._model_swapping:
            self._send_to_device(torch.device("cuda"))
            try:
                result = inference_func(self, *args, **kwargs)
            finally:
                self._send_to_device(torch.device("cpu"))
        else:
            result = inference_func(self, *args, **kwargs)

        return result

    return wrapper
