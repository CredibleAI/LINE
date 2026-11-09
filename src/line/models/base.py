"""Abstract base classes for model management.

This module provides BaseModel for device/GPU management and ExplainedModel
as an abstract schema for models that capture intermediate layer activations.
"""

import abc
import logging
import warnings
from collections.abc import Callable
from typing import Any

import torch
from tqdm import tqdm

from line.models.utils import gpu_inference_wrapper


class BaseModel(abc.ABC):
    """Abstract base class for managing machine learning models.

    Provides common functionality for loading, device management, and
    model swapping between CPU and GPU for efficient memory usage.
    """

    def __init__(
        self, model_id: str, device: torch.device, model_swapping: bool, **load_kwargs
    ) -> None:
        """Initialize the model.

        Args:
            model_id: Identifier or path for the model to load.
            device: Initial device to load model on ('cuda' or 'cpu').
            model_swapping: Whether to enable GPU/CPU swapping during inference.
            **load_kwargs: Additional keyword arguments passed to _load().
        """
        if model_swapping and not torch.cuda.is_available():
            raise RuntimeError(
                "model_swapping=True requires a CUDA-capable GPU, but none was found. "
                "Either disable model_swapping or run on a machine with a GPU."
            )
        self._model_id = model_id
        self._device = device
        self._model_swapping = model_swapping
        self._model = self._load(**load_kwargs)

    @abc.abstractmethod
    def _load(self, **kwargs) -> Any:
        """Load model resources from disk.

        Subclasses must implement this method to load and initialize the model
        with appropriate pretrained weights or architecture.

        Args:
            **kwargs: Model-specific loading arguments.

        Returns:
            Loaded model object.
        """
        pass

    def _send_to_device(self, device: torch.device) -> None:
        """Move model to specified device with warning suppression.

        Transfers the model to the specified device (GPU or CPU). CPU inference
        warnings are suppressed since the model is not meant to perform inference
        on CPU.

        Args:
            device: Target device ('cuda' for GPU or 'cpu' for CPU).
        """
        logging.info(f"Sending model {self._model_id} to device {device}.")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*use another device for inference.*")
            self._model.to(device)
        self._device = device

    @property
    def model_id(self):
        """Get the model identifier.

        Returns:
            The model ID used for initialization.
        """
        return self._model_id


class ExplainedModel(BaseModel, abc.ABC):
    """Abstract base for models that capture intermediate layer activations.

    Extends BaseModel with forward hook registration and a shared
    get_activations interface. Subclasses implement _load() for their
    architecture and _pool_activations() to reduce raw hook output to
    shape (N, num_neurons).
    """

    def __init__(
        self,
        model_id: str,
        layers: list[str],
        device: torch.device,
        model_swapping: bool,
        transforms: Callable[..., torch.Tensor] | None = None,
        **load_kwargs,
    ) -> None:
        """Initialize the explained model.

        Args:
            model_id: Identifier of the model architecture.
            layers: Names of the layers to capture activations from.
            device: Device to load model on ('cuda' or 'cpu').
            model_swapping: Whether to enable model swapping functionality.
            transforms: Optional transforms to apply to the input.

        Raises:
            ValueError: If any specified layer is not found in the model.
        """
        super().__init__(model_id, device, model_swapping, **load_kwargs)
        self._transforms = transforms
        self._activations: dict[str, torch.Tensor] = {}
        self._hooked_layers: list[str] = list(layers)
        self._register_forward_hooks(layers)

    @abc.abstractmethod
    def _pool_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """Reduce raw hook output to shape (N, num_neurons).

        Args:
            activations: Raw tensor captured by the forward hook.

        Returns:
            Pooled activations of shape (N, num_neurons).
        """
        pass

    @property
    def hooked_layers(self) -> list[str]:
        """Return the layer names that have forward hooks registered."""
        return list(self._hooked_layers)

    @property
    def get_model_layers(self) -> list[str]:
        """Get the names of the layers in the model.

        Returns:
            List of layer names corresponding to all named modules at any depth.
        """
        return [name for (name, _) in self._model.named_modules() if name]

    def _hook(self, layer_name: str, module, input, output) -> None:
        """Store layer output activations during forward pass.

        Args:
            layer_name: Name of the layer that triggered the hook.
            module: The module that triggered the hook.
            input: The input to the module.
            output: The output of the module.
        """
        self._activations[layer_name] = output

    def _register_forward_hooks(self, layers: list[str]) -> None:
        """Register forward hooks on the specified layers.

        Args:
            layers: Names of the layers to register hooks on.

        Raises:
            ValueError: If any layer name is not found in the model.
        """
        named_modules_dict = {
            name: module for (name, module) in self._model.named_modules() if name
        }
        for layer in layers:
            if layer not in named_modules_dict:
                raise ValueError(f"Layer {layer} not found in the model.")
            named_modules_dict[layer].register_forward_hook(
                lambda module, input, output, layer_name=layer: self._hook(
                    layer_name, module, input, output
                )
            )

    @gpu_inference_wrapper
    def get_activations(self, input_batch: Any, batch_size: int = 0) -> dict[str, torch.Tensor]:
        """Forward pass through model and return per-layer activations.

        When transforms are set, input_batch can be any type accepted by the
        transform callable (e.g. a list of PIL images). Without transforms,
        input_batch must be a tensor of shape (N, C, H, W).

        Args:
            input_batch: Input accepted by the configured transforms, or a
                pre-transformed tensor of shape (N, C, H, W) when no transforms
                are set.
            batch_size: Number of samples per forward pass. 0 means process all
                at once.

        Returns:
            Mapping from layer name to pooled activations of shape (N, num_neurons).
        """
        if self._transforms is not None:
            input_batch = self._transforms(input_batch)

        self._activations.clear()
        n = len(input_batch)
        effective_batch_size = batch_size if batch_size > 0 else n

        if effective_batch_size >= n:
            with torch.no_grad():
                _ = self._model(input_batch.float().to(self._device))
            return {
                layer: self._pool_activations(activation)
                for layer, activation in self._activations.items()
            }

        layer_results: dict[str, list[torch.Tensor]] = {layer: [] for layer in self._hooked_layers}
        n_batches = (n + effective_batch_size - 1) // effective_batch_size
        for i in tqdm(
            range(0, n, effective_batch_size), total=n_batches, desc="Activation batches"
        ):
            chunk = input_batch[i : i + effective_batch_size]
            with torch.no_grad():
                _ = self._model(chunk.float().to(self._device))
            for layer in self._hooked_layers:
                layer_results[layer].append(self._pool_activations(self._activations[layer]))
        return {layer: torch.cat(tensors, dim=0) for layer, tensors in layer_results.items()}
