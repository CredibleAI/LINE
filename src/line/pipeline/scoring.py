"""Scoring metrics for concept quality evaluation.

Provides metrics for quantifying concept quality from neuron activations on
synthetic concept images. This module currently implements average
activation scoring used in the paper only.
"""

import enum

import torch


def _calculate_average_activation(neuron_activations: torch.Tensor, *args, **kwargs) -> float:
    """Calculate average neuron activation.

    Args:
        neuron_activations: Activation values for a neuron.

    Returns:
        Mean activation value.
    """
    return neuron_activations.mean().item()


class Metric(enum.Enum):
    """Enumeration of concept scoring metrics.

    Each member wraps a MetricFn: Callable[[torch.Tensor], float].
    Adding a new metric requires defining a function with that signature
    and registering it as a member here.

    Attributes:
        AVG_ACTIVATION: Average activation magnitude on synthetic images.
    """

    AVG_ACTIVATION = enum.member(_calculate_average_activation)

    def compute(self, activations: torch.Tensor, *args, **kwargs) -> float:
        """Apply this metric to a tensor of neuron activations.

        Args:
            activations: 1-D neuron activations from synthetic concept images.

        Returns:
            Scalar score.
        """
        fn = self.value  # type: ignore[assignment]
        return fn(activations, *args, **kwargs)


def calculate_metric(activations: torch.Tensor, metric: Metric) -> float:
    """Calculate the specified metric for a concept.

    Args:
        activations: Neuron activations from synthetic concept images.
        metric: The metric to compute (AVG_ACTIVATION).

    Returns:
        Score value for the selected metric.

    Raises:
        ValueError: If activations are not one-dimensional.
    """
    if activations.ndim != 1:
        raise ValueError("Neuron activations must be one dimensional.")

    return metric.compute(activations)
