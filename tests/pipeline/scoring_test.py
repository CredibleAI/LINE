import math

import pytest
import torch

from line.pipeline import scoring


def test_calculate_metric_avg_activation():
    activations = torch.tensor([0.6, 0.8, 0.7])
    result = scoring.calculate_metric(activations, scoring.Metric.AVG_ACTIVATION)
    assert math.isclose(result, 0.7, rel_tol=1e-6)


def test_calculate_metric_raises_on_2d_input():
    activations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        scoring.calculate_metric(activations, scoring.Metric.AVG_ACTIVATION)


def test_calculate_metric_single_element():
    """A single-element tensor returns that element as the average."""
    activations = torch.tensor([0.75])
    result = scoring.calculate_metric(activations, scoring.Metric.AVG_ACTIVATION)
    assert math.isclose(result, 0.75)


def test_calculate_metric_negative_values():
    """Negative activations are averaged correctly (ReLU-free layers can produce negatives)."""
    activations = torch.tensor([-1.0, -2.0, -3.0])
    result = scoring.calculate_metric(activations, scoring.Metric.AVG_ACTIVATION)
    assert math.isclose(result, -2.0)


def test_calculate_metric_empty_tensor_returns_nan():
    """An empty 1-D tensor produces nan from mean(), not a crash.

    torch.mean() on an empty tensor returns nan with a UserWarning.
    This is the current (silent) behaviour — callers that treat nan as a valid
    score could silently rank the concept incorrectly.
    """
    import math as _math

    activations = torch.tensor([], dtype=torch.float32)
    result = scoring.calculate_metric(activations, scoring.Metric.AVG_ACTIVATION)
    assert _math.isnan(result)
