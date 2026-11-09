"""Tests for line.utils — set_random_seed and ORIGINAL_PROMPT_DIR."""

import random
from pathlib import Path

import numpy as np
import torch

from line.utils import ORIGINAL_PROMPT_DIR, set_random_seed


def test_original_prompt_dir_contains_prompt_main():
    """prompt_main.txt must exist inside ORIGINAL_PROMPT_DIR."""
    assert Path(ORIGINAL_PROMPT_DIR, "prompt_main.txt").is_file()


def test_original_prompt_dir_contains_prompt_summary():
    """prompt_summary.txt must exist inside ORIGINAL_PROMPT_DIR."""
    assert Path(ORIGINAL_PROMPT_DIR, "prompt_summary.txt").is_file()


def test_set_random_seed_seeds_torch_reproducibly():
    """The same seed must produce the same torch.rand() sequence."""
    set_random_seed(42)
    val_a = torch.rand(1).item()
    set_random_seed(42)
    val_b = torch.rand(1).item()
    assert val_a == val_b


def test_set_random_seed_seeds_numpy_reproducibly():
    """The same seed must produce the same np.random.rand() sequence."""
    set_random_seed(42)
    val_a = np.random.rand()
    set_random_seed(42)
    val_b = np.random.rand()
    assert val_a == val_b


def test_set_random_seed_seeds_random_module_with_given_seed():
    """Regression guard: the random module must be seeded with the given seed value.

    Previously ``random.seed(0)`` was hardcoded, so every call used seed 0
    regardless of the argument. The bug is now fixed; this test guards against
    regression.
    """
    set_random_seed(1)
    val_seed_1 = random.random()

    set_random_seed(2)
    val_seed_2 = random.random()

    assert val_seed_1 != val_seed_2


def test_set_random_seed_makes_random_choices_reproducible():
    """The same seed must produce identical random.choices() sequences."""
    set_random_seed(42)
    choices_a = [random.choice("abcdefghij") for _ in range(20)]
    set_random_seed(42)
    choices_b = [random.choice("abcdefghij") for _ in range(20)]
    assert choices_a == choices_b


def test_set_random_seed_different_seeds_produce_different_torch_values():
    """Different seeds must yield different torch random draws (reproducibility sanity)."""
    set_random_seed(0)
    val_0 = torch.rand(1).item()
    set_random_seed(99)
    val_99 = torch.rand(1).item()
    assert val_0 != val_99
