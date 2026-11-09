import importlib.resources
import random

import numpy as np
import torch

ORIGINAL_PROMPT_DIR = str(importlib.resources.files("line").joinpath("prompts"))


def set_random_seed(seed: int = 42):
    """Configure reproducible seeds across common randomness sources.

    Args:
        seed: Integer seed value for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Commented out to avoid errors with non-deterministic operations.
    # torch.use_deterministic_algorithms(True)
