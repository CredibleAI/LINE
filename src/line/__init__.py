"""LINE: LLM-based Iterative Neuron Explanations for Vision Models."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from line.config import RunConfig
    from line.data.imagenet_classes import IMAGENET2012_CLASSES
    from line.line import LINE
    from line.utils import ORIGINAL_PROMPT_DIR

# Lazy so that `import line` stays cheap as line.line and line.utils import heavy load torch.
_LAZY_EXPORTS = {
    "IMAGENET2012_CLASSES": "line.data.imagenet_classes",
    "LINE": "line.line",
    "ORIGINAL_PROMPT_DIR": "line.utils",
    "RunConfig": "line.config",
}

__all__ = ["IMAGENET2012_CLASSES", "LINE", "ORIGINAL_PROMPT_DIR", "RunConfig"]


def __getattr__(name: str) -> Any:
    """Import a public name on first access (PEP 562)."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'line' has no attribute {name!r}")
    return getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
