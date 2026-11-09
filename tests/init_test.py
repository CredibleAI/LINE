import os
import subprocess
import sys
from pathlib import Path

import pytest

import line
from line.config import RunConfig
from line.data.imagenet_classes import IMAGENET2012_CLASSES
from line.line import LINE
from line.utils import ORIGINAL_PROMPT_DIR

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def test_importing_history_does_not_load_model_libraries():
    """Result readers must stay importable without torch, transformers and diffusers."""
    code = (
        "import sys\n"
        "import line.pipeline.history\n"
        "loaded = [m for m in ('torch', 'transformers', 'diffusers') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    env = {**os.environ, "PYTHONPATH": str(_SRC_DIR)}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_public_names_resolve_to_their_modules():
    assert line.LINE is LINE
    assert line.RunConfig is RunConfig
    assert line.IMAGENET2012_CLASSES is IMAGENET2012_CLASSES
    assert line.ORIGINAL_PROMPT_DIR == ORIGINAL_PROMPT_DIR


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="does_not_exist"):
        _ = line.does_not_exist
