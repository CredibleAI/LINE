"""Tests for line.models.t2i — T2I."""

from typing import Annotated
from unittest.mock import MagicMock

import pytest
import pytest_mock
import torch

from line.models.t2i import T2I

T2IFixture = Annotated[T2I, pytest.fixture]

_FAKE_IMAGE = None
_NUM_INFERENCE_STEPS = None
_GUIDANCE_SCALE = None
_DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def t2i_model(mocker: pytest_mock.MockerFixture) -> T2I:
    mocker.patch("line.models.t2i.T2I._load")
    mocker.patch("line.models.base.BaseModel._send_to_device")
    return T2I(
        "test-model-id",
        _DEVICE,
        model_swapping=False,
        num_inference_steps=_NUM_INFERENCE_STEPS,
        guidance_scale=_GUIDANCE_SCALE,
    )


def _make_fake_pipeline(mocker: pytest_mock.MockerFixture, model: T2I):
    """Patch model._model to return one fake image per prompt in each call."""

    def fake_call(**kwargs):
        return type("", (object,), {"images": [_FAKE_IMAGE] * len(kwargs["prompt"])})()

    return mocker.patch.object(model, "_model", side_effect=fake_call)


# ---------------------------------------------------------------------------
# generate_images
# ---------------------------------------------------------------------------


def test_generate_images_total_count(
    t2i_model: T2IFixture, mocker: pytest_mock.MockerFixture
) -> None:
    prompts = [f"prompt_{i}" for i in range(3)]
    generators = [torch.Generator(_DEVICE).manual_seed(i) for i in range(len(prompts))]
    _make_fake_pipeline(mocker, t2i_model)

    result = t2i_model.generate_images(prompts, generators)

    assert len(result) == len(prompts)


def test_generate_images_generators_passed_to_pipeline(
    t2i_model: T2IFixture, mocker: pytest_mock.MockerFixture
) -> None:
    """Generators are forwarded to the pipeline unchanged."""
    prompts = ["prompt_a", "prompt_b"]
    generators = [torch.Generator(_DEVICE).manual_seed(i) for i in range(len(prompts))]
    fake_pipeline = _make_fake_pipeline(mocker, t2i_model)

    t2i_model._batch_size = 0
    t2i_model.generate_images(prompts, generators)

    call_kwargs = fake_pipeline.call_args
    assert call_kwargs.kwargs["prompt"] == prompts
    assert call_kwargs.kwargs["generator"] == generators


def test_generate_images_custom_generators_used(
    t2i_model: T2IFixture, mocker: pytest_mock.MockerFixture
) -> None:
    custom_gens = [MagicMock(spec=torch.Generator), MagicMock(spec=torch.Generator)]
    fake_pipeline = _make_fake_pipeline(mocker, t2i_model)

    t2i_model.generate_images(["p0", "p1"], custom_gens)

    passed_generators = fake_pipeline.call_args.kwargs["generator"]
    assert passed_generators == custom_gens


def test_generate_images_batches_correctly(
    t2i_model: T2IFixture, mocker: pytest_mock.MockerFixture
) -> None:
    """With 4 total prompts and batch_size=3, pipeline is called twice."""
    n_total = 4  # calls: [3, 1]
    t2i_model._batch_size = 3
    prompts = [f"p{i}" for i in range(n_total)]
    generators = [torch.Generator(_DEVICE).manual_seed(i) for i in range(n_total)]
    fake_pipeline = _make_fake_pipeline(mocker, t2i_model)

    t2i_model.generate_images(prompts, generators)

    assert fake_pipeline.call_count == 2


def test_generate_images_batch_size_zero_uses_single_call(
    t2i_model: T2IFixture, mocker: pytest_mock.MockerFixture
) -> None:
    """batch_size=0 means all images are processed in a single pipeline call."""
    t2i_model._batch_size = 0
    prompts = ["p1", "p2", "p3"]
    generators = [torch.Generator(_DEVICE).manual_seed(i) for i in range(len(prompts))]
    fake_pipeline = _make_fake_pipeline(mocker, t2i_model)

    t2i_model.generate_images(prompts, generators)

    assert fake_pipeline.call_count == 1


# ---------------------------------------------------------------------------
# generate_images_from_concept
# ---------------------------------------------------------------------------


def test_generate_images_from_concept_total_count(
    t2i_model: T2IFixture, mocker: pytest_mock.MockerFixture
) -> None:
    mocker.patch(
        "line.models.t2i.concept_image_prompt", side_effect=lambda pt, c, seed=None: f"{pt}:{c}"
    )
    _make_fake_pipeline(mocker, t2i_model)

    concepts = [("A realistic photo of a", "cat"), ("A realistic photo of a", "dog")]
    result = t2i_model.generate_images_from_concept(concepts, n_images=3)

    assert len(result) == len(concepts) * 3


def test_generate_images_from_concept_builds_prompts(
    t2i_model: T2IFixture, mocker: pytest_mock.MockerFixture
) -> None:
    prompt_spy = mocker.patch(
        "line.models.t2i.concept_image_prompt", side_effect=lambda pt, c, seed=None: f"{pt}:{c}"
    )
    _make_fake_pipeline(mocker, t2i_model)

    t2i_model.generate_images_from_concept(
        [("A realistic photo of a", "cat"), ("A realistic photo of a", "dog")], n_images=1
    )

    # Default generator for copy 0 has seed 0.
    prompt_spy.assert_any_call("A realistic photo of a", "cat", seed=0)
    prompt_spy.assert_any_call("A realistic photo of a", "dog", seed=0)


def test_generate_images_from_concept_seed_pattern(
    t2i_model: T2IFixture, mocker: pytest_mock.MockerFixture
) -> None:
    """Each (image_idx, concept) pair gets a fresh generator seeded with image_idx
    so scores are independent of batch position."""
    n_images = 3
    concepts = [("pt", "cat"), ("pt", "dog")]
    n_concepts = len(concepts)
    mocker.patch("line.models.t2i.concept_image_prompt", side_effect=lambda pt, c, **_: f"{pt}:{c}")
    fake_pipeline = _make_fake_pipeline(mocker, t2i_model)

    t2i_model._batch_size = 0  # single call so we can inspect the full list
    t2i_model.generate_images_from_concept(concepts, n_images)

    passed_generators = fake_pipeline.call_args.kwargs["generator"]
    assert len(passed_generators) == n_concepts * n_images

    for copy_idx in range(n_images):
        gen_group = passed_generators[copy_idx * n_concepts : (copy_idx + 1) * n_concepts]
        assert all(g.initial_seed() == copy_idx for g in gen_group)

    assert len(set(id(g) for g in passed_generators)) == n_concepts * n_images


def test_concept_generator_seeds_independent_of_batch_size(
    t2i_model: T2IFixture, mocker: pytest_mock.MockerFixture
) -> None:
    """_batch_size only controls pipeline call chunking; the seed assigned to
    each (image_idx, concept) generator must not change with batch size."""
    concepts = [("pt", f"c{i}") for i in range(4)]
    n_images = 2
    mocker.patch("line.models.t2i.concept_image_prompt", side_effect=lambda pt, c, **_: f"{pt}:{c}")

    def collected_seeds(batch_size: int) -> list[int]:
        fake = _make_fake_pipeline(mocker, t2i_model)
        t2i_model._batch_size = batch_size
        t2i_model.generate_images_from_concept(concepts, n_images)
        return [g.initial_seed() for call in fake.call_args_list for g in call.kwargs["generator"]]

    assert collected_seeds(1) == collected_seeds(2) == collected_seeds(0)


def test_generate_images_from_concept_wrong_generator_count_raises(
    t2i_model: T2IFixture, mocker: pytest_mock.MockerFixture
) -> None:
    _make_fake_pipeline(mocker, t2i_model)
    with pytest.raises(AssertionError):
        t2i_model.generate_images_from_concept(
            [("pt", "concept")], n_images=3, generators=[MagicMock()]
        )


# ---------------------------------------------------------------------------
# Model swapping
# ---------------------------------------------------------------------------


def test_model_swapping_sends_to_cuda_then_cpu(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch("line.models.t2i.T2I._load")
    mocker.patch("line.models.base.torch.cuda.is_available", return_value=True)
    send_spy = mocker.patch("line.models.base.BaseModel._send_to_device")
    model = T2I(
        "m", torch.device("cpu"), model_swapping=True, num_inference_steps=1, guidance_scale=1
    )
    send_spy.reset_mock()

    _make_fake_pipeline(mocker, model)
    model.generate_images(["p"], [torch.Generator(torch.device("cpu")).manual_seed(0)])

    calls = [c.args[0] for c in send_spy.call_args_list]
    assert calls == [torch.device("cuda"), torch.device("cpu")]


def test_no_model_swapping_skips_device_transfer(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch("line.models.t2i.T2I._load")
    send_spy = mocker.patch("line.models.base.BaseModel._send_to_device")
    model = T2I(
        "m", torch.device("cpu"), model_swapping=False, num_inference_steps=1, guidance_scale=1
    )
    send_spy.reset_mock()

    _make_fake_pipeline(mocker, model)
    model.generate_images(["p"], [torch.Generator(_DEVICE).manual_seed(0)])

    send_spy.assert_not_called()


# ---------------------------------------------------------------------------
# Conditional num_inference_steps / guidance_scale
# ---------------------------------------------------------------------------


def test_generate_images_omits_num_inference_steps_when_none(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch("line.models.t2i.T2I._load")
    mocker.patch("line.models.base.BaseModel._send_to_device")
    model = T2I("m", _DEVICE, model_swapping=False, num_inference_steps=None, guidance_scale=None)
    fake_pipeline = _make_fake_pipeline(mocker, model)

    model.generate_images(["p"], [torch.Generator(_DEVICE).manual_seed(0)])

    call_kwargs = fake_pipeline.call_args.kwargs
    assert "num_inference_steps" not in call_kwargs


def test_generate_images_passes_num_inference_steps_when_set(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch("line.models.t2i.T2I._load")
    mocker.patch("line.models.base.BaseModel._send_to_device")
    model = T2I("m", _DEVICE, model_swapping=False, num_inference_steps=10, guidance_scale=None)
    fake_pipeline = _make_fake_pipeline(mocker, model)

    model.generate_images(["p"], [torch.Generator(_DEVICE).manual_seed(0)])

    call_kwargs = fake_pipeline.call_args.kwargs
    assert call_kwargs["num_inference_steps"] == 10


# ---------------------------------------------------------------------------
# Unsupported model ID — KeyError risk guard
# ---------------------------------------------------------------------------


def test_load_raises_for_unsupported_model_id(mocker: pytest_mock.MockerFixture) -> None:
    """RISK: _load does a dict lookup on _TEXT_TO_IMAGE_MODELS[self._model_id].

    An unsupported model_id raises KeyError with no descriptive message.
    This test documents the crash path so that callers know to validate the
    model_id before constructing a T2I instance.
    """
    # Do NOT mock _load so the real lookup executes
    mocker.patch("line.models.base.BaseModel._send_to_device")
    with pytest.raises((KeyError, Exception)):
        T2I(
            model_id="unsupported/model-xyz",
            device=torch.device("cpu"),
            model_swapping=False,
            num_inference_steps=1,
            guidance_scale=1,
        )
