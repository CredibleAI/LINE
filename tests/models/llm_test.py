"""Tests for line.models.llm — LLM."""

from typing import Annotated, cast
from unittest.mock import MagicMock

import pytest
import pytest_mock
import torch

from line.models.llm import (
    _ANSWER_TAG_CLOSE,
    _ANSWER_TAG_OPEN,
    _ASSISTANT_TAG,
    _EOT_TAG,
    _THINKING_TAG_CLOSE,
    _THINKING_TAG_OPEN,
    IGNORE_SENTINEL,
    LLM,
)

LLMFixture = Annotated[LLM, pytest.fixture]

_FAKE_PROMPT_PATH = "fake/prompt.txt"
_FAKE_SUMMARY_PATH = "fake/summary.txt"
_CONCEPT_HISTORIES: list[dict[str, float]] = [{"cat": 0.8, "dog": 0.5}]
_GENERATION_HISTORIES: list[list[tuple[str, float]]] = [[("cat", 0.8), ("dog", 0.5)]]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def llm(mocker: pytest_mock.MockerFixture) -> LLM:
    mocker.patch("line.models.llm.LLM._load", return_value=MagicMock())
    mocker.patch("line.models.base.BaseModel._send_to_device")
    mocker.patch("pathlib.Path.is_file", return_value=True)
    mocker.patch(
        "line.models.prompt_utils._read_prompt_template",
        side_effect=lambda path: f"template:{path}",
    )
    model = LLM(
        model_id="test-llm",
        device=torch.device("cpu"),
        model_swapping=False,
        max_new_tokens=128,
        prompt_path=_FAKE_PROMPT_PATH,
        summary_prompt_path=_FAKE_SUMMARY_PATH,
    )
    # Attach a mock pipeline so generate_concept can run
    mock_pipeline = MagicMock()
    mock_pipeline.device = MagicMock()
    mock_pipeline.tokenizer = MagicMock()
    mock_pipeline.tokenizer.eos_token_id = 1
    mock_pipeline.tokenizer.convert_tokens_to_ids.return_value = 2
    model._model = mock_pipeline
    return model


# ---------------------------------------------------------------------------
# generate_concept
# ---------------------------------------------------------------------------


def _make_llm_response(thinking: str, answer: str) -> str:
    return (
        f"user message{_ASSISTANT_TAG}\n"
        f"{_THINKING_TAG_OPEN}{thinking}{_THINKING_TAG_CLOSE}\n"
        f"{_ANSWER_TAG_OPEN}{answer}{_ANSWER_TAG_CLOSE}"
    )


@pytest.fixture
def llm_with_response(llm: LLMFixture, mocker: pytest_mock.MockerFixture) -> tuple[LLM, str, str]:
    thinking = "step by step reasoning"
    answer = "striped cat"
    mock_pipeline = cast(MagicMock, llm._model)
    mock_pipeline.return_value = [{"generated_text": _make_llm_response(thinking, answer)}]
    mock_pipeline.tokenizer.apply_chat_template.return_value = "formatted message"
    mocker.patch(
        "line.models.llm.prompt_utils.generate_prompt",
        return_value="generated prompt",
    )
    return llm, thinking, answer


def test_generate_concept_returns_answer_and_reasoning(
    llm_with_response: tuple,
) -> None:
    llm, expected_thinking, expected_answer = llm_with_response
    results = llm.generate_concept(_CONCEPT_HISTORIES, _GENERATION_HISTORIES)
    assert len(results) == 1
    answer, reasoning = results[0]
    assert answer == expected_answer
    assert reasoning == expected_thinking


def test_generate_concept_passes_eot_tag_to_tokenizer(
    llm_with_response: tuple,
) -> None:
    llm, _, _ = llm_with_response
    llm.generate_concept(_CONCEPT_HISTORIES, _GENERATION_HISTORIES)
    cast(MagicMock, llm._model).tokenizer.convert_tokens_to_ids.assert_called_once_with(_EOT_TAG)


def test_generate_concept_uses_full_prompt_when_no_top_k(
    llm_with_response: tuple, mocker: pytest_mock.MockerFixture
) -> None:
    llm, _, _ = llm_with_response
    mock_generate_prompt = mocker.patch(
        "line.models.llm.prompt_utils.generate_prompt",
        return_value="generated prompt",
    )
    llm.generate_concept(_CONCEPT_HISTORIES, _GENERATION_HISTORIES, top_k=None)
    args, _kwargs = mock_generate_prompt.call_args
    assert args[2] == f"template:{_FAKE_PROMPT_PATH}"


def test_generate_concept_uses_summary_prompt_when_top_k_given(
    llm_with_response: tuple, mocker: pytest_mock.MockerFixture
) -> None:
    llm, _, _ = llm_with_response
    mock_generate_prompt = mocker.patch(
        "line.models.llm.prompt_utils.generate_prompt",
        return_value="generated prompt",
    )
    llm.generate_concept(_CONCEPT_HISTORIES, _GENERATION_HISTORIES, top_k=3)
    args, _kwargs = mock_generate_prompt.call_args
    assert args[2] == f"template:{_FAKE_SUMMARY_PATH}"
    assert args[3] == 3


def test_generate_concept_passes_histories_to_prompt(
    llm_with_response: tuple, mocker: pytest_mock.MockerFixture
) -> None:
    llm, _, _ = llm_with_response
    mock_generate_prompt = mocker.patch(
        "line.models.llm.prompt_utils.generate_prompt",
        return_value="generated prompt",
    )
    llm.generate_concept(_CONCEPT_HISTORIES, _GENERATION_HISTORIES)
    args, _kwargs = mock_generate_prompt.call_args
    assert args[0] is _CONCEPT_HISTORIES[0]
    assert args[1] is _GENERATION_HISTORIES[0]


def test_generate_concept_multiple_messages(
    llm: LLMFixture, mocker: pytest_mock.MockerFixture
) -> None:
    """One result per input, each parsed from its own output, prompt built per history."""
    mock_pipeline = cast(MagicMock, llm._model)
    mock_pipeline.tokenizer.apply_chat_template.side_effect = lambda msgs, **_: msgs[0]["content"]
    mock_pipeline.return_value = [
        {"generated_text": _make_llm_response("reasoning A", "concept A")},
        {"generated_text": _make_llm_response("reasoning B", "concept B")},
        {"generated_text": _make_llm_response("reasoning C", "concept C")},
    ]
    mock_gen_prompt = mocker.patch(
        "line.models.llm.prompt_utils.generate_prompt",
        side_effect=lambda ch, gh, *_: f"prompt for {next(iter(ch.keys()))}",
    )

    ch_0, ch_1, ch_2 = {"cat": 0.9}, {"dog": 0.7}, {"bird": 0.5}
    gh_0, gh_1, gh_2 = [("cat", 0.9)], [("dog", 0.7)], [("bird", 0.5)]

    results = llm.generate_concept([ch_0, ch_1, ch_2], [gh_0, gh_1, gh_2])

    assert results == [
        ("concept A", "reasoning A"),
        ("concept B", "reasoning B"),
        ("concept C", "reasoning C"),
    ]
    assert mock_gen_prompt.call_count == 3
    assert mock_gen_prompt.call_args_list[0].args[:2] == (ch_0, gh_0)
    assert mock_gen_prompt.call_args_list[1].args[:2] == (ch_1, gh_1)
    assert mock_gen_prompt.call_args_list[2].args[:2] == (ch_2, gh_2)


# ---------------------------------------------------------------------------
# Model swapping
# ---------------------------------------------------------------------------


def test_model_swapping_moves_underlying_model_to_cuda_then_cpu(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch("line.models.llm.LLM._load", return_value=MagicMock())
    mocker.patch("line.models.base.BaseModel._send_to_device")
    mocker.patch("line.models.base.torch.cuda.is_available", return_value=True)
    mocker.patch("pathlib.Path.is_file", return_value=True)
    mocker.patch("line.models.prompt_utils._read_prompt_template", return_value="t")
    model = LLM(
        "m",
        torch.device("cpu"),
        model_swapping=True,
        max_new_tokens=1,
        prompt_path="p",
        summary_prompt_path="s",
    )
    send_spy = mocker.patch.object(model, "_send_to_device")

    mocker.patch("line.models.llm.prompt_utils.generate_prompt", return_value="prompt")
    mock_pipeline = cast(MagicMock, model._model)
    mock_pipeline.tokenizer.apply_chat_template.return_value = "msg"
    mock_pipeline.tokenizer.eos_token_id = 1
    mock_pipeline.tokenizer.convert_tokens_to_ids.return_value = 2
    mock_pipeline.return_value = [{"generated_text": _make_llm_response("r", "a")}]

    model.generate_concept(_CONCEPT_HISTORIES, _GENERATION_HISTORIES)

    calls = [c.args[0] for c in send_spy.call_args_list]
    assert calls == [torch.device("cuda"), torch.device("cpu")]


def test_no_model_swapping_skips_device_transfer(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch("line.models.llm.LLM._load", return_value=MagicMock())
    mocker.patch("line.models.base.BaseModel._send_to_device")
    mocker.patch("pathlib.Path.is_file", return_value=True)
    mocker.patch("line.models.prompt_utils._read_prompt_template", return_value="t")
    model = LLM(
        "m",
        torch.device("cpu"),
        model_swapping=False,
        max_new_tokens=1,
        prompt_path="p",
        summary_prompt_path="s",
    )
    send_spy = mocker.patch.object(model, "_send_to_device")

    mocker.patch("line.models.llm.prompt_utils.generate_prompt", return_value="prompt")
    mock_pipeline = cast(MagicMock, model._model)
    mock_pipeline.tokenizer.apply_chat_template.return_value = "msg"
    mock_pipeline.tokenizer.eos_token_id = 1
    mock_pipeline.tokenizer.convert_tokens_to_ids.return_value = 2
    mock_pipeline.return_value = [{"generated_text": _make_llm_response("r", "a")}]

    model.generate_concept(_CONCEPT_HISTORIES, _GENERATION_HISTORIES)

    send_spy.assert_not_called()


def test_send_to_device_moves_pipeline_model_and_updates_device(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch("line.models.llm.LLM._load", return_value=MagicMock())
    mocker.patch("line.models.base.BaseModel._send_to_device")
    mocker.patch("pathlib.Path.is_file", return_value=True)
    mocker.patch("line.models.prompt_utils._read_prompt_template", return_value="t")
    model = LLM(
        "m",
        torch.device("cpu"),
        model_swapping=False,
        max_new_tokens=1,
        prompt_path="p",
        summary_prompt_path="s",
    )

    model._send_to_device(torch.device("cuda"))

    model._model.model.to.assert_called_once_with(torch.device("cuda"))
    assert model._model.device == torch.device("cuda")
    assert model._device == torch.device("cuda")


# ---------------------------------------------------------------------------
# Malformed LLM response — IndexError exposure
# ---------------------------------------------------------------------------


def test_generate_concept_returns_sentinel_on_missing_assistant_tag(
    llm_with_response: tuple,
) -> None:
    """A response with no _ASSISTANT_TAG at all must return IGNORE_SENTINEL."""
    llm, _, _ = llm_with_response
    cast(MagicMock, llm._model).return_value = [
        {"generated_text": "no assistant tag anywhere in this text"}
    ]
    results = llm.generate_concept(_CONCEPT_HISTORIES, _GENERATION_HISTORIES)
    assert len(results) == 1
    answer, reasoning = results[0]
    assert answer == IGNORE_SENTINEL
    assert reasoning == ""


def test_generate_concept_returns_sentinel_on_missing_thinking_tag(
    llm_with_response: tuple,
) -> None:
    """A response without <thinking>/<answer> tags must return IGNORE_SENTINEL, not raw text."""
    llm, _, _ = llm_with_response
    cast(MagicMock, llm._model).return_value = [
        {"generated_text": f"prefix{_ASSISTANT_TAG}no tags at all just text"}
    ]
    results = llm.generate_concept(_CONCEPT_HISTORIES, _GENERATION_HISTORIES)
    assert len(results) == 1
    answer, reasoning = results[0]
    assert answer == IGNORE_SENTINEL
    assert reasoning == ""


def test_generate_concept_returns_sentinel_on_missing_answer_tag(
    llm_with_response: tuple,
) -> None:
    """A response with <thinking> but no <answer> tag must return IGNORE_SENTINEL."""
    llm, _, _ = llm_with_response
    cast(MagicMock, llm._model).return_value = [
        {
            "generated_text": (
                f"prefix{_ASSISTANT_TAG}<thinking>some reasoning</thinking>no answer here"
            )
        }
    ]
    results = llm.generate_concept(_CONCEPT_HISTORIES, _GENERATION_HISTORIES)
    assert len(results) == 1
    answer, reasoning = results[0]
    assert answer == IGNORE_SENTINEL
    assert reasoning == ""


def test_generate_concept_strips_partial_closing_tag_when_truncated(
    llm_with_response: tuple,
) -> None:
    """
    When max_new_tokens cuts off before </answer>, the partial tag must not appear in the answer.
    """
    llm, _, _ = llm_with_response
    truncated_response = (
        f"prefix{_ASSISTANT_TAG}\n"
        f"{_THINKING_TAG_OPEN}some reasoning{_THINKING_TAG_CLOSE}\n"
        f"{_ANSWER_TAG_OPEN}support garment\n</answer"  # truncated — no closing >
    )
    cast(MagicMock, llm._model).return_value = [{"generated_text": truncated_response}]
    results = llm.generate_concept(_CONCEPT_HISTORIES, _GENERATION_HISTORIES)
    assert len(results) == 1
    answer, _ = results[0]
    assert answer == "support garment"


# ---------------------------------------------------------------------------
# Batching — generate_concept respects batch_size
# ---------------------------------------------------------------------------


def test_generate_concept_respects_batch_size_makes_multiple_pipeline_calls(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """With batch_size=2 and 3 neurons, the underlying pipeline must be called twice.

    Verifies that the batching loop in generate_concept correctly slices the
    messages list and accumulates results from all batches.
    """
    mocker.patch("line.models.llm.LLM._load", return_value=MagicMock())
    mocker.patch("line.models.base.BaseModel._send_to_device")
    mocker.patch("pathlib.Path.is_file", return_value=True)
    mocker.patch("line.models.prompt_utils._read_prompt_template", return_value="t")
    model = LLM(
        model_id="test-llm",
        device=torch.device("cpu"),
        model_swapping=False,
        max_new_tokens=128,
        prompt_path=_FAKE_PROMPT_PATH,
        summary_prompt_path=_FAKE_SUMMARY_PATH,
        batch_size=2,
    )

    mock_pipeline = MagicMock()
    mock_pipeline.device = MagicMock()
    mock_pipeline.tokenizer = MagicMock()
    mock_pipeline.tokenizer.eos_token_id = 1
    mock_pipeline.tokenizer.convert_tokens_to_ids.return_value = 2
    mock_pipeline.tokenizer.apply_chat_template.return_value = "formatted"
    mock_pipeline.side_effect = [
        # First batch: 2 results
        [
            {"generated_text": _make_llm_response("r1", "a1")},
            {"generated_text": _make_llm_response("r2", "a2")},
        ],
        # Second batch: 1 result
        [{"generated_text": _make_llm_response("r3", "a3")}],
    ]
    model._model = mock_pipeline

    mocker.patch(
        "line.models.llm.prompt_utils.generate_prompt",
        return_value="generated prompt",
    )

    ch = [{"cat": 0.8}, {"dog": 0.5}, {"bird": 0.3}]
    gh = [[("cat", 0.8)], [("dog", 0.5)], [("bird", 0.3)]]

    results = model.generate_concept(ch, gh)

    assert mock_pipeline.call_count == 2
    assert results == [("a1", "r1"), ("a2", "r2"), ("a3", "r3")]


def test_generate_concept_batch_size_zero_uses_single_call(
    llm_with_response: tuple,
) -> None:
    """batch_size=0 means all neurons are processed in a single pipeline call."""
    llm, _, _ = llm_with_response
    mock_pipeline = cast(MagicMock, llm._model)
    # Default batch_size is 0 → one call for all inputs
    mock_pipeline.return_value = [
        {"generated_text": _make_llm_response("r1", "a1")},
        {"generated_text": _make_llm_response("r2", "a2")},
    ]
    mock_pipeline.tokenizer.apply_chat_template.return_value = "formatted"

    ch = [{"cat": 0.8}, {"dog": 0.5}]
    gh = [[("cat", 0.8)], [("dog", 0.5)]]
    llm.generate_concept(ch, gh)

    assert mock_pipeline.call_count == 1


# ---------------------------------------------------------------------------
# Deterministic flag — sampling kwargs
# ---------------------------------------------------------------------------


def test_generate_concept_deterministic_passes_do_sample_false(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch("line.models.llm.LLM._load", return_value=MagicMock())
    mocker.patch("line.models.base.BaseModel._send_to_device")
    mocker.patch("pathlib.Path.is_file", return_value=True)
    mocker.patch("line.models.prompt_utils._read_prompt_template", return_value="t")
    model = LLM(
        model_id="test-llm",
        device=torch.device("cpu"),
        model_swapping=False,
        max_new_tokens=128,
        prompt_path=_FAKE_PROMPT_PATH,
        summary_prompt_path=_FAKE_SUMMARY_PATH,
        deterministic=True,
    )
    mock_pipeline = MagicMock()
    mock_pipeline.device = MagicMock()
    mock_pipeline.tokenizer = MagicMock()
    mock_pipeline.tokenizer.eos_token_id = 1
    mock_pipeline.tokenizer.convert_tokens_to_ids.return_value = 2
    mock_pipeline.tokenizer.apply_chat_template.return_value = "msg"
    mock_pipeline.return_value = [{"generated_text": _make_llm_response("r", "a")}]
    model._model = mock_pipeline
    mocker.patch("line.models.llm.prompt_utils.generate_prompt", return_value="prompt")

    model.generate_concept(_CONCEPT_HISTORIES, _GENERATION_HISTORIES)

    _, kwargs = mock_pipeline.call_args
    assert kwargs.get("do_sample") is False
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_generate_concept_stochastic_passes_sampling_kwargs(
    llm_with_response: tuple,
) -> None:
    llm, _, _ = llm_with_response
    mock_pipeline = cast(MagicMock, llm._model)
    mock_pipeline.tokenizer.apply_chat_template.return_value = "msg"

    llm.generate_concept(_CONCEPT_HISTORIES, _GENERATION_HISTORIES)

    _, kwargs = mock_pipeline.call_args
    assert kwargs.get("do_sample") is True
    assert "temperature" in kwargs
    assert "top_p" in kwargs


# ---------------------------------------------------------------------------
# Initialization — prompt file validation
# ---------------------------------------------------------------------------


def _make_llm(mocker, prompt_path: str, summary_prompt_path: str) -> LLM:
    """Helper that constructs an LLM with _load and _send_to_device mocked out."""
    mocker.patch("line.models.llm.LLM._load", return_value=MagicMock())
    mocker.patch("line.models.base.BaseModel._send_to_device")
    return LLM(
        model_id="test-llm",
        device=torch.device("cpu"),
        model_swapping=False,
        max_new_tokens=128,
        prompt_path=prompt_path,
        summary_prompt_path=summary_prompt_path,
    )


def test_init_raises_when_prompt_path_missing(mocker: pytest_mock.MockerFixture, tmp_path) -> None:
    summary_file = tmp_path / "summary.txt"
    summary_file.write_text("summary template")

    with pytest.raises(FileNotFoundError, match="prompt_path"):
        _make_llm(mocker, str(tmp_path / "missing.txt"), str(summary_file))


def test_init_raises_when_summary_prompt_path_missing(
    mocker: pytest_mock.MockerFixture, tmp_path
) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("main template")

    with pytest.raises(FileNotFoundError, match="summary_prompt_path"):
        _make_llm(mocker, str(prompt_file), str(tmp_path / "missing.txt"))


def test_init_succeeds_when_both_prompt_files_exist(
    mocker: pytest_mock.MockerFixture, tmp_path
) -> None:
    prompt_file = tmp_path / "prompt.txt"
    summary_file = tmp_path / "summary.txt"
    prompt_file.write_text("main template")
    summary_file.write_text("summary template")

    model = _make_llm(mocker, str(prompt_file), str(summary_file))
    assert model._prompt_template == "main template"
    assert model._summary_prompt_template == "summary template"
