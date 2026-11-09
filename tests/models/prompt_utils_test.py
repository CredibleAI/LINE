import pytest_mock

from line.models import prompt_utils


def test_concept_image_prompt(mocker: pytest_mock.MockerFixture):
    fake_available_descriptions = (("fake angle",), ("fake lightning",))
    fake_prompt_text = "prompt text"
    fake_concept = "concept"
    mocker.patch(
        "line.models.prompt_utils._get_random_visual_descriptions",
        return_value=(descriptions[0] for descriptions in fake_available_descriptions),
    )

    expected = "prompt text concept, fake angle, fake lightning"
    actual = prompt_utils.concept_image_prompt(fake_prompt_text, fake_concept)
    assert actual == expected


def test_concept_image_prompt_seeded_is_deterministic():
    prompt1 = prompt_utils.concept_image_prompt("base text", "dog", seed=42)
    prompt2 = prompt_utils.concept_image_prompt("base text", "dog", seed=42)
    assert prompt1 == prompt2


def test_concept_image_prompt_different_seeds_differ():
    # Run enough seeds to be confident at least two produce distinct descriptions.
    results = {prompt_utils.concept_image_prompt("base text", "dog", seed=s) for s in range(20)}
    assert len(results) > 1, "All seeds produced identical prompts — seeding has no effect"


def test_generate_prompt_no_top_k():
    fake_concept_history = {"concept_1": 1.0, "concept_2": 2.0, "concept_3": 3.0}
    fake_generation_history = [("concept_1", 1.0), ("concept_2", 2.0)]
    fake_prompt_template = "list {concept_list}\nhistory {generation_history}"

    expected = """
    list concept_3: 3.00; concept_2: 2.00; concept_1: 1.00;\nhistory concept_1, concept_2
    """.strip()
    actual = prompt_utils.generate_prompt(
        fake_concept_history, fake_generation_history, fake_prompt_template
    )
    assert actual == expected


def test_generate_prompt_top_k():
    fake_concept_history = {"concept_1": 1.0, "concept_2": 2.0, "concept_3": 3.0}
    fake_generation_history = [("concept_1", 1.0), ("concept_2", 2.0), ("concept_3", 3.0)]
    fake_prompt_template = "list {concept_list}\nhistory {generation_history}"

    expected = "list concept_3: 3.00; concept_2: 2.00;\nhistory concept_1, concept_2, concept_3"
    actual = prompt_utils.generate_prompt(
        fake_concept_history, fake_generation_history, fake_prompt_template, 2
    )
    assert actual == expected


def test_generate_prompt_empty_concept_history_does_not_raise():
    """generate_prompt with an empty concept_history must not raise.

    An empty scoreboard produces concept_list=';' (the trailing semicolon from
    '; '.join([]) + ';'). The prompt is still filled in and returned as a string.
    """
    result = prompt_utils.generate_prompt(
        {}, [], "list {concept_list}\nhistory {generation_history}"
    )

    assert isinstance(result, str)
    # Template placeholders must be resolved
    assert "{concept_list}" not in result
    assert "{generation_history}" not in result


def test_generate_prompt_empty_generation_history_produces_empty_history_string():
    """An empty generation_history results in an empty history field, not a crash."""
    result = prompt_utils.generate_prompt(
        {"cat": 0.9}, [], "concepts: {concept_list} | history: {generation_history}"
    )

    # "history:" appears in the result (trailing space stripped by generate_prompt)
    assert "history:" in result
    # The history section should be present but empty (no concept names listed)
    history_part = result.split("history:")[1]
    assert "cat" not in history_part
