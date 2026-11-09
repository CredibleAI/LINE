import hashlib
from pathlib import Path
from unittest.mock import mock_open

import pytest
import pytest_mock
from PIL import Image

from line.pipeline import history, utils

# --- add_concept ---


def test_add_concept_appends_to_generation_history():
    concept_history: dict[str, float] = {}
    generation_history: list[tuple[str, float]] = []

    history.add_concept(concept_history, generation_history, "cat", 0.5, h_size=0)

    assert generation_history == [("cat", 0.5)]


def test_add_concept_updates_concept_history_in_place():
    concept_history: dict[str, float] = {}
    original_ref = concept_history
    generation_history: list[tuple[str, float]] = []

    history.add_concept(concept_history, generation_history, "cat", 0.5, h_size=0)

    assert concept_history is original_ref
    assert concept_history == {"cat": 0.5}


def test_add_concept_below_h_size_no_eviction():
    concept_history = {"a": 0.3, "b": 0.7}
    generation_history: list[tuple[str, float]] = []

    history.add_concept(concept_history, generation_history, "c", 0.5, h_size=5)

    assert "c" in concept_history
    assert len(concept_history) == 3


def test_add_concept_evicts_worst_when_better():
    concept_history = {"a": 0.1, "b": 0.8, "c": 0.5}
    generation_history: list[tuple[str, float]] = []

    history.add_concept(concept_history, generation_history, "d", 0.6, h_size=3)

    assert "d" in concept_history
    assert "a" not in concept_history
    assert len(concept_history) == 3


def test_add_concept_evicts_one_when_worse_than_min():
    concept_history = {"a": 0.4, "b": 0.8, "c": 0.6}
    generation_history: list[tuple[str, float]] = []

    history.add_concept(concept_history, generation_history, "d", 0.1, h_size=3)

    assert "d" in concept_history
    assert len(concept_history) == 3


def test_add_concept_unbounded_h_size_zero_never_evicts():
    concept_history = {"a": 0.1, "b": 0.2, "c": 0.3}
    generation_history: list[tuple[str, float]] = []

    history.add_concept(concept_history, generation_history, "d", 0.05, h_size=0)

    assert len(concept_history) == 4
    assert "d" in concept_history


def test_add_concept_appends_each_call_to_generation_history():
    concept_history: dict[str, float] = {}
    generation_history: list[tuple[str, float]] = []

    history.add_concept(concept_history, generation_history, "x", 0.1, h_size=0)
    history.add_concept(concept_history, generation_history, "y", 0.9, h_size=0)

    assert generation_history == [("x", 0.1), ("y", 0.9)]


# --- get_best_concept ---


def test_get_best_concept_returns_highest_score():
    concept_history = {"a": 0.3, "b": 0.9, "c": 0.5}

    best, score = history.get_best_concept(concept_history)

    assert best == "b"
    assert score == pytest.approx(0.9)


def test_get_best_concept_single_entry():
    concept_history = {"only": 0.42}

    best, score = history.get_best_concept(concept_history)

    assert best == "only"
    assert score == pytest.approx(0.42)


# --- save_to_file ---


def test_save_to_file_creates_directory(mocker: pytest_mock.MockerFixture):
    mock_mkdir = mocker.patch("pathlib.Path.mkdir")
    mocker.patch("builtins.open", new_callable=mock_open)

    history.save_to_file(["line1", "line2"], Path("test/dir"), "output.txt")

    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_save_to_file_writes_joined_lines(mocker: pytest_mock.MockerFixture):
    mocker.patch("pathlib.Path.mkdir")
    mock_file = mock_open()
    mocker.patch("builtins.open", mock_file)

    history.save_to_file(["a", "b", "c"], Path("test/dir"), "output.txt")

    mock_file().write.assert_called_once_with("a\nb\nc")


def test_save_to_file_empty(mocker: pytest_mock.MockerFixture):
    mocker.patch("pathlib.Path.mkdir")
    mock_file = mock_open()
    mocker.patch("builtins.open", mock_file)

    history.save_to_file([], Path("test/dir"), "output.txt")

    mock_file().write.assert_called_once_with("")


def test_save_params_writes_json_file(tmp_path: Path):
    """save_params writes a valid JSON file named params.json in the given directory."""
    params = {"run_id": "abc", "metric": "AVG"}

    history.save_params(params, tmp_path)

    output_file = tmp_path / history.PARAMS_FILENAME
    assert output_file.exists()
    import json

    loaded = json.loads(output_file.read_text())
    assert loaded == params


def test_save_params_creates_directory_if_missing(tmp_path: Path):
    """save_params creates the target directory when it does not yet exist."""
    target = tmp_path / "deep" / "nested" / "dir"
    assert not target.exists()

    history.save_params({"x": 1}, target)

    assert (target / history.PARAMS_FILENAME).exists()


def test_read_params_round_trips_saved_params(tmp_path: Path):
    """read_params returns exactly the same dict that was passed to save_params."""
    params = {"run_id": "xyz", "n_iters": 5, "seed": 42}

    history.save_params(params, tmp_path)
    loaded = history.read_params(tmp_path)

    assert loaded == params


def test_read_params_raises_when_file_missing(tmp_path: Path):
    """read_params raises FileNotFoundError when params.json does not exist."""
    with pytest.raises(FileNotFoundError):
        history.read_params(tmp_path)


# ---------------------------------------------------------------------------
# update_concept_history — edge cases and bug guards
# ---------------------------------------------------------------------------


def test_update_concept_history_all_equal_scores_does_not_crash():
    """Regression guard: equal-score scoreboards must not crash on stochastic removal.

    When all concept scores are identical, the weight vector [max-s for s in scores]
    is all-zero. The implementation falls back to uniform random removal in that case.
    """
    scoreboard = {"a": 0.5, "b": 0.5, "c": 0.5}
    # new score (0.1) < min (0.5) → triggers stochastic removal path
    result = history.update_concept_history(scoreboard, "d", 0.1, h_size=3)
    assert "d" in result
    assert len(result) == 3


def test_update_concept_history_duplicate_concept_name_keeps_size_bounded():
    """Adding a concept that already exists in a full scoreboard must not exceed h_size.

    There is no deduplication guard: if 'cat' already exists with score 0.5 and
    is re-added with score 0.9, the old 'cat' entry may or may not be removed before
    the new one is inserted. The scoreboard must not grow beyond h_size.
    """
    scoreboard = {"cat": 0.5, "dog": 0.8}
    result = history.update_concept_history(scoreboard, "cat", 0.9, h_size=2)
    assert len(result) == 2
    # The re-added concept must be present with the new score
    assert result.get("cat") is not None


def test_get_best_concept_raises_on_empty_scoreboard():
    """get_best_concept with an empty scoreboard raises ValueError (max of empty seq).

    Callers must ensure the scoreboard is non-empty before calling this function.
    This test documents the crash risk so it can be guarded against upstream.
    """
    with pytest.raises((ValueError, KeyError)):
        history.get_best_concept({})


def test_update_concept_history_is_immutable_does_not_mutate_input():
    """update_concept_history must return a new dict and never mutate its input."""
    original = {"a": 0.3, "b": 0.7}
    original_copy = dict(original)

    history.update_concept_history(original, "c", 0.5, h_size=2)

    assert original == original_copy


def test_update_concept_history_better_concept():
    scoreboard = {
        "concept_1": 0.42,
        "concept_2": 0.83,
        "concept_3": 0.13,
        "concept_4": 0.24,
    }

    updated = history.update_concept_history(scoreboard, "concept_5", 0.53, h_size=4)

    assert "concept_5" in updated
    assert "concept_3" not in updated


# --- readers ---

# Trimmed from a real run saved before the switch to JSONL (resnet18|imagenet, avgpool,
# neuron 222): one concept contains a newline, so it spans two physical lines.
_LEGACY_GENERATION_HISTORY = (
    "keypad,2.5989391803741455\n"
    "input device,2.052428960800171\n"
    "keypad\n alternative: hand device,2.231809616088867\n"
    "physical layout,0.7886157035827637"
)
_LEGACY_FINAL_CONCEPT_HISTORY = (
    "computer keyboard,4.659331798553467\n"
    "marmot,0.1233988031744957\n"
    "keypad,2.5989391803741455\n"
    "input device,2.052428960800171\n"
    "keypad\n alternative: hand device,2.231809616088867\n"
    "physical layout,0.7886157035827637"
)


def _write_legacy_histories(neuron_dir: Path) -> None:
    (neuron_dir / "generation_history.txt").write_text(_LEGACY_GENERATION_HISTORY)
    (neuron_dir / "final_concept_history.txt").write_text(_LEGACY_FINAL_CONCEPT_HISTORY)


def test_find_output_file_prefers_jsonl(tmp_path: Path):
    (tmp_path / "best_concepts.jsonl").write_text("{}")
    (tmp_path / "best_concepts.txt").write_text("{}")

    found = history.find_output_file(tmp_path, history.BEST_CONCEPTS_FILENAME)

    assert found == tmp_path / "best_concepts.jsonl"


def test_find_output_file_falls_back_to_legacy_txt(tmp_path: Path):
    (tmp_path / "best_concepts.txt").write_text("{}")

    found = history.find_output_file(tmp_path, history.BEST_CONCEPTS_FILENAME)

    assert found == tmp_path / "best_concepts.txt"


def test_find_output_file_returns_none_when_missing(tmp_path: Path):
    assert history.find_output_file(tmp_path, history.BEST_CONCEPTS_FILENAME) is None


@pytest.mark.parametrize(
    "concept",
    ["a, b", 'say "hi"', "keypad\n alternative", "\nkeypad", "key\rpad", "key\u2028pad"],
)
def test_generation_history_round_trips_awkward_concepts(tmp_path: Path, concept: str):
    records = utils.format_generation_history([("cat", 0.5), (concept, 0.25)], n_iters=1)
    history.save_to_file(
        utils.format_as_json_string(records), tmp_path, history.GENERATION_HISTORY_FILENAME
    )

    assert history.read_generation_history(tmp_path) == records
    assert len((tmp_path / history.GENERATION_HISTORY_FILENAME).read_text().splitlines()) == 2


def test_final_concept_history_round_trips(tmp_path: Path):
    records = utils.format_final_concept_history(
        {"cat": 0.9, "dog, big": 0.2}, [("dog, big", 0.2)], initial_concepts=["cat"], n_iters=0
    )
    history.save_to_file(
        utils.format_as_json_string(records), tmp_path, history.FINAL_CONCEPT_HISTORY_FILENAME
    )

    assert history.read_final_concept_history(tmp_path) == records


def test_read_generation_history_legacy(tmp_path: Path):
    _write_legacy_histories(tmp_path)

    assert history.read_generation_history(tmp_path) == [
        {"iteration": 1, "concept": "keypad", "score": 2.5989391803741455},
        {"iteration": 2, "concept": "input device", "score": 2.052428960800171},
        {
            "iteration": 3,
            "concept": "keypad\n alternative: hand device",
            "score": 2.231809616088867,
        },
        {"iteration": "summary", "concept": "physical layout", "score": 0.7886157035827637},
    ]


def test_read_final_concept_history_legacy_is_sorted_with_summary_flag(tmp_path: Path):
    _write_legacy_histories(tmp_path)

    records = history.read_final_concept_history(tmp_path)

    assert [(r["concept"], r["score"]) for r in records] == [
        ("computer keyboard", 4.659331798553467),
        ("keypad", 2.5989391803741455),
        ("keypad\n alternative: hand device", 2.231809616088867),
        ("input device", 2.052428960800171),
        ("physical layout", 0.7886157035827637),
        ("marmot", 0.1233988031744957),
    ]
    assert [r["concept"] for r in records if r["is_summary"]] == ["physical layout"]
    assert all(r["first_iteration"] is None for r in records)


def test_read_legacy_history_rejects_trailing_partial_entry(tmp_path: Path):
    (tmp_path / "generation_history.txt").write_text("keypad,2.5\nno score here")

    with pytest.raises(ValueError, match="trailing text"):
        history.read_generation_history(tmp_path)


def test_read_best_concepts_and_reasoning_from_legacy_txt(tmp_path: Path):
    (tmp_path / "best_concepts.txt").write_text(
        '{"iteration": 1, "best_concept": "cat", "best_score": 0.9}'
    )
    (tmp_path / "reasoning.txt").write_text('{"concept": "cat", "reasoning": "fur,\\nwhiskers"}')

    assert history.read_best_concepts(tmp_path) == [
        {"iteration": 1, "best_concept": "cat", "best_score": 0.9}
    ]
    assert history.read_reasoning(tmp_path) == [{"concept": "cat", "reasoning": "fur,\nwhiskers"}]


def test_readers_return_empty_when_files_missing(tmp_path: Path):
    assert history.read_generation_history(tmp_path) == []
    assert history.read_final_concept_history(tmp_path) == []
    assert history.read_best_concepts(tmp_path) == []
    assert history.read_reasoning(tmp_path) == []


# --- image folders ---


def _hash_suffix(concept: str) -> str:
    return hashlib.sha256(concept.encode()).hexdigest()[:8]


@pytest.mark.parametrize(
    ("concept", "slug"),
    [
        ("computer keyboard", "computer_keyboard"),
        ("keypad\n alternative", "keypad__alternative"),
        ("land animal/object", "land_animal_object"),
        ("back\\slash", "back_slash"),
        ("key\u2028pad", "key_pad"),
    ],
)
def test_image_dir_name_disambiguates_lossy_slugs(concept: str, slug: str):
    expected = f"iteration_3_{slug}_{_hash_suffix(concept)}"
    assert history.image_dir_name(3, concept) == expected


def test_image_dir_name_is_unchanged_for_non_lossy_concepts():
    assert history.image_dir_name(3, "keypad") == "iteration_3_keypad"


def test_image_dir_name_disambiguates_previously_colliding_concepts():
    # Both used to slug to "iteration_0_land_animal_object".
    assert history.image_dir_name(0, "land animal/object") != history.image_dir_name(
        0, "land animal_object"
    )


def test_image_dir_name_truncates_within_the_filesystem_limit():
    concept = "ä" * 300
    name = history.image_dir_name("summary", concept)

    assert name.startswith("iteration_summary_")
    assert name.endswith(f"…_{_hash_suffix(concept)}")
    assert len(name.encode()) <= 255


def test_save_images_from_iteration_creates_a_single_folder(tmp_path: Path):
    history.save_images_from_iteration(
        [Image.new("RGB", (2, 2))], tmp_path, 3, "land animal/object"
    )

    expected_dir = history.image_dir_name(3, "land animal/object")
    assert [p.name for p in tmp_path.iterdir()] == [expected_dir]
    assert (tmp_path / expected_dir / "image_1.jpeg").is_file()


def test_find_concept_images_returns_initial_pool_then_each_iteration(tmp_path: Path):
    records = utils.format_generation_history(
        [("keypad", 0.4), ("cat", 0.9), ("keypad", 0.4)], n_iters=2
    )
    history.save_to_file(
        utils.format_as_json_string(records), tmp_path, history.GENERATION_HISTORY_FILENAME
    )
    for name in ("iteration_0_keypad", "iteration_1_keypad", "iteration_summary_keypad"):
        (tmp_path / history.IMAGES_DIRNAME / name).mkdir(parents=True)

    found = history.find_concept_images(tmp_path, "keypad")

    assert [p.name for p in found] == [
        "iteration_0_keypad",
        "iteration_1_keypad",
        "iteration_summary_keypad",
    ]


def test_find_concept_images_finds_disambiguated_folders(tmp_path: Path):
    records = utils.format_generation_history([("land animal/object", 0.4)], n_iters=0)
    history.save_to_file(
        utils.format_as_json_string(records), tmp_path, history.GENERATION_HISTORY_FILENAME
    )
    history.save_images_from_iteration(
        [Image.new("RGB", (2, 2))],
        tmp_path / history.IMAGES_DIRNAME,
        "summary",
        "land animal/object",
    )

    found = history.find_concept_images(tmp_path, "land animal/object")

    assert [p.name for p in found] == [history.image_dir_name("summary", "land animal/object")]


def test_find_concept_images_falls_back_to_legacy_folder_names(tmp_path: Path):
    records = utils.format_generation_history([("land animal/object", 0.4)], n_iters=0)
    history.save_to_file(
        utils.format_as_json_string(records), tmp_path, history.GENERATION_HISTORY_FILENAME
    )
    legacy = tmp_path / history.IMAGES_DIRNAME / "iteration_summary_land_animal" / "object"
    legacy.mkdir(parents=True)

    assert history.find_concept_images(tmp_path, "land animal/object") == [legacy]


def test_find_concept_images_returns_empty_when_there_are_none(tmp_path: Path):
    assert history.find_concept_images(tmp_path, "cat") == []
