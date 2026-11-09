from line.pipeline import utils


def test_format_as_json_string():
    data = [{"concept": "cat", "score": 0.9}, {"concept": "dog", "score": 0.7}]
    result = utils.format_as_json_string(data)
    assert result == [
        '{"concept": "cat", "score": 0.9}',
        '{"concept": "dog", "score": 0.7}',
    ]


def test_format_as_json_string_empty():
    assert utils.format_as_json_string([]) == []


def test_format_generation_history_labels_iterations_and_summary():
    generation_history = [("cat", 0.5), ("dog", 0.8), ("pet", 0.7)]
    result = utils.format_generation_history(generation_history, n_iters=2)
    assert result == [
        {"iteration": 1, "concept": "cat", "score": 0.5},
        {"iteration": 2, "concept": "dog", "score": 0.8},
        {"iteration": "summary", "concept": "pet", "score": 0.7},
    ]


def test_format_generation_history_without_main_iterations():
    result = utils.format_generation_history([("cat", 0.5)], n_iters=0)
    assert result == [{"iteration": "summary", "concept": "cat", "score": 0.5}]


def test_format_generation_history_empty():
    assert utils.format_generation_history([], n_iters=3) == []


def test_format_final_concept_history_sorts_best_first_with_stable_ties():
    concept_history = {"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.9}
    result = utils.format_final_concept_history(
        concept_history, [], initial_concepts=["a", "b", "c", "d"], n_iters=0
    )
    assert [r["concept"] for r in result] == ["b", "d", "c", "a"]
    assert list(concept_history) == ["a", "b", "c", "d"]


def test_format_final_concept_history_first_iteration_and_summary():
    concept_history = {"cat": 0.9, "dog": 0.2, "keypad": 0.6, "grid": 0.4}
    generation_history = [
        ("keypad", 0.6),  # iteration 1
        ("cat", 0.9),  # iteration 2: an initial concept proposed again
        ("keypad", 0.6),  # iteration 3: proposed again
        ("grid", 0.4),  # summary
    ]
    result = utils.format_final_concept_history(
        concept_history, generation_history, initial_concepts=["cat", "dog"], n_iters=3
    )
    assert result == [
        {"concept": "cat", "score": 0.9, "first_iteration": 0, "is_summary": False},
        {"concept": "keypad", "score": 0.6, "first_iteration": 1, "is_summary": False},
        {"concept": "grid", "score": 0.4, "first_iteration": "summary", "is_summary": True},
        {"concept": "dog", "score": 0.2, "first_iteration": 0, "is_summary": False},
    ]


def test_format_final_concept_history_summary_already_on_board():
    concept_history = {"cat": 0.9, "keypad": 0.6}
    generation_history = [("keypad", 0.6), ("keypad", 0.6)]  # iteration 1, then summary
    result = utils.format_final_concept_history(
        concept_history, generation_history, initial_concepts=["cat"], n_iters=1
    )
    assert result[1] == {
        "concept": "keypad",
        "score": 0.6,
        "first_iteration": 1,
        "is_summary": True,
    }


def test_format_final_concept_history_empty():
    assert utils.format_final_concept_history({}, [], initial_concepts=[], n_iters=3) == []
