import dataclasses
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pytest_mock
import torch

from line import config
from line.config import NeuronState, RunConfig
from line.line import LINE
from line.pipeline import history

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_neuron_state(neuron_id: int, layer: str = "layer4") -> NeuronState:
    return NeuronState(
        neuron_id=neuron_id,
        layer=layer,
        model_id="test-expl",
        save_directory=Path(f"/tmp/neuron_{neuron_id}"),
        h_size=0,
        concept_history={f"concept_{neuron_id}": 0.9},
    )


def _make_run_config(
    load_config: config.LoadConfig,
    image_gen_config: config.ImageGenerationConfig,
    concept_history_config: config.ConceptHistoryConfig,
    history_managing_config: config.HistoryManagingConfig,
    neurons: dict[str, list[int]] | None = None,
    n_iters: int = 3,
    metric: str = "AVG_ACTIVATION",
) -> RunConfig:
    return RunConfig(
        load_config=load_config,
        image_generation_config=image_gen_config,
        concept_history_config=concept_history_config,
        history_managing_config=history_managing_config,
        n_iters=n_iters,
        neurons=neurons if neurons is not None else {"layer4": [0, 1]},
        initializations_path="/tmp/activations",
        metric=metric,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def load_config() -> config.LoadConfig:
    return config.LoadConfig(
        language_model_id="test-lm",
        text_to_image_model_id="test-t2i",
        explained_model_id="test-expl",
        language_model_kwargs={},
        text_to_image_model_kwargs={},
        explained_model_kwargs={},
        model_swapping=False,
    )


@pytest.fixture
def image_gen_config() -> config.ImageGenerationConfig:
    return config.ImageGenerationConfig(n_images=2, prompt_text="A realistic photo of a")


@pytest.fixture
def concept_history_config() -> config.ConceptHistoryConfig:
    return config.ConceptHistoryConfig(h_size=0)


@pytest.fixture
def history_managing_config(tmp_path: Path) -> config.HistoryManagingConfig:
    return config.HistoryManagingConfig(
        save_images=False,
        save_histories=False,
        save_directory=str(tmp_path),
    )


@pytest.fixture
def pipeline(
    load_config: config.LoadConfig,
    image_gen_config: config.ImageGenerationConfig,
    concept_history_config: config.ConceptHistoryConfig,
    history_managing_config: config.HistoryManagingConfig,
) -> LINE:
    mock_explained_model = MagicMock()
    mock_explained_model.hooked_layers = ["layer4"]
    run_config = _make_run_config(
        load_config, image_gen_config, concept_history_config, history_managing_config
    )
    return LINE(run_config, mock_explained_model, torch.device("cpu"))


# ---------------------------------------------------------------------------
# _generate_images
# ---------------------------------------------------------------------------


def test_generate_images_passes_correct_args(pipeline: LINE):
    mock_images = [MagicMock(), MagicMock()]
    pipeline._t2i_model = MagicMock()
    pipeline._t2i_model.generate_images_from_concept.return_value = mock_images

    result = pipeline._generate_images(["cat"])

    pipeline._t2i_model.generate_images_from_concept.assert_called_once_with(
        concepts=[("A realistic photo of a", "cat")],
        n_images=2,
    )
    assert result == mock_images


def test_generate_images_multiple_concepts(pipeline: LINE):
    mock_images = [MagicMock() for _ in range(6)]  # 3 concepts * 2 images
    pipeline._t2i_model = MagicMock()
    pipeline._t2i_model.generate_images_from_concept.return_value = mock_images

    result = pipeline._generate_images(["cat", "dog", "bird"])

    pipeline._t2i_model.generate_images_from_concept.assert_called_once_with(
        concepts=[
            ("A realistic photo of a", "cat"),
            ("A realistic photo of a", "dog"),
            ("A realistic photo of a", "bird"),
        ],
        n_images=2,
    )
    assert result == mock_images


# ---------------------------------------------------------------------------
# _initialize_concept_histories
# ---------------------------------------------------------------------------


def test_initialize_concept_histories_calls_get_initial_concepts_once_per_layer(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    """get_initial_concepts is called once per layer, not once per neuron."""
    mock_get = mocker.patch(
        "line.line.initialization.get_initial_concepts",
        return_value={0: ["cat"], 1: ["dog"]},
    )
    # n_images=2, 2 unique concepts → 4 images total
    fake_images = [MagicMock() for _ in range(4)]
    mocker.patch.object(pipeline, "_generate_images", return_value=fake_images)

    fake_activations = {"layer4": torch.zeros(4, 5)}
    pipeline._explained_model = MagicMock()
    pipeline._explained_model.get_activations.return_value = fake_activations

    # Populate neuron_states so the method can iterate
    pipeline._neuron_states = {
        ("layer4", 0): _make_neuron_state(0),
        ("layer4", 1): _make_neuron_state(1),
    }

    pipeline._initialize_concept_histories()

    mock_get.assert_called_once()
    call_kwargs = mock_get.call_args.kwargs
    assert set(call_kwargs["neuron_ids"]) == {0, 1}


def test_initialize_concept_histories_generates_unique_concepts_only(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    """Images are generated for unique concepts across neurons, not per neuron."""
    mocker.patch(
        "line.line.initialization.get_initial_concepts",
        return_value={0: ["cat", "dog"], 1: ["dog", "bird"]},  # "dog" shared
    )
    mock_generate = mocker.patch.object(
        pipeline, "_generate_images", return_value=[MagicMock()] * 6
    )
    pipeline._explained_model = MagicMock()
    pipeline._explained_model.get_activations.return_value = {"layer4": torch.zeros(6, 5)}
    pipeline._neuron_states = {
        ("layer4", 0): _make_neuron_state(0),
        ("layer4", 1): _make_neuron_state(1),
    }

    pipeline._initialize_concept_histories()

    # Only 3 unique concepts: cat, dog, bird
    called_concepts = mock_generate.call_args.args[0]
    assert len(called_concepts) == 3
    assert set(called_concepts) == {"cat", "dog", "bird"}


def test_initialize_concept_histories_scores_assigned_per_neuron(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    """Each neuron's concept_history is populated with correct scores and its seed recorded."""
    mocker.patch(
        "line.line.initialization.get_initial_concepts",
        return_value={0: ["cat"], 1: ["dog"]},
    )
    # 2 concepts, n_images=2 → 4 images; layout: [cat_g0, dog_g0, cat_g1, dog_g1]
    n_images, n_concepts = 2, 2
    fake_act = torch.zeros(n_images * n_concepts, 5)
    fake_act[0, 0] = 0.4  # cat_g0, neuron 0
    fake_act[2, 0] = 0.6  # cat_g1, neuron 0  → mean = 0.5
    fake_act[1, 1] = 0.2  # dog_g0, neuron 1
    fake_act[3, 1] = 0.4  # dog_g1, neuron 1  → mean = 0.3
    mocker.patch.object(pipeline, "_generate_images", return_value=[MagicMock()] * 4)
    pipeline._explained_model = MagicMock()
    pipeline._explained_model.get_activations.return_value = {"layer4": fake_act}

    state0 = _make_neuron_state(0)
    state1 = _make_neuron_state(1)
    pipeline._neuron_states = {("layer4", 0): state0, ("layer4", 1): state1}

    pipeline._initialize_concept_histories()

    assert math.isclose(state0.concept_history["cat"], 0.5, rel_tol=1e-5)
    assert math.isclose(state1.concept_history["dog"], 0.3, rel_tol=1e-5)
    assert state0.initial_concepts == ["cat"]
    assert state1.initial_concepts == ["dog"]


def test_initialize_concept_histories_groups_neurons_by_layer(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    """Neurons in different layers are grouped: get_initial_concepts is called once
    per layer and receives only the neuron IDs that belong to that layer."""
    layer4_path = Path("/tmp/layer4")
    layer3_path = Path("/tmp/layer3")

    calls: list[tuple[Path, list[int]]] = []

    def fake_get(activations_path, neuron_ids, **_):
        calls.append((activations_path, list(neuron_ids)))
        return {nid: ["cat"] for nid in neuron_ids}

    mocker.patch("line.line.initialization.get_initial_concepts", side_effect=fake_get)
    mocker.patch.object(pipeline, "_generate_images", return_value=[MagicMock()] * 4)
    pipeline._explained_model = MagicMock()
    pipeline._explained_model.get_activations.return_value = {
        "layer4": torch.zeros(4, 5),
        "layer3": torch.zeros(4, 5),
    }
    pipeline._neuron_states = {
        ("layer4", 0): _make_neuron_state(0, "layer4"),
        ("layer4", 2): _make_neuron_state(2, "layer4"),
        ("layer3", 1): _make_neuron_state(1, "layer3"),
    }
    pipeline._model_layer_activations_paths = {"layer4": layer4_path, "layer3": layer3_path}

    pipeline._initialize_concept_histories()

    assert len(calls) == 2
    neuron_ids_by_path = {path: sorted(ids) for path, ids in calls}
    assert neuron_ids_by_path[layer4_path] == [0, 2]
    assert neuron_ids_by_path[layer3_path] == [1]


def test_initialize_concept_histories_multi_layer_single_image_batch(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    """All unique concepts across layers are generated in one _generate_images call."""
    mocker.patch(
        "line.line.initialization.get_initial_concepts",
        side_effect=[
            {0: ["cat", "dog"]},  # layer4
            {1: ["dog", "bird"]},  # layer3 — "dog" shared
        ],
    )
    mock_generate = mocker.patch.object(
        pipeline, "_generate_images", return_value=[MagicMock()] * 6
    )
    pipeline._explained_model = MagicMock()
    pipeline._explained_model.get_activations.return_value = {
        "layer4": torch.zeros(6, 5),
        "layer3": torch.zeros(6, 5),
    }
    pipeline._neuron_states = {
        ("layer4", 0): _make_neuron_state(0, "layer4"),
        ("layer3", 1): _make_neuron_state(1, "layer3"),
    }
    pipeline._model_layer_activations_paths = {
        "layer4": Path("/tmp/layer4"),
        "layer3": Path("/tmp/layer3"),
    }

    pipeline._initialize_concept_histories()

    mock_generate.assert_called_once()
    assert set(mock_generate.call_args.args[0]) == {"cat", "dog", "bird"}


def test_initialize_concept_histories_multi_layer_scores_use_correct_layer_activations(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    """Each neuron's score is sliced from its own layer's activation tensor,
    not the other layer's."""
    mocker.patch(
        "line.line.initialization.get_initial_concepts",
        side_effect=[{0: ["cat"]}, {1: ["dog"]}],
    )
    # 2 unique concepts, n_images=2 → layout [cat_g0, dog_g0, cat_g1, dog_g1]
    layer4_act = torch.zeros(4, 5)
    layer4_act[[0, 2], 0] = torch.tensor([0.3, 0.7])  # cat rows, neuron 0 → mean 0.5

    layer3_act = torch.zeros(4, 5)
    layer3_act[[1, 3], 1] = torch.tensor([0.1, 0.3])  # dog rows, neuron 1 → mean 0.2

    mocker.patch.object(pipeline, "_generate_images", return_value=[MagicMock()] * 4)
    pipeline._explained_model = MagicMock()
    pipeline._explained_model.get_activations.return_value = {
        "layer4": layer4_act,
        "layer3": layer3_act,
    }
    state0 = _make_neuron_state(0, "layer4")
    state1 = _make_neuron_state(1, "layer3")
    pipeline._neuron_states = {("layer4", 0): state0, ("layer3", 1): state1}
    pipeline._model_layer_activations_paths = {
        "layer4": Path("/tmp/layer4"),
        "layer3": Path("/tmp/layer3"),
    }

    pipeline._initialize_concept_histories()

    assert math.isclose(state0.concept_history["cat"], 0.5, rel_tol=1e-5)
    assert math.isclose(state1.concept_history["dog"], 0.2, rel_tol=1e-5)


def test_initialize_concept_histories_passes_initial_h_size_not_h_size(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    """_initialize_concept_histories must pass initial_h_size to get_initial_concepts,
    not h_size, so the two scoreboard sizes remain independent."""
    pipeline._concept_history_config = config.ConceptHistoryConfig(h_size=20, initial_h_size=4)
    mock_get = mocker.patch(
        "line.line.initialization.get_initial_concepts",
        return_value={},
    )
    pipeline._neuron_states = {("layer4", 0): _make_neuron_state(0)}

    pipeline._initialize_concept_histories()

    mock_get.assert_called_once()
    assert mock_get.call_args.kwargs["h_size"] == 4


# ---------------------------------------------------------------------------
# _run_iteration
# ---------------------------------------------------------------------------


def test_run_iteration_calls_llm_once_for_all_neurons(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    states = [_make_neuron_state(0), _make_neuron_state(1)]
    pipeline._lang_model = MagicMock()
    pipeline._lang_model.generate_concept.return_value = [("cat", "reason1"), ("dog", "reason2")]
    n_neurons, n_images = 2, 2
    n_total = n_neurons * n_images
    mocker.patch.object(pipeline, "_generate_images", return_value=[MagicMock()] * n_total)
    fake_act = {"layer4": torch.zeros(n_neurons * n_images, 5)}
    pipeline._explained_model = MagicMock()
    pipeline._explained_model.get_activations.return_value = fake_act
    mocker.patch("line.line.history.add_concept")
    mocker.patch("line.line.history.get_best_concept", return_value=("cat", 0.0))

    pipeline._run_iteration(states, iteration=1)

    pipeline._lang_model.generate_concept.assert_called_once_with(
        [states[0].concept_history, states[1].concept_history],
        [states[0].generation_history, states[1].generation_history],
        None,
    )


def test_run_iteration_calls_t2i_once_for_all_concepts(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    states = [_make_neuron_state(0), _make_neuron_state(1)]
    pipeline._lang_model = MagicMock()
    pipeline._lang_model.generate_concept.return_value = [("cat", ""), ("dog", "")]
    mock_generate = mocker.patch.object(
        pipeline, "_generate_images", return_value=[MagicMock()] * 4
    )
    fake_act = {"layer4": torch.zeros(4, 5)}
    pipeline._explained_model = MagicMock()
    pipeline._explained_model.get_activations.return_value = fake_act
    mocker.patch("line.line.history.add_concept")
    mocker.patch("line.line.history.get_best_concept", return_value=("cat", 0.0))

    pipeline._run_iteration(states, iteration=1)

    mock_generate.assert_called_once_with(["cat", "dog"])


def test_run_iteration_calls_explained_model_once(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    states = [_make_neuron_state(0), _make_neuron_state(1)]
    pipeline._lang_model = MagicMock()
    pipeline._lang_model.generate_concept.return_value = [("cat", ""), ("dog", "")]
    mocker.patch.object(pipeline, "_generate_images", return_value=[MagicMock()] * 4)
    pipeline._explained_model = MagicMock()
    pipeline._explained_model.get_activations.return_value = {"layer4": torch.zeros(4, 5)}
    mocker.patch("line.line.history.add_concept")
    mocker.patch("line.line.history.get_best_concept", return_value=("cat", 0.0))

    pipeline._run_iteration(states, iteration=1)

    pipeline._explained_model.get_activations.assert_called_once()


def test_run_iteration_updates_states_and_returns_results(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    state0 = _make_neuron_state(0)
    state1 = _make_neuron_state(1)
    pipeline._lang_model = MagicMock()
    pipeline._lang_model.generate_concept.return_value = [
        ("cat", "reason_cat"),
        ("dog", "reason_dog"),
    ]
    # 2 neurons, 2 images: layout [c0_g0, c1_g0, c0_g1, c1_g1]
    # neuron 0 → concept "cat" at indices [0, 2], activations col 0
    # neuron 1 → concept "dog" at indices [1, 3], activations col 1
    fake_act = torch.zeros(4, 5)
    fake_act[[0, 2], 0] = torch.tensor([0.4, 0.6])  # cat, neuron 0 → mean 0.5
    fake_act[[1, 3], 1] = torch.tensor([0.2, 0.4])  # dog, neuron 1 → mean 0.3
    mocker.patch.object(pipeline, "_generate_images", return_value=[MagicMock()] * 4)
    pipeline._explained_model = MagicMock()
    pipeline._explained_model.get_activations.return_value = {"layer4": fake_act}
    mocker.patch("line.line.history.add_concept")
    mocker.patch("line.line.history.get_best_concept", return_value=("cat", 0.5))

    results = pipeline._run_iteration([state0, state1], iteration=1)

    assert results[0][0] == "cat"
    assert math.isclose(results[0][1], 0.5, rel_tol=1e-5)
    assert results[1][0] == "dog"
    assert math.isclose(results[1][1], 0.3, rel_tol=1e-5)
    assert state0.reasoning == [{"concept": "cat", "reasoning": "reason_cat"}]
    assert state1.reasoning == [{"concept": "dog", "reasoning": "reason_dog"}]


def test_run_iteration_uses_summary_prompt_with_top_k(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    states = [_make_neuron_state(0)]
    pipeline._lang_model = MagicMock()
    pipeline._lang_model.generate_concept.return_value = [("cat", "")]
    mocker.patch.object(pipeline, "_generate_images", return_value=[MagicMock()] * 2)
    pipeline._explained_model = MagicMock()
    pipeline._explained_model.get_activations.return_value = {"layer4": torch.zeros(2, 5)}
    mocker.patch("line.line.history.add_concept")
    mocker.patch("line.line.history.get_best_concept", return_value=("cat", 0.0))

    pipeline._top_k = 3
    pipeline._run_iteration(states, iteration="summary")

    pipeline._lang_model.generate_concept.assert_called_once_with(
        [states[0].concept_history], [states[0].generation_history], 3
    )


def test_run_iteration_does_not_save_images_by_default(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    states = [_make_neuron_state(0)]
    pipeline._lang_model = MagicMock()
    pipeline._lang_model.generate_concept.return_value = [("cat", "")]
    mocker.patch.object(pipeline, "_generate_images", return_value=[MagicMock()] * 2)
    pipeline._explained_model = MagicMock()
    pipeline._explained_model.get_activations.return_value = {"layer4": torch.zeros(2, 5)}
    mocker.patch("line.line.history.add_concept")
    mocker.patch("line.line.history.get_best_concept", return_value=("cat", 0.0))
    mock_save = mocker.patch("line.line.history.save_images_from_iteration")

    pipeline._run_iteration(states, iteration=1)

    mock_save.assert_not_called()


def test_run_iteration_saves_images_per_neuron_when_enabled(
    load_config: config.LoadConfig,
    image_gen_config: config.ImageGenerationConfig,
    concept_history_config: config.ConceptHistoryConfig,
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
):
    mock_explained_model = MagicMock()
    mock_explained_model.hooked_layers = ["layer4"]
    run_config = _make_run_config(
        load_config,
        image_gen_config,
        concept_history_config,
        config.HistoryManagingConfig(
            save_images=True, save_histories=False, save_directory=str(tmp_path)
        ),
    )
    p = LINE(run_config, mock_explained_model, torch.device("cpu"))

    states = [_make_neuron_state(0), _make_neuron_state(1)]
    p._lang_model = MagicMock()
    p._lang_model.generate_concept.return_value = [("cat", ""), ("dog", "")]
    fake_images = [MagicMock() for _ in range(4)]
    mocker.patch.object(p, "_generate_images", return_value=fake_images)
    p._explained_model = MagicMock()
    p._explained_model.get_activations.return_value = {"layer4": torch.zeros(4, 5)}
    mocker.patch("line.line.history.add_concept")
    mocker.patch("line.line.history.get_best_concept", return_value=("cat", 0.0))
    mock_save = mocker.patch("line.line.history.save_images_from_iteration")

    p._run_iteration(states, iteration=2)

    # One save call per neuron
    assert mock_save.call_count == 2
    saved_concepts = {call.args[3] for call in mock_save.call_args_list}
    assert saved_concepts == {"cat", "dog"}


def test_run_iteration_multi_layer_uses_correct_layer_activations(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    """Each neuron's score is sliced from its own layer's activation tensor, not a shared one."""
    state0 = _make_neuron_state(0, "layer4")
    state1 = _make_neuron_state(1, "layer3")
    pipeline._lang_model = MagicMock()
    pipeline._lang_model.generate_concept.return_value = [("cat", ""), ("dog", "")]

    # n_neurons=2, n_images=2 → layout [n0_g0, n1_g0, n0_g1, n1_g1]
    # state0 indices: [0, 2], state1 indices: [1, 3]
    layer4_act = torch.zeros(4, 5)
    layer4_act[[0, 2], 0] = torch.tensor([0.4, 0.6])  # cat, neuron 0 → mean 0.5

    layer3_act = torch.zeros(4, 5)
    layer3_act[[1, 3], 1] = torch.tensor([0.1, 0.3])  # dog, neuron 1 → mean 0.2

    mocker.patch.object(pipeline, "_generate_images", return_value=[MagicMock()] * 4)
    pipeline._explained_model = MagicMock()
    pipeline._explained_model.get_activations.return_value = {
        "layer4": layer4_act,
        "layer3": layer3_act,
    }
    mocker.patch("line.line.history.add_concept")
    mocker.patch("line.line.history.get_best_concept", return_value=("cat", 0.0))

    results = pipeline._run_iteration([state0, state1], iteration=1)

    # If state1 incorrectly read layer4, indices [1,3] col 1 are all 0.0 → score 0.0, not 0.2
    assert math.isclose(results[0][1], 0.5, rel_tol=1e-5)
    assert math.isclose(results[1][1], 0.2, rel_tol=1e-5)


def test_run_iteration_activation_indices_with_three_neurons(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    """Index formula ``g * n_neurons + i`` is exercised with n_neurons=3, n_images=2.

    With 3 neurons the layout is [n0_g0, n1_g0, n2_g0, n0_g1, n1_g1, n2_g1].
    Indices per neuron:
      neuron 0 → [0, 3]   (g*3+0)
      neuron 1 → [1, 4]   (g*3+1)
      neuron 2 → [2, 5]   (g*3+2)

    Using n_neurons==2 masks bugs because n_neurons and n_images are equal.
    """
    state0 = _make_neuron_state(0)
    state1 = _make_neuron_state(1)
    state2 = _make_neuron_state(2)
    pipeline._lang_model = MagicMock()
    pipeline._lang_model.generate_concept.return_value = [
        ("cat", ""),
        ("dog", ""),
        ("bird", ""),
    ]

    # n_neurons=3, n_images=2 → 6 images total
    fake_act = torch.zeros(6, 5)
    fake_act[[0, 3], 0] = torch.tensor([0.2, 0.8])  # cat,  neuron 0 → mean 0.5
    fake_act[[1, 4], 1] = torch.tensor([0.1, 0.3])  # dog,  neuron 1 → mean 0.2
    fake_act[[2, 5], 2] = torch.tensor([0.6, 0.4])  # bird, neuron 2 → mean 0.5

    mocker.patch.object(pipeline, "_generate_images", return_value=[MagicMock()] * 6)
    pipeline._explained_model = MagicMock()
    pipeline._explained_model.get_activations.return_value = {"layer4": fake_act}
    mocker.patch("line.line.history.add_concept")
    mocker.patch("line.line.history.get_best_concept", return_value=("cat", 0.0))

    results = pipeline._run_iteration([state0, state1, state2], iteration=1)

    assert math.isclose(results[0][1], 0.5, rel_tol=1e-5)
    assert math.isclose(results[1][1], 0.2, rel_tol=1e-5)
    assert math.isclose(results[2][1], 0.5, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


def test_run_pipeline_calls_setup_once(pipeline: LINE, mocker: pytest_mock.MockerFixture):
    state0 = _make_neuron_state(0)
    state1 = _make_neuron_state(1)

    def fake_setup() -> None:
        pipeline._neuron_states = {
            ("layer4", 0): state0,
            ("layer4", 1): state1,
        }

    mock_setup = mocker.patch.object(pipeline, "_setup", side_effect=fake_setup)
    mocker.patch.object(pipeline, "_run_iteration", return_value=[("cat", 0.9), ("dog", 0.8)])
    mocker.patch("line.line.history.save_to_file")
    mocker.patch("line.line.history.get_best_concept", return_value=("cat", 0.9))

    pipeline._n_iters = 2
    pipeline.run_pipeline()

    mock_setup.assert_called_once()


def test_run_pipeline_calls_run_iteration_n_plus_one_times(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    """n_iters main iterations + 1 summary = n_iters+1 total _run_iteration calls."""
    state0 = _make_neuron_state(0)
    state1 = _make_neuron_state(1)

    def fake_setup() -> None:
        pipeline._neuron_states = {
            ("layer4", 0): state0,
            ("layer4", 1): state1,
        }

    mocker.patch.object(pipeline, "_setup", side_effect=fake_setup)
    mock_iter = mocker.patch.object(
        pipeline, "_run_iteration", return_value=[("cat", 0.9), ("dog", 0.8)]
    )
    mocker.patch("line.line.history.save_to_file")
    mocker.patch("line.line.history.get_best_concept", return_value=("cat", 0.9))

    pipeline._n_iters = 3
    pipeline.run_pipeline()

    assert mock_iter.call_count == 4


def test_run_pipeline_summary_uses_top_k_config(pipeline: LINE, mocker: pytest_mock.MockerFixture):
    state0 = _make_neuron_state(0)

    def fake_setup() -> None:
        pipeline._neuron_states = {("layer4", 0): state0}

    mocker.patch.object(pipeline, "_setup", side_effect=fake_setup)
    mock_iter = mocker.patch.object(pipeline, "_run_iteration", return_value=[("cat", 0.9)])
    mocker.patch("line.line.history.save_to_file")
    mocker.patch("line.line.history.get_best_concept", return_value=("cat", 0.9))

    pipeline._n_iters = 2
    pipeline._top_k = 5
    pipeline.run_pipeline()

    assert pipeline._top_k == 5
    last_call = mock_iter.call_args_list[-1]
    assert last_call.kwargs.get("iteration") == "summary"


def test_run_pipeline_returns_best_concept_per_neuron(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    state0 = _make_neuron_state(0)
    state1 = _make_neuron_state(1)

    def fake_setup() -> None:
        pipeline._neuron_states = {
            ("layer4", 0): state0,
            ("layer4", 1): state1,
        }

    mocker.patch.object(pipeline, "_setup", side_effect=fake_setup)
    mocker.patch.object(pipeline, "_run_iteration", return_value=[("cat", 0.9), ("dog", 0.8)])
    mocker.patch("line.line.history.save_to_file")
    state0.concept_history = {"cat": 0.99}
    state1.concept_history = {"dog": 0.95}

    pipeline._n_iters = 1
    result = pipeline.run_pipeline()

    assert result[("layer4", 0)] == "cat"
    assert result[("layer4", 1)] == "dog"


# ---------------------------------------------------------------------------
# NeuronState
# ---------------------------------------------------------------------------


def test_neuron_state_initial_accumulators_are_empty():
    state = NeuronState(
        neuron_id=5,
        layer="layer4",
        model_id="test-expl",
        save_directory=Path("/tmp/n5"),
        h_size=0,
    )

    assert state.neuron_id == 5
    assert state.concept_history == {}
    assert state.generation_history == []
    assert state.reasoning == []
    assert state.best_concepts == []


# ---------------------------------------------------------------------------
# LINE.__init__ path construction
# ---------------------------------------------------------------------------


def test_pipeline_model_layer_activations_paths(tmp_path: Path):
    mock_explained_model = MagicMock()
    mock_explained_model.hooked_layers = ["layer4"]
    run_config = RunConfig(
        load_config=config.LoadConfig(
            language_model_id="lm",
            text_to_image_model_id="t2i",
            explained_model_id="resnet50",
            language_model_kwargs={},
            text_to_image_model_kwargs={},
            explained_model_kwargs={},
            model_swapping=False,
        ),
        image_generation_config=config.ImageGenerationConfig(n_images=1, prompt_text=""),
        concept_history_config=config.ConceptHistoryConfig(),
        history_managing_config=config.HistoryManagingConfig(
            save_images=False, save_histories=False, save_directory=str(tmp_path)
        ),
        n_iters=1,
        neurons={"layer4": [0]},
        initializations_path="/data/activations",
    )
    p = LINE(run_config, mock_explained_model, torch.device("cpu"))

    assert p._model_layer_activations_paths["layer4"] == Path("/data/activations/resnet50/layer4")


def test_pipeline_init_saves_run_config_params(tmp_path: Path):
    """
    LINE.__init__ persists the full RunConfig fields plus run_id
    and model_layer_activations_paths.
    """
    from line.pipeline.history import read_params

    mock_explained_model = MagicMock()
    mock_explained_model.hooked_layers = ["layer4"]
    run_config = RunConfig(
        load_config=config.LoadConfig(
            language_model_id="lm",
            text_to_image_model_id="t2i",
            explained_model_id="resnet50",
            language_model_kwargs={},
            text_to_image_model_kwargs={},
            explained_model_kwargs={},
            model_swapping=False,
        ),
        image_generation_config=config.ImageGenerationConfig(n_images=1, prompt_text=""),
        concept_history_config=config.ConceptHistoryConfig(),
        history_managing_config=config.HistoryManagingConfig(
            save_images=False, save_histories=False, save_directory=str(tmp_path)
        ),
        n_iters=2,
        neurons={"layer4": [0]},
        initializations_path="/data/activations",
        metric="AVG_ACTIVATION",
        top_k=5,
    )
    p = LINE(run_config, mock_explained_model, torch.device("cpu"))

    saved = read_params(p.save_directory)

    expected = {
        "run_id": p._run_id,
        **dataclasses.asdict(run_config),
        "model_layer_activations_paths": {
            layer: str(path) for layer, path in p._model_layer_activations_paths.items()
        },
    }
    assert saved == expected


@pytest.mark.parametrize("seed", [42, 123, None])
def test_pipeline_calls_set_random_seed_on_run(
    pipeline: LINE,
    mocker: pytest_mock.MockerFixture,
    seed: int | None,
):
    state0 = _make_neuron_state(0)
    state1 = _make_neuron_state(1)

    def fake_setup() -> None:
        pipeline._neuron_states = {("layer4", 0): state0, ("layer4", 1): state1}

    mocker.patch.object(pipeline, "_setup", side_effect=fake_setup)
    mocker.patch.object(pipeline, "_run_iteration", return_value=[("cat", 0.9), ("dog", 0.8)])
    mocker.patch("line.line.history.save_to_file")
    mocker.patch("line.line.history.get_best_concept", return_value=("cat", 0.9))
    mock_seed = mocker.patch("line.line.set_random_seed")

    pipeline.run_pipeline(seed=seed)

    if seed is None:
        mock_seed.assert_not_called()
    else:
        mock_seed.assert_called_once_with(seed)


# ---------------------------------------------------------------------------
# _initialize_concept_histories — early return on empty concepts
# ---------------------------------------------------------------------------


def test_initialize_concept_histories_early_return_when_no_concepts(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    """If every neuron returns an empty concept list, _generate_images must not be called.

    The early ``if not unique_concepts: return`` guard must prevent a zero-image
    T2I call that would crash downstream index arithmetic.
    """
    mocker.patch(
        "line.line.initialization.get_initial_concepts",
        return_value={0: []},
    )
    mock_generate = mocker.patch.object(pipeline, "_generate_images")
    pipeline._explained_model = MagicMock()
    pipeline._neuron_states = {("layer4", 0): _make_neuron_state(0)}

    pipeline._initialize_concept_histories()

    mock_generate.assert_not_called()
    pipeline._explained_model.get_activations.assert_not_called()


# ---------------------------------------------------------------------------
# _save_neuron_histories
# ---------------------------------------------------------------------------


def test_save_neuron_histories_writes_generation_and_final_files(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    """_save_neuron_histories must call save_to_file for both history files."""
    mock_save = mocker.patch("line.line.history.save_to_file")
    state = _make_neuron_state(0)
    state.generation_history = [("cat", 0.5), ("dog", 0.8)]
    state.concept_history = {"dog": 0.8}

    pipeline._save_neuron_histories(state)

    assert mock_save.call_count == 2
    saved_filenames = {call.args[2] for call in mock_save.call_args_list}
    assert "generation_history.jsonl" in saved_filenames
    assert "final_concept_history.jsonl" in saved_filenames


# ---------------------------------------------------------------------------
# run_pipeline — save_histories flag
# ---------------------------------------------------------------------------


def test_run_pipeline_calls_save_neuron_histories_when_flag_enabled(
    load_config: config.LoadConfig,
    image_gen_config: config.ImageGenerationConfig,
    concept_history_config: config.ConceptHistoryConfig,
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
):
    """When save_histories=True, _save_neuron_histories is called once per neuron."""
    mocker.patch("line.line.history.save_to_file")
    mock_explained_model = MagicMock()
    mock_explained_model.hooked_layers = ["layer4"]

    run_config = _make_run_config(
        load_config,
        image_gen_config,
        concept_history_config,
        config.HistoryManagingConfig(
            save_images=False, save_histories=True, save_directory=str(tmp_path)
        ),
        neurons={"layer4": [0, 1]},
        n_iters=1,
    )
    p = LINE(run_config, mock_explained_model, torch.device("cpu"))

    state0 = _make_neuron_state(0)
    state1 = _make_neuron_state(1)

    def fake_setup() -> None:
        p._neuron_states = {("layer4", 0): state0, ("layer4", 1): state1}

    mocker.patch.object(p, "_setup", side_effect=fake_setup)
    mocker.patch.object(p, "_run_iteration", return_value=[("cat", 0.9), ("dog", 0.8)])
    mock_save_histories = mocker.patch.object(p, "_save_neuron_histories")
    mocker.patch("line.line.history.get_best_concept", return_value=("cat", 0.9))

    p.run_pipeline()

    assert mock_save_histories.call_count == 2
    called_states = [call.args[0] for call in mock_save_histories.call_args_list]
    assert state0 in called_states
    assert state1 in called_states


def test_run_pipeline_does_not_call_save_neuron_histories_when_flag_disabled(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    """When save_histories=False (the default fixture), _save_neuron_histories is never called."""
    state0 = _make_neuron_state(0)
    state1 = _make_neuron_state(1)

    def fake_setup() -> None:
        pipeline._neuron_states = {("layer4", 0): state0, ("layer4", 1): state1}

    mocker.patch.object(pipeline, "_setup", side_effect=fake_setup)
    mocker.patch.object(pipeline, "_run_iteration", return_value=[("cat", 0.9), ("dog", 0.8)])
    mocker.patch("line.line.history.save_to_file")
    mocker.patch("line.line.history.get_best_concept", return_value=("cat", 0.9))
    mock_save_histories = mocker.patch.object(pipeline, "_save_neuron_histories")

    pipeline._n_iters = 1
    pipeline.run_pipeline()

    mock_save_histories.assert_not_called()


# ---------------------------------------------------------------------------
# JSONL output files
# ---------------------------------------------------------------------------


def test_run_pipeline_saves_reasoning_and_best_concepts_as_jsonl(
    pipeline: LINE, mocker: pytest_mock.MockerFixture
):
    """Reasoning and best concepts are written under their .jsonl names."""
    state0 = _make_neuron_state(0)

    def fake_setup() -> None:
        pipeline._neuron_states = {("layer4", 0): state0}

    mocker.patch.object(pipeline, "_setup", side_effect=fake_setup)
    mocker.patch.object(pipeline, "_run_iteration", return_value=[("cat", 0.9)])
    mock_save = mocker.patch("line.line.history.save_to_file")

    pipeline._n_iters = 1
    pipeline.run_pipeline()

    saved_filenames = {call.args[2] for call in mock_save.call_args_list}
    assert saved_filenames == {"reasoning.jsonl", "best_concepts.jsonl"}


def test_save_neuron_histories_round_trips_through_readers(pipeline: LINE, tmp_path: Path):
    """Files written by _save_neuron_histories read back unchanged, one record per line."""
    state = _make_neuron_state(0)
    state.save_directory = tmp_path
    state.initial_concepts = ["cat"]
    state.concept_history = {"cat": 0.9, "keypad\n alt, 2": 0.4, "dog": 0.7}
    state.generation_history = [("keypad\n alt, 2", 0.4), ("dog", 0.7), ("dog", 0.7), ("cat", 0.9)]

    pipeline._n_iters = 3
    pipeline._save_neuron_histories(state)

    assert history.read_generation_history(tmp_path) == [
        {"iteration": 1, "concept": "keypad\n alt, 2", "score": 0.4},
        {"iteration": 2, "concept": "dog", "score": 0.7},
        {"iteration": 3, "concept": "dog", "score": 0.7},
        {"iteration": "summary", "concept": "cat", "score": 0.9},
    ]
    assert history.read_final_concept_history(tmp_path) == [
        {"concept": "cat", "score": 0.9, "first_iteration": 0, "is_summary": True},
        {"concept": "dog", "score": 0.7, "first_iteration": 2, "is_summary": False},
        {"concept": "keypad\n alt, 2", "score": 0.4, "first_iteration": 1, "is_summary": False},
    ]
    final_file = tmp_path / history.FINAL_CONCEPT_HISTORY_FILENAME
    assert len(final_file.read_text().splitlines()) == 3
