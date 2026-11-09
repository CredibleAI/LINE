import random
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pytest_mock
import torch

from line.pipeline import initialization, scoring

# ---------------------------------------------------------------------------
# _load_saved_activations
# ---------------------------------------------------------------------------


def test_load_saved_activations(mocker: pytest_mock.MockerFixture):
    fake_paths = [Path("concept_1.pt"), Path("concept_2.pt"), Path("concept_3.pt")]
    mocker.patch("pathlib.Path.iterdir", return_value=iter(fake_paths))
    test_tensor = torch.tensor([1, 2, 3])
    mocker.patch("torch.load", return_value=test_tensor)

    result = list(initialization._load_saved_activations(Path("fake/path")))
    assert result == [
        ("concept 1", test_tensor),
        ("concept 2", test_tensor),
        ("concept 3", test_tensor),
    ]


# ---------------------------------------------------------------------------
# _select_best_concepts
# ---------------------------------------------------------------------------


def test_select_best_concepts():
    scores = {"a": 10.0, "b": 3.0, "c": 12.0, "d": 1.0}
    assert initialization._select_best_concepts(scores, 2) == ["c", "a"]


def test_select_best_concepts_n_exceeds_length():
    scores = {"a": 10.0, "b": 3.0}
    assert set(initialization._select_best_concepts(scores, 5)) == {"a", "b"}


# ---------------------------------------------------------------------------
# get_initial_concepts
# ---------------------------------------------------------------------------


def _fake_saved(mocker: pytest_mock.MockerFixture, concept_acts: dict[str, list]) -> None:
    """Patch ensure_concept_scores with pre-computed scores from concept → row-list activations."""
    scores = {
        name: torch.tensor(rows, dtype=torch.float).mean(dim=0)
        for name, rows in concept_acts.items()
    }
    mocker.patch(
        "line.pipeline.initialization.ensure_concept_scores",
        return_value=scores,
    )


def test_get_initial_concepts_unbounded_returns_all(mocker: pytest_mock.MockerFixture):
    """h_size=0 returns all concepts for each neuron."""
    # 3 concepts, 2 images each, 4 model neurons
    _fake_saved(
        mocker,
        {
            "cat": [[0.9, 0.1, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0]],
            "dog": [[0.1, 0.7, 0.0, 0.0], [0.2, 0.6, 0.0, 0.0]],
            "car": [[0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.4, 0.0]],
        },
    )

    result = initialization.get_initial_concepts(
        Path("fake"), neuron_ids=[0, 1], metric=scoring.Metric.AVG_ACTIVATION, h_size=0
    )

    assert set(result[0]) == {"cat", "dog", "car"}
    assert set(result[1]) == {"cat", "dog", "car"}


def test_get_initial_concepts_bounded_respects_h_size(mocker: pytest_mock.MockerFixture):
    """h_size limits total with best_ratio split (2 best + 2 random from 6)."""
    acts = {f"c{i}": [[float(i), 0.0]] for i in range(6)}
    _fake_saved(mocker, acts)

    result = initialization.get_initial_concepts(
        Path("fake"),
        neuron_ids=[0],
        metric=scoring.Metric.AVG_ACTIVATION,
        h_size=4,
        best_ratio=0.5,
    )

    assert len(result[0]) == 4
    assert "c5" in result[0]  # top-1 by avg activation for neuron 0
    assert "c4" in result[0]  # top-2


def test_get_initial_concepts_pool_ratio_restricts_shared_bank(
    mocker: pytest_mock.MockerFixture,
):
    """pool_ratio<1 draws one restricted bank shared across neurons."""
    acts = {f"c{i}": [[float(i), float(i)]] for i in range(6)}
    _fake_saved(mocker, acts)

    random.seed(0)
    result = initialization.get_initial_concepts(
        Path("fake"),
        neuron_ids=[0, 1],
        metric=scoring.Metric.AVG_ACTIVATION,
        h_size=0,  # return the whole (restricted) pool per neuron
        pool_ratio=0.5,  # 3 of 6 concepts
    )

    assert len(result[0]) == 3
    # The restricted bank is drawn once and shared across all neurons.
    assert set(result[0]) == set(result[1])
    assert set(result[0]).issubset(set(acts))


def test_get_initial_concepts_pool_ratio_full_returns_all(
    mocker: pytest_mock.MockerFixture,
):
    """pool_ratio=1.0 leaves the full bank available (baseline unchanged)."""
    acts = {f"c{i}": [[float(i), float(i)]] for i in range(6)}
    _fake_saved(mocker, acts)

    result = initialization.get_initial_concepts(
        Path("fake"),
        neuron_ids=[0],
        metric=scoring.Metric.AVG_ACTIVATION,
        h_size=0,
        pool_ratio=1.0,
    )

    assert set(result[0]) == set(acts)


def test_get_initial_concepts_loads_scores_once_for_multiple_neurons(
    mocker: pytest_mock.MockerFixture,
):
    """ensure_concept_scores is called once regardless of how many neuron_ids."""
    _fake_saved(mocker, {"cat": [[0.5, 0.3]], "dog": [[0.2, 0.8]]})
    mock_ensure = mocker.patch(
        "line.pipeline.initialization.ensure_concept_scores",
        return_value={
            "cat": torch.tensor([0.5, 0.3]),
            "dog": torch.tensor([0.2, 0.8]),
        },
    )

    initialization.get_initial_concepts(
        Path("fake"), neuron_ids=[0, 1], metric=scoring.Metric.AVG_ACTIVATION, h_size=0
    )

    mock_ensure.assert_called_once()


# ---------------------------------------------------------------------------
# compute_control_activations
# ---------------------------------------------------------------------------


def test_compute_control_activations_saves_all_layers_in_one_pass(
    mocker: pytest_mock.MockerFixture, tmp_path: Path
):
    """model.get_activations is called once per concept, saving all layers."""
    concept_dir = tmp_path / "imagenet"
    (concept_dir / "cat").mkdir(parents=True)
    img_path = concept_dir / "cat" / "img.jpg"
    img_path.write_bytes(b"")  # placeholder

    mocker.patch(
        "line.pipeline.initialization._load_concept_images",
        return_value=[MagicMock()],
    )
    fake_act = {
        "layer1": torch.zeros(1, 4),
        "layer2": torch.ones(1, 4),
    }
    mock_model = MagicMock()
    mock_model.get_activations.return_value = fake_act

    layer1_path = tmp_path / "layer1"
    layer2_path = tmp_path / "layer2"
    initialization.compute_control_activations(
        concept_dir,
        activations_paths={"layer1": layer1_path, "layer2": layer2_path},
        model=mock_model,
    )

    # Only one forward pass for the one concept
    mock_model.get_activations.assert_called_once()
    # Both layers saved
    assert torch.equal(torch.load(layer1_path / "cat.pt"), fake_act["layer1"])
    assert torch.equal(torch.load(layer2_path / "cat.pt"), fake_act["layer2"])


def test_compute_control_activations_interrupted_leaves_no_partial_cache(
    mocker: pytest_mock.MockerFixture, tmp_path: Path
):
    """A run killed midway must not leave a non-empty layer directory behind.

    ensure_control_activations treats any non-empty directory as complete, so
    partial output would be reused as if it were the full control set.
    """
    mocker.patch(
        "line.pipeline.initialization._get_concept_directories_and_names",
        return_value=[(Path("cat_dir"), "cat"), (Path("dog_dir"), "dog")],
    )
    mocker.patch("line.pipeline.initialization._load_concept_images", return_value=[MagicMock()])
    mock_model = MagicMock()
    mock_model.get_activations.side_effect = [{"layer1": torch.zeros(1, 4)}, RuntimeError("killed")]
    layer_path = tmp_path / "model" / "layer1"

    with pytest.raises(RuntimeError, match="killed"):
        initialization.compute_control_activations(
            Path("imagenet"), activations_paths={"layer1": layer_path}, model=mock_model
        )

    assert not layer_path.exists()
    assert list(layer_path.parent.iterdir()) == []  # staging directory cleaned up


def test_compute_control_activations_replaces_existing_empty_dir(
    mocker: pytest_mock.MockerFixture, tmp_path: Path
):
    mocker.patch(
        "line.pipeline.initialization._get_concept_directories_and_names",
        return_value=[(Path("cat_dir"), "cat")],
    )
    mocker.patch("line.pipeline.initialization._load_concept_images", return_value=[MagicMock()])
    mock_model = MagicMock()
    mock_model.get_activations.return_value = {"layer1": torch.zeros(1, 4)}
    layer_path = tmp_path / "layer1"
    layer_path.mkdir()

    initialization.compute_control_activations(
        Path("imagenet"), activations_paths={"layer1": layer_path}, model=mock_model
    )

    assert [path.name for path in layer_path.iterdir()] == ["cat.pt"]


def test_staged_directories_concurrent_calls_do_not_share_staging(tmp_path: Path):
    """Overlapping calls in one process must not reuse or clobber each other's staging dir.

    The first to finish commits; the other fails instead of mixing its files in.
    """
    layer_path = tmp_path / "layer1"

    with pytest.raises(OSError), initialization._staged_directories([layer_path]) as (outer,):
        (outer / "outer.pt").touch()
        with initialization._staged_directories([layer_path]) as (inner,):
            assert inner != outer
            (inner / "inner.pt").touch()
        assert [path.name for path in outer.iterdir()] == ["outer.pt"]

    assert [path.name for path in layer_path.iterdir()] == ["inner.pt"]
    assert [path.name for path in tmp_path.iterdir()] == ["layer1"]  # staging dirs cleaned up


# ---------------------------------------------------------------------------
# ensure_control_activations
# ---------------------------------------------------------------------------


def test_ensure_control_activations_skips_if_all_present(
    mocker: pytest_mock.MockerFixture, tmp_path: Path
):
    layer1_path = tmp_path / "layer1"
    layer1_path.mkdir()
    (layer1_path / "cat.pt").write_bytes(b"")
    layer2_path = tmp_path / "layer2"
    layer2_path.mkdir()
    (layer2_path / "cat.pt").write_bytes(b"")

    mock_compute = mocker.patch("line.pipeline.initialization.compute_control_activations")

    initialization.ensure_control_activations(
        imagenet_data_path=None,
        activations_paths={"layer1": layer1_path, "layer2": layer2_path},
        model=MagicMock(),
    )

    mock_compute.assert_not_called()


def test_ensure_control_activations_computes_only_missing_layers(
    mocker: pytest_mock.MockerFixture, tmp_path: Path
):
    layer1_path = tmp_path / "layer1"
    layer1_path.mkdir()
    (layer1_path / "cat.pt").write_bytes(b"")  # layer1 already present

    layer2_path = tmp_path / "layer2"  # layer2 missing

    mock_compute = mocker.patch("line.pipeline.initialization.compute_control_activations")
    imagenet_path = tmp_path / "imagenet"
    imagenet_path.mkdir()

    initialization.ensure_control_activations(
        imagenet_data_path=imagenet_path,
        activations_paths={"layer1": layer1_path, "layer2": layer2_path},
        model=MagicMock(),
    )

    mock_compute.assert_called_once()
    _, passed_paths, _ = mock_compute.call_args.args
    assert "layer2" in passed_paths
    assert "layer1" not in passed_paths


def test_ensure_control_activations_raises_if_missing_and_no_imagenet(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="layer1"):
        initialization.ensure_control_activations(
            imagenet_data_path=None,
            activations_paths={"layer1": tmp_path / "layer1"},
            model=MagicMock(),
        )


# ---------------------------------------------------------------------------
# _load_saved_activations — naming convention edge cases
# ---------------------------------------------------------------------------


def test_load_saved_activations_replaces_all_underscores_with_spaces(
    mocker: pytest_mock.MockerFixture,
):
    """_load_saved_activations converts ALL underscores to spaces in concept names.

    ImageNet directory names use underscores as word separators
    (e.g. 'golden_retriever', 'red_blue_car').  The replacement is global, so
    callers must be aware that a stem like 'red_blue_car' becomes 'red blue car'.
    """
    fake_paths = [Path("golden_retriever.pt"), Path("red_blue_car.pt")]
    mocker.patch("pathlib.Path.iterdir", return_value=iter(fake_paths))
    mocker.patch("torch.load", return_value=torch.zeros(1))

    result = dict(initialization._load_saved_activations(Path("fake")))

    assert "golden retriever" in result
    assert "red blue car" in result
    assert "golden_retriever" not in result
    assert "red_blue_car" not in result


# ---------------------------------------------------------------------------
# ensure_control_activations — directory exists but is empty
# ---------------------------------------------------------------------------


def test_ensure_control_activations_treats_existing_empty_dir_as_missing(
    mocker: pytest_mock.MockerFixture, tmp_path: Path
):
    """A directory that exists but contains no .pt files should trigger recomputation.

    ensure_control_activations uses ``any(path.iterdir())`` to detect whether
    activations are present; an empty directory must be treated as missing.
    """
    empty_layer = tmp_path / "layer1"
    empty_layer.mkdir()  # exists but has no files

    imagenet_path = tmp_path / "imagenet"
    imagenet_path.mkdir()
    mock_compute = mocker.patch("line.pipeline.initialization.compute_control_activations")

    initialization.ensure_control_activations(
        imagenet_data_path=imagenet_path,
        activations_paths={"layer1": empty_layer},
        model=MagicMock(),
    )

    mock_compute.assert_called_once()
    _, passed_paths, _ = mock_compute.call_args.args
    assert "layer1" in passed_paths


# ---------------------------------------------------------------------------
# get_initial_concepts — single neuron edge case
# ---------------------------------------------------------------------------


def test_get_initial_concepts_single_neuron_bounded(mocker: pytest_mock.MockerFixture):
    """h_size=1 with best_ratio=1.0 returns only the single best concept."""
    acts = {"cat": [[0.9, 0.0]], "dog": [[0.2, 0.0]], "bird": [[0.5, 0.0]]}
    _fake_saved(mocker, acts)

    result = initialization.get_initial_concepts(
        Path("fake"),
        neuron_ids=[0],
        metric=scoring.Metric.AVG_ACTIVATION,
        h_size=1,
        best_ratio=1.0,
    )

    assert len(result[0]) == 1
    assert result[0][0] == "cat"


# ---------------------------------------------------------------------------
# _load_saved_activations — scores file filtering
# ---------------------------------------------------------------------------


def test_load_saved_activations_skips_scores_files(mocker: pytest_mock.MockerFixture):
    """scores_*.pt files are excluded from the activation iterator."""
    fake_paths = [
        Path("cat.pt"),
        Path("scores_AVG_ACTIVATION.pt"),
        Path("dog.pt"),
    ]
    mocker.patch("pathlib.Path.iterdir", return_value=iter(fake_paths))
    mocker.patch("torch.load", return_value=torch.zeros(1))

    result = list(initialization._load_saved_activations(Path("fake")))

    names = [name for name, _ in result]
    assert names == ["cat", "dog"]
    assert "scores AVG ACTIVATION" not in names


# ---------------------------------------------------------------------------
# _compute_all_scores
# ---------------------------------------------------------------------------


def test_compute_all_scores_shape_and_values(mocker: pytest_mock.MockerFixture):
    """Returns a tensor of shape [n_neurons] per concept with correct scores."""
    fake = [
        ("cat", torch.tensor([[1.0, 0.0], [3.0, 0.0]])),  # neuron 0 mean=2.0, neuron 1 mean=0.0
        ("dog", torch.tensor([[0.5, 1.0]])),  # neuron 0 mean=0.5, neuron 1 mean=1.0
    ]
    mocker.patch(
        "line.pipeline.initialization._load_saved_activations",
        return_value=iter(fake),
    )

    result = initialization._compute_all_scores(Path("fake"), scoring.Metric.AVG_ACTIVATION)

    assert set(result.keys()) == {"cat", "dog"}
    assert result["cat"].shape == (2,)
    assert result["dog"].shape == (2,)
    assert result["cat"][0].item() == pytest.approx(2.0)
    assert result["dog"][1].item() == pytest.approx(1.0)


def test_compute_all_scores_uses_metric(mocker: pytest_mock.MockerFixture):
    """calculate_metric is called once per (concept, neuron) pair."""
    fake = [("cat", torch.tensor([[0.5, 0.3]]))]
    mocker.patch(
        "line.pipeline.initialization._load_saved_activations",
        return_value=iter(fake),
    )
    mock_metric = mocker.patch(
        "line.pipeline.initialization.calculate_metric",
        side_effect=[0.5, 0.3],
    )

    initialization._compute_all_scores(Path("fake"), scoring.Metric.AVG_ACTIVATION)

    assert mock_metric.call_count == 2
    for call in mock_metric.call_args_list:
        tensor_arg, metric_arg = call.args
        assert tensor_arg.ndim == 1
        assert metric_arg == scoring.Metric.AVG_ACTIVATION


# ---------------------------------------------------------------------------
# ensure_concept_scores
# ---------------------------------------------------------------------------


def test_ensure_concept_scores_computes_and_saves_on_cache_miss(
    mocker: pytest_mock.MockerFixture, tmp_path: Path
):
    """On first call, scores are computed and saved to disk."""
    activations_path = tmp_path / "layer1"
    activations_path.mkdir()

    expected = {"cat": torch.tensor([0.9, 0.1])}
    mock_compute = mocker.patch(
        "line.pipeline.initialization._compute_all_scores",
        return_value=expected,
    )

    result = initialization.ensure_concept_scores(activations_path, scoring.Metric.AVG_ACTIVATION)

    mock_compute.assert_called_once_with(activations_path, scoring.Metric.AVG_ACTIVATION)
    assert result is expected
    scores_path = activations_path / "scores_AVG_ACTIVATION.pt"
    assert list(activations_path.iterdir()) == [scores_path]  # no staging file left behind
    assert torch.equal(torch.load(scores_path)["cat"], expected["cat"])


def test_ensure_concept_scores_loads_from_cache_on_hit(
    mocker: pytest_mock.MockerFixture, tmp_path: Path
):
    """When the scores file covers the saved concepts, it is loaded and compute is skipped."""
    activations_path = tmp_path / "layer1"
    activations_path.mkdir()
    (activations_path / "cat.pt").write_bytes(b"")
    scores_path = activations_path / "scores_AVG_ACTIVATION.pt"
    scores_path.write_bytes(b"")  # file exists

    cached = {"cat": torch.tensor([0.9, 0.1])}
    mocker.patch("torch.load", return_value=cached)
    mock_compute = mocker.patch("line.pipeline.initialization._compute_all_scores")

    result = initialization.ensure_concept_scores(activations_path, scoring.Metric.AVG_ACTIVATION)

    mock_compute.assert_not_called()
    assert result is cached


@pytest.mark.parametrize(
    "cached_concepts",
    [["cat"], ["cat", "dog", "bird"]],
    ids=["missing_concept", "stale_concept"],
)
def test_ensure_concept_scores_rejects_cache_not_matching_saved_concepts(
    tmp_path: Path, cached_concepts: list[str]
):
    """A cache that differs from the saved concepts raises and is left untouched.

    A run that scored the directory while another run was still writing the
    activations cached only the concepts present at that time; reusing it would
    restrict the initial pool of every later run.
    """
    activations_path = tmp_path / "layer1"
    activations_path.mkdir()
    torch.save(torch.tensor([[1.0, 2.0]]), activations_path / "cat.pt")
    torch.save(torch.tensor([[3.0, 4.0]]), activations_path / "dog.pt")
    scores_path = activations_path / "scores_AVG_ACTIVATION.pt"
    torch.save({name: torch.zeros(2) for name in cached_concepts}, scores_path)
    cache_bytes = scores_path.read_bytes()

    with pytest.raises(ValueError, match="do not match the saved control concepts"):
        initialization.ensure_concept_scores(activations_path, scoring.Metric.AVG_ACTIVATION)

    assert scores_path.read_bytes() == cache_bytes
