"""Concept scoreboard initialization for the LINE pipeline.

Handles computing (if necessary) and loading control concept activations from
disk, and selecting the initial set of concepts (top-scoring + random) to seed
the scoreboard H before iterative refinement begins.
"""

import contextlib
import logging
import os
import random
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from line.models.base import ExplainedModel
from line.pipeline.scoring import Metric, calculate_metric


def get_model_layer_activations_path(base_path: Path, model_id: str, layer: str) -> Path:
    """Construct the path where activations for a given model and layer are stored.

    Args:
        base_path: Root directory for all control activations.
        model_id: Identifier of the explained model (e.g., 'resnet18').
        layer: Name of the layer (e.g., 'avgpool').

    Returns:
        Path of the form ``base_path / model_id / layer``.
    """
    return base_path / model_id / layer


def _get_concept_directories_and_names(
    imagenet_data_path: Path,
) -> Iterator[tuple[Path, str]]:
    """Iterate over concept subdirectories and their names.

    Args:
        imagenet_data_path: ImageNet-style directory containing per-concept subdirectories.

    Yields:
        Tuples of (directory_path, concept_name) for each concept.
    """
    for concept_dir in imagenet_data_path.iterdir():
        if concept_dir.is_dir():
            yield concept_dir, concept_dir.name


def _load_concept_images(concept_dir: Path) -> list[Image.Image]:
    """Load all images from a concept directory as PIL images.

    Args:
        concept_dir: Directory containing images for one concept.

    Returns:
        List of PIL images loaded from the directory.
    """
    images = []
    for image_path in concept_dir.iterdir():
        if image_path.is_file():
            with Image.open(image_path) as pil_image:
                images.append(pil_image.convert("RGB"))
        else:
            logging.warning(f"Skipping non-file entry in concept directory: {image_path}")
    return images


@contextlib.contextmanager
def _staged_directories(final_paths: list[Path]) -> Iterator[list[Path]]:
    """Yield empty staging directories that replace ``final_paths`` once the block completes.

    Args:
        final_paths: Directories to populate. Each must be missing or empty.

    Yields:
        Staging directories, in the order of ``final_paths``.
    """
    staging_paths: list[Path] = []
    try:
        for path in final_paths:
            staging_path = path.with_name(f".staging_{path.name}-{uuid.uuid4()}")
            staging_path.mkdir(parents=True)
            staging_paths.append(staging_path)
        yield staging_paths
        for staging_path, final_path in zip(staging_paths, final_paths, strict=True):
            if final_path.exists():
                final_path.rmdir()  # Fails rather than overwrite a non-empty directory.
            os.replace(staging_path, final_path)
    finally:
        for staging_path in staging_paths:
            shutil.rmtree(staging_path, ignore_errors=True)


def compute_control_activations(
    imagenet_data_path: Path,
    activations_paths: dict[str, Path],
    model: ExplainedModel,
    batch_size: int = 0,
) -> None:
    """Compute and save control concept activations.

    For each concept subdirectory, runs a single forward pass through the model
    (which returns all hooked layers simultaneously) and saves each layer's
    activation tensor to its respective directory. Each directory appears only
    once all concepts are computed.

    Args:
        imagenet_data_path: ImageNet-style directory with per-concept image subdirectories.
        activations_paths: Mapping from layer name to the directory where its
            activation ``.pt`` files will be written.
        model: The explained model to extract activations from.
        batch_size: Number of images per forward pass. 0 means process all at once.
    """
    logging.info(f"Computing control activations for layers: {list(activations_paths)}.")
    concept_dirs = list(_get_concept_directories_and_names(imagenet_data_path))
    with _staged_directories(list(activations_paths.values())) as staging_paths:
        layer_staging_paths = dict(zip(activations_paths, staging_paths, strict=True))
        for concept_dir, concept_name in tqdm(concept_dirs, desc="Processing control concepts"):
            logging.info(f"Processing concept '{concept_name}'.")
            images = _load_concept_images(concept_dir)
            activations = model.get_activations(images, batch_size=batch_size)
            for layer, path in layer_staging_paths.items():
                torch.save(activations[layer], path / f"{concept_name}.pt")


def ensure_control_activations(
    imagenet_data_path: Path | None,
    activations_paths: dict[str, Path],
    model: ExplainedModel,
    batch_size: int = 0,
) -> None:
    """Ensure control activations exist for all layers, computing any that are missing.

    Checks each entry in ``activations_paths``; if all are present and non-empty
    this is a no-op. Otherwise computes and saves only the missing layers in one
    pass through the model, avoiding redundant forward passes.

    Args:
        imagenet_data_path: ImageNet-style directory with per-concept image subdirectories.
            Required when any layer's activations are missing; pass None only if
            all activations are guaranteed to exist.
        activations_paths: Mapping from layer name to the directory of activation
            ``.pt`` files.
        model: The explained model to extract activations from.
        batch_size: Number of images per forward pass. 0 means process all at once.

    Raises:
        ValueError: If any layer's activations are missing and imagenet_data_path is None.
    """
    missing = {
        layer: path
        for layer, path in activations_paths.items()
        if not (path.exists() and any(path.iterdir()))
    }
    if not missing:
        logging.info("All control activations already exist; no computation needed.")
        return

    if imagenet_data_path is None:
        raise ValueError(
            f"Control activations not found for layers {list(missing)} "
            "and no imagenet_data_path was provided to compute them."
        )

    compute_control_activations(imagenet_data_path, missing, model, batch_size=batch_size)


def _get_saved_concept_paths(activations_path: Path) -> dict[str, Path]:
    """Map each control concept saved in a directory to its activation file.

    Args:
        activations_path: Directory containing activation .pt files.

    Returns:
        Mapping from concept name to its .pt file, skipping cached scores.
    """
    return {
        filepath.stem.replace("_", " "): filepath
        for filepath in activations_path.iterdir()
        if filepath.suffix == ".pt" and not filepath.stem.startswith("scores_")
    }


def _load_saved_activations(
    activations_path: Path,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Load and iterate over saved control concept activations from disk.

    Args:
        activations_path: Directory containing activation .pt files.

    Yields:
        Tuples of (concept_name, activations_tensor) for each saved concept.
    """
    for name, filepath in _get_saved_concept_paths(activations_path).items():
        yield name, torch.load(filepath, weights_only=True, map_location="cpu")


def _get_scores_path(activations_path: Path, metric: Metric) -> Path:
    return activations_path / f"scores_{metric.name}.pt"


def _compute_all_scores(activations_path: Path, metric: Metric) -> dict[str, torch.Tensor]:
    """Compute metric scores for every concept across all neurons in the layer.

    Returns:
        Mapping from concept name to a 1-D tensor of shape ``[n_neurons]``.
    """
    scores: dict[str, torch.Tensor] = {}
    for concept_name, act in _load_saved_activations(activations_path):
        n_neurons = act.shape[1]
        scores[concept_name] = torch.tensor(
            [calculate_metric(act[:, i], metric) for i in range(n_neurons)]
        )
    return scores


def ensure_concept_scores(activations_path: Path, metric: Metric) -> dict[str, torch.Tensor]:
    """Return cached concept scores, computing and saving them if not yet on disk.

    Scores are saved to ``scores_{metric.name}.pt`` inside ``activations_path``.

    Returns:
        Mapping from concept name to a 1-D tensor of shape ``[n_neurons]``.

    Raises:
        ValueError: If the cached file does not cover exactly the saved control
            concepts, e.g. because it was computed while another run was still
            writing the activations. The file is left untouched.
    """
    scores_path = _get_scores_path(activations_path, metric)
    if scores_path.exists():
        logging.info(f"Loading cached concept scores from {scores_path}.")
        scores = torch.load(scores_path, weights_only=True, map_location="cpu")
        concept_names = _get_saved_concept_paths(activations_path).keys()
        if scores.keys() != concept_names:
            raise ValueError(
                f"Cached concept scores in {scores_path} do not match the saved control "
                f"concepts ({len(scores)} cached, {len(concept_names)} saved). Point "
                "--control_activations_path at a directory with a complete cache, or move "
                "this file aside to recompute it."
            )
        return scores

    logging.info(f"Computing concept scores for metric '{metric.name}'.")
    scores = _compute_all_scores(activations_path, metric)
    # Write under a temporary name first so concurrent runs never load a partial file.
    staging_path = scores_path.with_name(f".staging_{scores_path.name}-{uuid.uuid4()}")
    try:
        torch.save(scores, staging_path)
        os.replace(staging_path, scores_path)
    finally:
        staging_path.unlink(missing_ok=True)
    logging.info(f"Saved concept scores to {scores_path}.")
    return scores


def _select_best_concepts(concept_scores: dict[str, float], n_best_concepts: int) -> list[str]:
    """Select the top concepts by score.

    Args:
        concept_scores: Dictionary mapping concepts to scores.
        n_best_concepts: Number of top concepts to select.

    Returns:
        List of the top concept names sorted by score (highest first).
    """
    sorted_concepts = sorted(concept_scores, key=lambda k: concept_scores[k], reverse=True)
    if n_best_concepts < len(sorted_concepts):
        return sorted_concepts[:n_best_concepts]
    return sorted_concepts


def get_initial_concepts(
    activations_path: Path,
    neuron_ids: list[int],
    metric: Metric,
    h_size: int,
    best_ratio: float = 0.5,
    pool_ratio: float = 1.0,
) -> dict[int, list[str]]:
    """Get initial concepts for multiple neurons, loading activations from disk once.

    Loads all concept activation tensors from ``activations_path`` in a single
    pass and scores each concept per neuron using ``metric``, avoiding repeated
    disk I/O when many neurons share the same layer.

    Args:
        activations_path: Directory containing activation .pt files.
        neuron_ids: Neuron indices to select concepts for.
        metric: Scoring metric used to rank concepts (reuses pipeline scoring).
        h_size: Maximum scoreboard size. 0 means all concepts are returned.
        best_ratio: Proportion of h_size slots filled with top-scoring concepts;
            the remainder are filled randomly.
        pool_ratio: Fraction (0, 1] of the control-concept bank made available to
            seed the scoreboard. When < 1.0, a random subset of concepts is drawn
            once (shared across all neurons) before per-neuron selection; 1.0 uses
            the full bank. Scores are still computed over the full bank so the
            on-disk cache is unaffected.

    Returns:
        Dictionary mapping each neuron_id to its list of initial concept names.
    """
    all_scores = ensure_concept_scores(activations_path, metric)
    logging.info(f"Loaded {len(all_scores)} control concepts from {activations_path}.")

    if not (0.0 < pool_ratio <= 1.0):
        raise ValueError(f"pool_ratio must be in (0, 1], got {pool_ratio}.")

    if pool_ratio < 1.0:
        n_pool = max(1, int(len(all_scores) * pool_ratio))
        kept = set(random.sample(list(all_scores), n_pool))
        all_scores = {name: t for name, t in all_scores.items() if name in kept}
        logging.info(
            f"Restricted initial concept pool to {len(all_scores)} concepts "
            f"(pool_ratio={pool_ratio})."
        )

    results: dict[int, list[str]] = {}
    for neuron_id in neuron_ids:
        concept_scores = {name: tensor[neuron_id].item() for name, tensor in all_scores.items()}
        if h_size == 0 or len(concept_scores) <= h_size:
            results[neuron_id] = list(concept_scores.keys())
            continue

        n_best = int(h_size * best_ratio)
        n_random = h_size - n_best
        best = _select_best_concepts(concept_scores, n_best)
        excluded = [k for k in concept_scores if k not in best]
        results[neuron_id] = [*best, *random.sample(excluded, n_random)]

    return results
