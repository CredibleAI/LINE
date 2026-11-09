"""Concept scoreboard update and result persistence for the LINE pipeline."""

import hashlib
import json
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from PIL import Image

PARAMS_FILENAME = "params.json"
IMAGES_DIRNAME = "images"
GENERATION_HISTORY_FILENAME = "generation_history.jsonl"
FINAL_CONCEPT_HISTORY_FILENAME = "final_concept_history.jsonl"
REASONING_FILENAME = "reasoning.jsonl"
BEST_CONCEPTS_FILENAME = "best_concepts.jsonl"
# Runs saved before the switch to JSONL use the same file names with this suffix.
LEGACY_SUFFIX = ".txt"

_UNSAFE_IN_DIR_NAME = re.compile(r"[\s/\\\x00-\x1f\x7f]")
_MAX_SLUG_BYTES = 100
_DISAMBIGUATOR_HEX_DIGITS = 8


def update_concept_history(
    concept_history: dict[str, float], new_concept: str, score: float, h_size: int
) -> dict[str, float]:
    """Update the scoreboard by replacing the worst or a stochastically chosen concept.

    If the new concept's score exceeds the current minimum in H, the
    lowest-scoring entry is replaced. Otherwise, an existing concept r is
    removed with probability proportional to (max_score - s_r), so
    higher-scoring concepts survive with greater probability.

    Args:
        concept_history: Dictionary mapping concept names to their scores.
        new_concept: Name of the new concept to add.
        score: Score of the new concept.
        h_size: Maximum scoreboard size. 0 means unbounded - always add without removing.

    Returns:
        Updated concept history with the new concept added and one entry removed.
    """
    if h_size == 0 or len(concept_history) < h_size:
        return {**concept_history, new_concept: score}

    worst_score_concept = min(concept_history, key=lambda k: concept_history[k])

    if score > concept_history[worst_score_concept]:
        concept_to_remove = worst_score_concept
    else:
        max_score = max(concept_history.values())
        weights = [max_score - s for s in concept_history.values()]
        # All scores are equal; fall back to uniform random removal
        if all(w == 0 for w in weights):
            concept_to_remove = random.choice(list(concept_history.keys()))
        else:  # Use weighted random removal based on score differences
            concept_to_remove = random.choices(
                list(concept_history.keys()),
                weights=weights,
            )[0]

    updated_concept_history = {k: v for k, v in concept_history.items() if k != concept_to_remove}
    updated_concept_history[new_concept] = score
    return updated_concept_history


def add_concept(
    concept_history: dict[str, float],
    generation_history: list[tuple[str, float]],
    new_concept: str,
    score: float,
    h_size: int,
) -> None:
    """Append to generation_history and update the concept_history scoreboard in-place.

    Args:
        concept_history: Scoreboard mapping concept names to scores (mutated in-place).
        generation_history: Ordered list of all (concept, score) pairs (mutated in-place).
        new_concept: Name of the new concept to add.
        score: Score of the new concept.
        h_size: Maximum scoreboard size. 0 means unbounded.
    """
    generation_history.append((new_concept, score))
    updated = update_concept_history(concept_history, new_concept, score, h_size)
    concept_history.clear()
    concept_history.update(updated)


def get_best_concept(concept_history: dict[str, float]) -> tuple[str, float]:
    """Return the concept with the highest score from the scoreboard.

    Args:
        concept_history: Scoreboard mapping concept names to scores.

    Returns:
        Tuple of (best_concept_name, score).
    """
    best_concept = max(concept_history, key=lambda k: concept_history[k])
    return best_concept, concept_history[best_concept]


def save_to_file(lines: Iterable[str], save_directory: Path, filename: str) -> None:
    """Write lines to a text file, creating the directory if needed.

    Args:
        lines: Iterable of strings to write, one per line.
        save_directory: Directory to save the file to.
        filename: Name of the file to save.
    """
    save_directory.mkdir(parents=True, exist_ok=True)
    with open(save_directory / filename, "w") as f:
        f.write("\n".join(lines))


def save_images_from_iteration(
    images: list[Image.Image],
    save_directory: Path,
    iteration: int | Literal["summary"],
    concept: str,
) -> None:
    """Save images generated in a single pipeline iteration.

    Creates a subdirectory for the iteration and saves each image as a JPEG file.

    Args:
        images: Sequence of PIL Image objects to save.
        save_directory: Base directory to save images to.
        iteration: Iteration number or ``"summary"`` for the final step.
        concept: Concept name, used in directory naming.
    """
    save_path = save_directory / image_dir_name(iteration, concept)
    save_path.mkdir(parents=True, exist_ok=True)
    for image_number, image in enumerate(images, 1):
        image.save(save_path / f"image_{image_number}.jpeg", format="JPEG")


def save_params(params: dict[str, Any], save_directory: Path) -> None:
    """Save pipeline run parameters to PARAMS_FILENAME.

    Args:
        params: Mapping of parameter names to values.
        save_directory: Directory to save the file to.
    """
    save_directory.mkdir(parents=True, exist_ok=True)
    with open(save_directory / PARAMS_FILENAME, "w") as f:
        json.dump(params, f, indent=4)


def read_params(save_directory: Path) -> dict[str, Any]:
    """Read pipeline run parameters from PARAMS_FILENAME.

    Args:
        save_directory: Directory where parameters were saved.

    Returns:
        Dictionary of parameter names to values.
    """
    params_path = save_directory / PARAMS_FILENAME
    if not params_path.exists():
        raise FileNotFoundError(f"No parameters file found at {params_path}")

    with open(params_path) as f:
        params = json.load(f)
    return params


def find_output_file(neuron_dir: Path, filename: str) -> Path | None:
    """Locate a per-neuron output file, falling back to its pre-JSONL name.

    Args:
        neuron_dir: Directory holding one neuron's results.
        filename: Current file name, e.g. ``GENERATION_HISTORY_FILENAME``.

    Returns:
        Path to the file under its current name, else under its legacy ``.txt``
        name, else None if neither exists.
    """
    path = neuron_dir / filename
    for candidate in (path, path.with_suffix(LEGACY_SUFFIX)):
        if candidate.exists():
            return candidate
    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a file holding one JSON object per line, skipping empty lines.

    Args:
        path: Path to the JSONL file.

    Returns:
        Parsed objects in file order.
    """
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_legacy_concept_scores(path: Path) -> list[tuple[str, float]]:
    """Parse a pre-JSONL history file of unescaped ``concept,score`` lines.

    A concept containing a newline spans several physical lines, so lines are
    accumulated until the text after the last comma parses as a float.

    Args:
        path: Path to a legacy ``generation_history.txt`` or ``final_concept_history.txt``.

    Returns:
        (concept, score) pairs in file order.

    Raises:
        ValueError: If the file ends with text that is not a complete entry.
    """
    pairs: list[tuple[str, float]] = []
    buffer: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip() and not buffer:
            continue
        buffer.append(line)
        concept, sep, score_str = "\n".join(buffer).rpartition(",")
        if not sep:
            continue
        try:
            score = float(score_str)
        except ValueError:
            continue
        pairs.append((concept, score))
        buffer = []
    if buffer:
        raise ValueError(f"{path}: trailing text is not a 'concept,score' entry: {buffer!r}")
    return pairs


def read_best_concepts(neuron_dir: Path) -> list[dict[str, Any]]:
    """Read one neuron's best-concept records, one per LLM call.

    Args:
        neuron_dir: Directory holding one neuron's results.

    Returns:
        Records with keys ``iteration``, ``best_concept`` and ``best_score``;
        empty if the file does not exist.
    """
    path = find_output_file(neuron_dir, BEST_CONCEPTS_FILENAME)
    # The legacy .txt file holds the same JSON lines.
    return read_jsonl(path) if path is not None else []


def read_reasoning(neuron_dir: Path) -> list[dict[str, Any]]:
    """Read one neuron's LLM reasoning records, one per LLM call.

    Args:
        neuron_dir: Directory holding one neuron's results.

    Returns:
        Records with keys ``concept`` and ``reasoning``; empty if the file does not exist.
    """
    path = find_output_file(neuron_dir, REASONING_FILENAME)
    # The legacy .txt file holds the same JSON lines.
    return read_jsonl(path) if path is not None else []


def read_generation_history(neuron_dir: Path) -> list[dict[str, Any]]:
    """Read one neuron's generation history, one record per LLM call in call order.

    Args:
        neuron_dir: Directory holding one neuron's results.

    Returns:
        Records with keys ``iteration`` (1-based, or ``"summary"``), ``concept`` and
        ``score``; empty if the file does not exist.
    """
    path = find_output_file(neuron_dir, GENERATION_HISTORY_FILENAME)
    if path is None:
        return []
    if path.suffix != LEGACY_SUFFIX:
        return read_jsonl(path)
    pairs = _read_legacy_concept_scores(path)
    # Legacy files were written after the summary call, so the last entry is the summary.
    return [
        {
            "iteration": position if position < len(pairs) else "summary",
            "concept": concept,
            "score": score,
        }
        for position, (concept, score) in enumerate(pairs, start=1)
    ]


def read_final_concept_history(neuron_dir: Path) -> list[dict[str, Any]]:
    """Read one neuron's final scoreboard, sorted by score, best first.

    Args:
        neuron_dir: Directory holding one neuron's results.

    Returns:
        Records with keys ``concept``, ``score``, ``first_iteration`` and ``is_summary``;
        empty if the file does not exist. For runs saved before the switch to JSONL,
        ``first_iteration`` is None because the old files do not record it.
    """
    path = find_output_file(neuron_dir, FINAL_CONCEPT_HISTORY_FILENAME)
    if path is None:
        return []
    if path.suffix != LEGACY_SUFFIX:
        return read_jsonl(path)
    generation_history = read_generation_history(neuron_dir)
    summary_concept = generation_history[-1]["concept"] if generation_history else None
    records = [
        {
            "concept": concept,
            "score": score,
            "first_iteration": None,
            "is_summary": concept == summary_concept,
        }
        for concept, score in _read_legacy_concept_scores(path)
    ]
    return sorted(records, key=lambda record: record["score"], reverse=True)


def _validate_iteration(iteration: int | str) -> None:
    if isinstance(iteration, str) and iteration != "summary":
        raise ValueError(f"iteration must be an int or 'summary', got {iteration!r}")


def _raw_slug(concept: str) -> tuple[str, bool]:
    """Filesystem-safe slug for `concept`, and whether producing it was lossy."""
    slug = _UNSAFE_IN_DIR_NAME.sub("_", concept)
    lossy = slug != concept
    if len(slug.encode()) > _MAX_SLUG_BYTES:
        slug = slug.encode()[:_MAX_SLUG_BYTES].decode(errors="ignore") + "…"
        lossy = True
    return slug, lossy


def _slug(concept: str) -> str:
    """Filesystem-safe slug for `concept`, disambiguated whenever `_raw_slug` was lossy.
    Add a hash of the original concept to the slug to ensure uniqueness in case of collision.
    """
    slug, lossy = _raw_slug(concept)
    if lossy:
        slug += "_" + hashlib.sha256(concept.encode()).hexdigest()[:_DISAMBIGUATOR_HEX_DIGITS]
    return slug


def image_dir_name(iteration: int | str, concept: str) -> str:
    """Name of the folder holding one concept's generated images.

    Args:
        iteration: Iteration number or ``"summary"`` for the final step.
        concept: Concept name as the language model wrote it.

    Returns:
        Folder name of the form ``iteration_<iteration>_<concept slug>``, with the slug
        disambiguated (see `_slug`) whenever making it filesystem-safe was lossy.
    """
    _validate_iteration(iteration)
    return f"iteration_{iteration}_{_slug(concept)}"


def _legacy_image_dir_name(iteration: int | str, concept: str) -> str:
    """Folder name used by runs saved before concept slugs were made filesystem-safe."""
    _validate_iteration(iteration)
    slug = concept.replace(" ", "_")
    if len(slug) > 100:
        slug = slug[:100] + "…"
    return f"iteration_{iteration}_{slug}"


def find_concept_images(neuron_dir: Path, concept: str) -> list[Path]:
    """Locate the folders holding one concept's generated images.

    Args:
        neuron_dir: Directory holding one neuron's results.
        concept: Concept name as recorded in the history files.

    Returns:
        Existing folders, one per step that generated images for the concept: the
        initial pool first, then every call that proposed it. Empty if there are none.
    """
    images_dir = neuron_dir / IMAGES_DIRNAME
    iterations: list[int | str] = [0]
    iterations += [
        record["iteration"]
        for record in read_generation_history(neuron_dir)
        if record["concept"] == concept
    ]
    found = []
    for iteration in dict.fromkeys(iterations):
        for name in (
            image_dir_name(iteration, concept),
            _legacy_image_dir_name(iteration, concept),
        ):
            path = images_dir / name
            if path.is_dir():
                found.append(path)
                break
    return found
