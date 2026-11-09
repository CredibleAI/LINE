"""Utility functions for formatting pipeline artifacts.

Pure formatting helpers with no I/O or side effects — suitable for use
anywhere in the pipeline without pulling in heavy dependencies.
"""

import json
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Literal


def format_as_json_string(
    dictionaries_list: Sequence[Mapping[str, Any]],
) -> Sequence[str]:
    """Convert a list of dictionaries to JSON-formatted strings.

    Args:
        dictionaries_list: Sequence of dictionaries to format.

    Returns:
        Sequence of JSON-formatted strings, one per dictionary.
    """
    return [json.dumps(d) for d in dictionaries_list]


def _iteration_label(position: int, n_iters: int) -> int | Literal["summary"]:
    """Label the ``position``-th LLM call (1-based); calls after ``n_iters`` are the summary."""
    return position if position <= n_iters else "summary"


def format_generation_history(
    generation_history: Sequence[tuple[str, float]],
    n_iters: int,
) -> list[dict[str, Any]]:
    """Format the generation history as one record per LLM call, in call order.

    Args:
        generation_history: Ordered (concept, score) pairs, one per LLM call.
        n_iters: Number of main iterations; later calls are the summary step.

    Returns:
        Records with keys ``iteration`` (1-based, or ``"summary"``), ``concept`` and ``score``.
    """
    return [
        {"iteration": _iteration_label(position, n_iters), "concept": concept, "score": score}
        for position, (concept, score) in enumerate(generation_history, start=1)
    ]


def format_final_concept_history(
    concept_history: Mapping[str, float],
    generation_history: Sequence[tuple[str, float]],
    initial_concepts: Collection[str],
    n_iters: int,
) -> list[dict[str, Any]]:
    """Format the final scoreboard as records sorted by score, best first.

    Args:
        concept_history: Final scoreboard mapping concept names to scores.
        generation_history: Ordered (concept, score) pairs, one per LLM call.
        initial_concepts: Concepts the scoreboard was seeded with.
        n_iters: Number of main iterations; later calls are the summary step.

    Returns:
        Records with keys ``concept``, ``score``, ``first_iteration`` (0 for the
        initial pool, otherwise the first call that proposed the concept) and
        ``is_summary`` (whether the summary call proposed the concept).
    """
    first_iteration: dict[str, int | Literal["summary"]] = dict.fromkeys(initial_concepts, 0)
    summary_concepts: set[str] = set()
    for position, (concept, _) in enumerate(generation_history, start=1):
        label = _iteration_label(position, n_iters)
        first_iteration.setdefault(concept, label)
        if label == "summary":
            summary_concepts.add(concept)
    ranked = sorted(concept_history.items(), key=lambda item: item[1], reverse=True)
    return [
        {
            "concept": concept,
            "score": score,
            "first_iteration": first_iteration.get(concept),
            "is_summary": concept in summary_concepts,
        }
        for concept, score in ranked
    ]
