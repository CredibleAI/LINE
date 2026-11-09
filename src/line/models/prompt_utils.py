"""Functions for creating and formatting prompts for image and language models.

This module provides utilities to generate prompts for text-to-image models
and language models by combining concepts with visual descriptions and
formatting generation history.
"""

import random

_ANGLES = ("extreme close-up", "wide angle shot", "aerial view", "low angle")
_LIGHTNING = ("cinematic lighting", "natural sunlight", "studio lighting")
_AVAILABLE_DESCRIPTIONS = (_ANGLES, _LIGHTNING)


def _get_random_visual_descriptions() -> list[str]:
    """Get random visual descriptions for image generation.

    Returns an iterator that yields one random description from each
    category of visual descriptions (angles and lighting).

    Returns:
        List of random visual descriptions, one from each category.
    """
    return [random.choice(descriptions) for descriptions in _AVAILABLE_DESCRIPTIONS]


def _get_seeded_random_visual_descriptions(seed: int) -> list[str]:
    """Get deterministic random visual descriptions for image generation based on a seed.

    Returns an iterator that yields one random description from each
    category of visual descriptions (angles and lighting).

    Returns:
        List of random visual descriptions, one from each category.
    """
    rng = random.Random(seed)
    return [rng.choice(descriptions) for descriptions in _AVAILABLE_DESCRIPTIONS]


def concept_image_prompt(prompt_text: str, concept: str, seed: int | None = None) -> str:
    """Create an image generation prompt by combining text with visual descriptions.

    Combines a base prompt text with a concept and adds random visual descriptions
    (angles and lighting) to guide the image generation model.

    Args:
        prompt_text: Base prompt text for the image.
        concept: Concept name to include in the prompt.
        seed: Optional seed for deterministic random visual descriptions. If None,
              uses non-deterministic randomness.

    Returns:
        Complete prompt string for text-to-image generation.
    """
    text_to_image_prompt = f"{prompt_text} {concept}"
    if seed is not None:
        random_visual_descriptions = _get_seeded_random_visual_descriptions(seed)
    else:
        random_visual_descriptions = _get_random_visual_descriptions()

    return ", ".join((text_to_image_prompt, *random_visual_descriptions))


def _read_prompt_template(prompt_path: str) -> str:
    """Read a prompt template from a file.

    Args:
        prompt_path: Path to the prompt template file.

    Returns:
        The contents of the prompt template file as a string.
    """
    with open(prompt_path) as prompt_file:
        prompt_template = prompt_file.read()
    return prompt_template


def generate_prompt(
    concept_history: dict[str, float],
    generation_history: list[tuple[str, float]],
    prompt_template: str,
    top_k: int | None = None,
) -> str:
    """Generate a prompt for the language model by filling a template.

    Creates a formatted prompt by filling a template with the concept history
    (sorted by score) and generation history. Optionally limits to top-k concepts.

    Args:
        concept_history: Dictionary mapping concept names to their scores.
        generation_history: Sequence of generated concept names.
        prompt_template: Prompt template string with {concept_list} and
                         {generation_history} placeholders.
        top_k: Optional number of top-scoring concepts to include. If None,
               includes all concepts.

    Returns:
        Formatted prompt string with concept and generation history inserted.
    """
    score_sorted_concepts = [
        f"{k}: {v:.2f}"
        for k, v in sorted(concept_history.items(), key=lambda item: item[1], reverse=True)
    ]
    if top_k is not None and top_k > 0:
        score_sorted_concepts = score_sorted_concepts[:top_k]

    concept_list = "; ".join(score_sorted_concepts) + ";"
    generation_list = [concept for concept, _ in generation_history]

    return prompt_template.format(
        concept_list=concept_list, generation_history=", ".join(generation_list)
    ).strip()
