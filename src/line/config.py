"""Configuration dataclasses for the pipeline.

Defines configuration dataclasses for model loading, image generation,
concept history, and result management.
"""

import dataclasses
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from line.utils import ORIGINAL_PROMPT_DIR


@dataclasses.dataclass
class LoadConfig:
    """Configuration for loading machine learning models.

    Attributes:
        language_model_id: Identifier for the language model.
        text_to_image_model_id: Identifier for the text-to-image model.
        explained_model_id: Identifier for the model to explain.
        language_model_kwargs: Additional arguments for language model initialization.
        text_to_image_model_kwargs: Additional arguments for text-to-image model initialization.
        explained_model_kwargs: Additional arguments for explained model initialization.
        model_swapping: Whether to enable GPU/CPU model swapping for memory efficiency.
    """

    language_model_id: str
    text_to_image_model_id: str
    explained_model_id: str
    language_model_kwargs: Mapping[str, Any]
    text_to_image_model_kwargs: Mapping[str, Any]
    explained_model_kwargs: Mapping[str, Any]
    model_swapping: bool
    max_new_tokens: int = 256
    prompt_path: str = os.path.join(ORIGINAL_PROMPT_DIR, "prompt_main.txt")
    summary_prompt_path: str = os.path.join(ORIGINAL_PROMPT_DIR, "prompt_summary.txt")
    llm_batch_size: int = 0
    explained_model_batch_size: int = 0
    deterministic_llm: bool = False


@dataclasses.dataclass
class ImageGenerationConfig:
    """Configuration for image generation.

    Attributes:
        n_images: Number of images to generate per concept.
        prompt_text: Base prompt text for image generation.
    """

    n_images: int
    prompt_text: str
    t2i_batch_size: int = 2


@dataclasses.dataclass
class ConceptHistoryConfig:
    """Configuration for concept history initialization and size.

    Attributes:
        h_size: Maximum scoreboard size during iterative updates. 0 means unbounded.
        initial_h_size: Number of concepts to seed the scoreboard with at startup.
            0 means use all available control concepts.
        best_ratio: Proportion of initial_h_size slots filled with top-scoring concepts;
            the remainder are filled randomly.
        init_pool_ratio: Fraction (0, 1] of the control-concept bank made available
            to seed the scoreboard at startup. 1.0 uses the full bank (default).
    """

    h_size: int = 0
    initial_h_size: int = 10
    best_ratio: float = 0.5
    init_pool_ratio: float = 1.0


@dataclasses.dataclass
class NeuronState:
    """Per-neuron mutable state: histories and result accumulators.

    Attributes:
        neuron_id: Index of the neuron being explained.
        layer: Name of the model layer this neuron belongs to.
        model_id: Identifier of the explained model.
        save_directory: Directory to save neuron-specific artifacts.
        h_size: Maximum scoreboard size. 0 means unbounded.
        concept_history: Scoreboard mapping concept names to scores.
        generation_history: Ordered list of all (concept, score) pairs generated so far.
        reasoning: List of reasoning dicts produced per iteration.
        best_concepts: List of best-concept dicts produced per iteration.
        initial_concepts: Concepts the scoreboard was seeded with, in seeding order.
    """

    neuron_id: int
    layer: str
    model_id: str
    save_directory: Path
    h_size: int
    concept_history: dict[str, float] = dataclasses.field(default_factory=dict)
    generation_history: list[tuple[str, float]] = dataclasses.field(default_factory=list)
    reasoning: list[dict] = dataclasses.field(default_factory=list)
    best_concepts: list[dict] = dataclasses.field(default_factory=list)
    initial_concepts: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class HistoryManagingConfig:
    """Configuration for saving experiment results.

    Attributes:
        save_images: Whether to save generated images to disk.
        save_histories: Whether to save generation and concept history to disk.
        save_directory: Directory path where results will be saved.
    """

    save_images: bool
    save_histories: bool
    save_directory: str


@dataclasses.dataclass
class RunConfig:
    """Top-level configuration that groups all pipeline sub-configurations.

    Attributes:
        load_config: Configuration for loading machine learning models.
        image_generation_config: Configuration for image generation.
        concept_history_config: Configuration for concept history initialization.
        history_managing_config: Configuration for saving experiment results.
        n_iters: Number of main iterations before the summary step.
        neurons: Mapping from layer name to the list of neuron indices to explain.
        initializations_path: Directory where control activations are stored.
        metric: Name of the scoring metric. Must match a
            :class:`~line.pipeline.scoring.Metric` member.
        top_k: Number of top concepts passed to the LLM in the final summary step.
        imagenet_data_path: Path to ImageNet-style images used to compute control
            activations. Required when activations have not been pre-computed.
    """

    load_config: LoadConfig
    image_generation_config: ImageGenerationConfig
    concept_history_config: ConceptHistoryConfig
    history_managing_config: HistoryManagingConfig
    n_iters: int
    neurons: dict[str, list[int]]
    initializations_path: str
    metric: str = "AVG_ACTIVATION"
    top_k: int = 3
    imagenet_data_path: str | None = None

    @classmethod
    def from_namespace(cls, args: Any, neurons: dict[str, list[int]]) -> "RunConfig":
        """Build a RunConfig from an argparse.Namespace.

        Args:
            args: Parsed arguments from the argument parser.

        Returns:
            Populated RunConfig instance.
        """
        t2i_kwargs: dict[str, Any] = {
            "num_inference_steps": args.num_inf_steps,
            "guidance_scale": args.guidance_scale,
        }
        return cls(
            load_config=LoadConfig(
                language_model_id=args.language_model,
                text_to_image_model_id=args.t2i_model,
                explained_model_id=args.explained_model,
                model_swapping=args.model_swapping,
                max_new_tokens=args.max_new_tokens,
                prompt_path=args.prompt_path,
                summary_prompt_path=args.summary_prompt_path,
                llm_batch_size=args.llm_batch_size,
                explained_model_batch_size=args.explained_model_batch_size,
                deterministic_llm=args.deterministic_llm,
                text_to_image_model_kwargs=t2i_kwargs,
                language_model_kwargs={},
                explained_model_kwargs={},
            ),
            image_generation_config=ImageGenerationConfig(
                n_images=args.num_img,
                prompt_text=args.prompt_text,
                t2i_batch_size=args.t2i_batch_size,
            ),
            concept_history_config=ConceptHistoryConfig(
                h_size=args.h_size,
                initial_h_size=args.initial_h_size,
                best_ratio=args.best_ratio,
                init_pool_ratio=args.init_pool_ratio,
            ),
            history_managing_config=HistoryManagingConfig(
                save_images=args.save_images,
                save_histories=args.save_histories,
                save_directory=args.save_dir,
            ),
            n_iters=args.n_iters,
            neurons=neurons,
            initializations_path=args.control_activations_path,
            imagenet_data_path=args.imagenet_data_path,
            metric=args.metric,
            top_k=args.top_k,
        )
