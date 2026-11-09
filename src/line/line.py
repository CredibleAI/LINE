"""LINE: LLM-based Iterative Neuron Explanations for Vision Models.
Integrates language models, image generation models, and vision models
to iteratively generate and refine concept descriptions that explain
individual neuron behavior in the LINE framework.
"""

import dataclasses
import datetime
import logging
from collections import defaultdict
from pathlib import Path
from typing import Literal

import torch
from PIL import Image

from line import config
from line.config import NeuronState
from line.models import LLM, T2I
from line.models.base import ExplainedModel
from line.pipeline import history, initialization, scoring, utils
from line.utils import set_random_seed


class LINE:
    """Main pipeline for neuron labeling with LLM-generated concepts and T2I images.

    Orchestrates an iterative process of generating concepts, creating images,
    and scoring concepts based on their ability to activate target neurons.
    Supports explaining multiple neurons across multiple layers in a single run,
    sharing the T2I and LLM models across all neurons.
    """

    def __init__(
        self,
        run_config: config.RunConfig,
        explained_model: ExplainedModel,
        device: torch.device,
    ) -> None:
        """Initialize the explanation pipeline.

        Args:
            run_config: Top-level configuration grouping all pipeline settings.
            explained_model: Pre-instantiated model to extract activations from.
            device: Device to run T2I and LLM models on.
        """
        load_config = run_config.load_config
        image_generation_config = run_config.image_generation_config
        concept_history_config = run_config.concept_history_config
        history_managing_config = run_config.history_managing_config
        neurons = run_config.neurons
        imagenet_data_path = run_config.imagenet_data_path
        metric = scoring.Metric[run_config.metric]

        if imagenet_data_path is not None and not Path(imagenet_data_path).exists():
            raise ValueError(f"imagenet_data_path '{imagenet_data_path}' does not exist.")

        self._load_config = load_config
        self._device = device
        self._explained_model = explained_model
        self._imagenet_data_path = (
            Path(imagenet_data_path) if imagenet_data_path is not None else None
        )
        self._image_generation_config = image_generation_config
        self._concept_history_config = concept_history_config
        self._history_managing_config = history_managing_config
        self._neurons = neurons
        self._model_layer_activations_paths = {
            layer: initialization.get_model_layer_activations_path(
                Path(run_config.initializations_path), load_config.explained_model_id, layer
            )
            for layer in neurons
        }
        self._metric = metric
        self._n_iters = run_config.n_iters
        self._top_k = run_config.top_k
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self._run_id = f"{timestamp}_{load_config.explained_model_id}"
        self.save_directory = Path(self._history_managing_config.save_directory) / self._run_id

        history.save_params(
            {
                "run_id": self._run_id,
                **dataclasses.asdict(run_config),
                "model_layer_activations_paths": {
                    layer: str(path) for layer, path in self._model_layer_activations_paths.items()
                },
            },
            self.save_directory,
        )

    def _load_models(self) -> None:
        """Load shared T2I and LLM models, and create one NeuronState per (layer, neuron)."""
        logging.info(f"Loading T2I: {self._load_config.text_to_image_model_id}.")
        self._t2i_model = T2I(
            model_id=self._load_config.text_to_image_model_id,
            device=self._device,
            model_swapping=self._load_config.model_swapping,
            batch_size=self._image_generation_config.t2i_batch_size,
            **self._load_config.text_to_image_model_kwargs,
        )
        logging.info(f"Loading LLM: {self._load_config.language_model_id}.")
        self._lang_model = LLM(
            model_id=self._load_config.language_model_id,
            device=self._device,
            model_swapping=self._load_config.model_swapping,
            max_new_tokens=self._load_config.max_new_tokens,
            prompt_path=self._load_config.prompt_path,
            summary_prompt_path=self._load_config.summary_prompt_path,
            batch_size=self._load_config.llm_batch_size,
            deterministic=self._load_config.deterministic_llm,
        )
        self._neuron_states: dict[tuple[str, int], NeuronState] = {
            (layer, neuron_id): NeuronState(
                neuron_id=neuron_id,
                layer=layer,
                model_id=self._load_config.explained_model_id,
                save_directory=(self.save_directory / f"layer_{layer}" / f"neuron_{neuron_id}"),
                h_size=self._concept_history_config.h_size,
            )
            for layer, neuron_ids in self._neurons.items()
            for neuron_id in neuron_ids
        }

    def _setup(self) -> None:
        """Compute activations if needed, load models, initialize per-neuron scoreboards."""
        logging.info("Setting up pipeline.")
        initialization.ensure_control_activations(
            imagenet_data_path=self._imagenet_data_path,
            activations_paths=self._model_layer_activations_paths,
            model=self._explained_model,
            batch_size=self._load_config.explained_model_batch_size,
        )
        self._load_models()
        self._initialize_concept_histories()
        logging.info("Setup complete.")

    def _generate_images(self, concepts: list[str]) -> list[Image.Image]:
        """Generate images for a list of concepts.

        Args:
            concepts: Concept names to generate images for.

        Returns:
            List of generated PIL Images, length ``len(concepts) * n_images``,
            ordered as ``[img_0_c0, img_0_c1, …, img_1_c0, img_1_c1, …]``.
        """
        return self._t2i_model.generate_images_from_concept(
            concepts=[(self._image_generation_config.prompt_text, c) for c in concepts],
            n_images=self._image_generation_config.n_images,
        )

    def _initialize_concept_histories(self) -> None:
        """Initialize scoreboards for all neurons.

        Gathers initial concepts for every neuron, generates images for all
        unique concepts in a single T2I call, runs the explained model once on
        the full image batch, then slices activations per neuron to build each
        scoreboard.
        """
        # Group neurons by layer so activations are loaded from disk once per layer.
        neurons_by_layer: dict[str, list[int]] = defaultdict(list)
        for layer, neuron_id in self._neuron_states:
            neurons_by_layer[layer].append(neuron_id)

        initial_concepts_per_neuron: dict[tuple[str, int], list[str]] = {}
        for layer, neuron_ids in neurons_by_layer.items():
            logging.info(f"Getting initial concepts for layer {layer} ({len(neuron_ids)} neurons).")
            per_neuron_concepts = initialization.get_initial_concepts(
                activations_path=self._model_layer_activations_paths[layer],
                neuron_ids=neuron_ids,
                metric=self._metric,
                h_size=self._concept_history_config.initial_h_size,
                best_ratio=self._concept_history_config.best_ratio,
                pool_ratio=self._concept_history_config.init_pool_ratio,
            )
            for neuron_id, concepts in per_neuron_concepts.items():
                initial_concepts_per_neuron[(layer, neuron_id)] = concepts

        # Deduplicate for image generation: one T2I call.
        unique_concepts = sorted(
            {c for concepts in initial_concepts_per_neuron.values() for c in concepts}
        )

        if not unique_concepts:
            return

        logging.info(f"Generating images for {len(unique_concepts)} unique initial concepts.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        all_images = self._generate_images(unique_concepts)

        logging.info("Collecting activations for all initial concept images.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        activations = self._explained_model.get_activations(
            all_images, batch_size=self._load_config.explained_model_batch_size
        )

        n_images = self._image_generation_config.n_images
        n_unique = len(unique_concepts)
        concept_to_idx = {c: i for i, c in enumerate(unique_concepts)}

        # Image layout from generate_images:
        #   [c0_g0, c1_g0, …, c_{n-1}_g0,  c0_g1, …, c_{n-1}_g_{n_images-1}]
        for (layer, neuron_id), state in self._neuron_states.items():
            concept_history: dict[str, float] = {}
            for concept in initial_concepts_per_neuron[(layer, neuron_id)]:
                idx = concept_to_idx[concept]
                # Indices for concept i across all n_images copies:
                #   [g * n_unique + i  for g in range(n_images)]
                indices = [g * n_unique + idx for g in range(n_images)]
                concept_activations = activations[layer][indices, neuron_id]
                concept_history[concept] = scoring.calculate_metric(
                    concept_activations, self._metric
                )
                if self._history_managing_config.save_images:
                    concept_images = [all_images[g * n_unique + idx] for g in range(n_images)]
                    history.save_images_from_iteration(
                        concept_images,
                        state.save_directory / history.IMAGES_DIRNAME,
                        0,
                        concept,
                    )
            state.concept_history = concept_history
            state.initial_concepts = list(concept_history)

    def _run_iteration(
        self,
        states: list[NeuronState],
        iteration: int | Literal["summary"],
    ) -> list[tuple[str, float]]:
        """Run one iteration for all neurons simultaneously.

        Generates new concepts for every neuron in one LLM, creates images
        for all concepts, then extracts activations for all images from the explained-model.
        Finally, scoring and updating each neuron's state.

        Args:
            states: Per-neuron states to process together.
            iteration: Iteration number or ``"summary"`` for the final step.

        Returns:
            List of (new_concept, score) tuples, one per entry in ``states``.
        """
        n_neurons = len(states)
        n_images = self._image_generation_config.n_images
        top_k = self._top_k if iteration == "summary" else None

        # 1. LLM generation: one call for all neurons
        logging.info(f"LLM: generating concepts for {len(states)} neurons.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        llm_results = self._lang_model.generate_concept(
            [s.concept_history for s in states],
            [s.generation_history for s in states],
            top_k,
        )
        new_concepts, reasonings = (
            map(list, zip(*llm_results, strict=True)) if llm_results else ([], [])
        )

        # 2. T2I: one batched call for all concepts.
        logging.info(f"T2I: generating images for {len(new_concepts)} concepts.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        all_images = self._generate_images(new_concepts)

        # Optionally save generated images to disk, organized by neuron and iteration.
        if self._history_managing_config.save_images:
            for i, (state, concept) in enumerate(zip(states, new_concepts, strict=True)):
                concept_images = [all_images[g * n_neurons + i] for g in range(n_images)]
                history.save_images_from_iteration(
                    concept_images,
                    state.save_directory / history.IMAGES_DIRNAME,
                    iteration,
                    concept,
                )

        # 3. Explained model: one forward pass for all images.
        logging.info(f"Explained model: extracting activations for {len(all_images)} images.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        activations = self._explained_model.get_activations(
            all_images, batch_size=self._load_config.explained_model_batch_size
        )

        # 4. Score and update each neuron state.
        results: list[tuple[str, float]] = []
        for i, state in enumerate(states):
            indices = [g * n_neurons + i for g in range(n_images)]
            concept_activations = activations[state.layer][indices, state.neuron_id]
            score = scoring.calculate_metric(concept_activations, self._metric)

            history.add_concept(
                state.concept_history,
                state.generation_history,
                new_concepts[i],
                score,
                state.h_size,
            )
            state.reasoning.append({"concept": new_concepts[i], "reasoning": reasonings[i]})
            best_concept, best_score = history.get_best_concept(state.concept_history)
            state.best_concepts.append(
                {
                    "iteration": iteration,
                    "best_concept": best_concept,
                    "best_score": best_score,
                }
            )
            results.append((new_concepts[i], score))

        return results

    def _save_neuron_histories(self, state: NeuronState) -> None:
        """Save generation history and final scoreboard for one neuron as JSONL."""
        history.save_to_file(
            utils.format_as_json_string(
                utils.format_generation_history(state.generation_history, self._n_iters)
            ),
            state.save_directory,
            history.GENERATION_HISTORY_FILENAME,
        )
        history.save_to_file(
            utils.format_as_json_string(
                utils.format_final_concept_history(
                    state.concept_history,
                    state.generation_history,
                    state.initial_concepts,
                    self._n_iters,
                )
            ),
            state.save_directory,
            history.FINAL_CONCEPT_HISTORY_FILENAME,
        )

    def run_pipeline(self, seed: int | None = None) -> dict[tuple[str, int], str]:
        """Run the complete pipeline for all (layer, neuron) pairs.

        Each iteration processes all neurons together: concepts are generated,
        images are synthesised, and activations are extracted in fully batched
        calls before scores are assigned per neuron.

        Args:
            seed: Random seed for reproducibility. If None, seeding is skipped.

        Returns:
            Dictionary mapping (layer, neuron_id) to its best concept.
        """
        if seed is not None:
            set_random_seed(seed)
        self._setup()

        states = list(self._neuron_states.values())

        for iter_number in range(1, self._n_iters + 1):
            logging.info(f"ITERATION {iter_number} OF {self._n_iters}.")
            iter_results = self._run_iteration(states, iteration=iter_number)
            for state, (new_concept, score) in zip(states, iter_results, strict=True):
                logging.info(
                    f"LAYER {state.layer} NEURON {state.neuron_id}"
                    f" — PROPOSED: {new_concept} (score {score:.4f})."
                )

        logging.info("RUNNING SUMMARY ITERATION.")
        summary_results = self._run_iteration(states, iteration="summary")
        for state, (summary_concept, summary_score) in zip(states, summary_results, strict=True):
            logging.info(
                f"LAYER {state.layer} NEURON {state.neuron_id}"
                f" — SUMMARY: {summary_concept} (score {summary_score:.4f})."
            )

        results: dict[tuple[str, int], str] = {}
        for state in states:
            if self._history_managing_config.save_histories:
                self._save_neuron_histories(state)

            best_concept, best_score = history.get_best_concept(state.concept_history)
            logging.info(
                f"LAYER {state.layer} NEURON {state.neuron_id}"
                f" — BEST: {best_concept} (score {best_score:.4f})."
            )
            history.save_to_file(
                utils.format_as_json_string(state.reasoning),
                state.save_directory,
                history.REASONING_FILENAME,
            )
            history.save_to_file(
                utils.format_as_json_string(state.best_concepts),
                state.save_directory,
                history.BEST_CONCEPTS_FILENAME,
            )
            results[(state.layer, state.neuron_id)] = best_concept

        logging.info(f"Pipeline complete. {len(results)} neurons explained.")
        return results
