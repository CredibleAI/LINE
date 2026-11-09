"""Text-to-image model for generating synthetic images from prompts.

This module provides functionality to load and use diffusion-based text-to-image
models for generating synthetic images based on text prompts and concepts.
"""

import logging

import diffusers
import torch
from PIL import Image
from tqdm import tqdm

from line.models.base import BaseModel
from line.models.prompt_utils import concept_image_prompt
from line.models.utils import gpu_inference_wrapper

_TEXT_TO_IMAGE_MODELS = {
    "stable-diffusion-v1-5/stable-diffusion-v1-5": diffusers.StableDiffusionPipeline,
    "stabilityai/sdxl-turbo": diffusers.AutoPipelineForText2Image,
    "stabilityai/stable-diffusion-xl-base-1.0": diffusers.DiffusionPipeline,
    "black-forest-labs/FLUX.1-dev": diffusers.FluxPipeline,
}
_MODELS_WITHOUT_VARIANT = {diffusers.FluxPipeline, diffusers.StableDiffusionPipeline}
_TORCH_DTYPE = torch.float16
_PIPELINE_VARIANT = "fp16"


class T2I(BaseModel):
    """Text-to-image generation model wrapper.

    Wraps diffusion-based text-to-image models to generate synthetic images
    from text prompts using specified configurations.
    """

    def __init__(
        self,
        model_id: str,
        device: torch.device,
        model_swapping: bool,
        num_inference_steps: int | None,
        guidance_scale: float | None,
        batch_size: int = 2,
    ) -> None:
        """Initialize the image generation model.

        Args:
            model_id: Identifier of the text-to-image model (e.g., 'stabilityai/sd-turbo').
            device: Device to load model on ('cuda' or 'cpu').
            model_swapping: Whether to enable model swapping functionality.
            num_inference_steps: Number of denoising steps for image generation.
            guidance_scale: Classifier-free guidance scale for conditioning strength.
            batch_size: Number of images per pipeline call.
        """
        super().__init__(model_id, device, model_swapping)
        self._batch_size = batch_size
        self._model_kwargs: dict = {}
        if num_inference_steps is not None:
            self._model_kwargs["num_inference_steps"] = num_inference_steps
        if guidance_scale is not None:
            self._model_kwargs["guidance_scale"] = guidance_scale

    def _load(self, **kwargs) -> diffusers.DiffusionPipeline:
        """Load the text-to-image diffusion pipeline.

        Returns:
            Loaded diffusion pipeline on the specified device.
        """
        pipeline_cls = _TEXT_TO_IMAGE_MODELS[self._model_id]
        model = pipeline_cls.from_pretrained(
            self._model_id,
            torch_dtype=_TORCH_DTYPE,
            **(
                {"variant": _PIPELINE_VARIANT}
                if pipeline_cls not in _MODELS_WITHOUT_VARIANT
                else {}
            ),
        )
        if hasattr(model, "set_progress_bar_config"):
            model.set_progress_bar_config(disable=True)

        load_device = torch.device("cpu") if self._model_swapping else self._device
        return model.to(load_device)

    @gpu_inference_wrapper
    def generate_images(
        self,
        prompts: list[str],
        generators: list[torch.Generator],
    ) -> list[Image.Image]:
        """Generate synthetic images from a list of prompts.

        Args:
            prompts: Fully constructed text prompts.
            generators: One generator per prompt; must have the same length as ``prompts``.

        Returns:
            List of PIL Images, length ``len(prompts)``.
        """
        _batch_size = self._batch_size or len(prompts)
        logging.info(f"Generating {len(prompts)} images in batches of {_batch_size}.")

        all_images = []
        n_batches = (len(prompts) + _batch_size - 1) // _batch_size
        for i in tqdm(range(0, len(prompts), _batch_size), total=n_batches, desc="T2I batches"):
            all_images.extend(
                self._model(
                    prompt=prompts[i : i + _batch_size],
                    generator=generators[i : i + _batch_size],
                    **self._model_kwargs,
                ).images
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return all_images

    def generate_images_from_concept(
        self,
        concepts: list[tuple[str, str]],
        n_images: int,
        generators: list[torch.Generator] | None = None,
    ) -> list[Image.Image]:
        """Generate images for a list of (prompt_text, concept) pairs.

        Args:
            concepts: List of (prompt_text, concept) tuples.
            n_images: Number of images to generate per concept.
            generators: ``n_images`` generators, one per copy. Defaults to
                generators seeded ``0 … n_images-1``.

        Returns:
            List of PIL Images, length ``len(concepts) * n_images``.
        """
        if generators is None:
            seeds = list(range(n_images))
        else:
            assert len(generators) == n_images, "Number of generators must match n_images."
            seeds = [g.initial_seed() for g in generators]

        # Generator-major ordering: [c0_g0, c1_g0, …, c_{n-1}_g0, c0_g1, …]
        # A fresh generator is created for every (seed, concept) pair so that
        # a concept's score does not depend on its position in the batch.
        all_generators = [
            torch.Generator(self._device).manual_seed(seed) for seed in seeds for _ in concepts
        ]
        all_prompts = [
            concept_image_prompt(pt, c, seed=seed) for seed in seeds for pt, c in concepts
        ]
        assert len(all_prompts) == len(all_generators) == len(concepts) * n_images, (
            "Mismatch in total prompts and generators."
        )

        return self.generate_images(
            prompts=all_prompts,
            generators=all_generators,
        )
