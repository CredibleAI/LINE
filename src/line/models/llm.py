"""Large language model wrapper for concept generation and reasoning.

This module provides functionality to use a language model to generate and reason
about concepts based on activation patterns and generation history.
"""

import logging
import warnings
from pathlib import Path

import torch
import transformers
from tqdm import tqdm

from line.models import prompt_utils
from line.models.base import BaseModel
from line.models.utils import gpu_inference_wrapper

IGNORE_SENTINEL = "(ignore this)"
_ASSISTANT_TAG = "<|start_header_id|>assistant<|end_header_id|>"
_EOT_TAG = "<|eot_id|>"
_THINKING_TAG_OPEN = "<thinking>"
_THINKING_TAG_CLOSE = "</thinking>"
_ANSWER_TAG_OPEN = "<answer>"
_ANSWER_TAG_CLOSE = "</answer>"
_TORCH_DTYPE = torch.bfloat16
_TEMPERATURE = 0.5
_TOP_P = 0.9


class LLM(BaseModel):
    """Language model for generating and reasoning about neuron concepts.

    Uses a large language model to generate concept names and explanations
    based on activation history and concept scores.
    """

    def __init__(
        self,
        model_id: str,
        device: torch.device,
        model_swapping: bool,
        max_new_tokens: int,
        prompt_path: str,
        summary_prompt_path: str,
        batch_size: int = 0,
        deterministic: bool = False,
    ) -> None:
        """Initialize the language model.

        Args:
            model_id: Identifier of the language model (e.g., 'meta-llama/Llama-2-7b-chat').
            device: Device to load model on ('cuda' or 'cpu').
            model_swapping: Whether to enable model swapping functionality.
            max_new_tokens: Maximum number of tokens to generate per response.
            prompt_path: Path to the prompt template file for generation.
            summary_prompt_path: Path to the prompt template for summary generation.
            batch_size: Number of prompts per model call. 0 means process all at once.
            deterministic: If True, use greedy decoding (do_sample=False) — selects the
                highest-probability token at each step with no sampling parameters.
                Enables reproducible outputs.
        """
        for path, name in [
            (prompt_path, "prompt_path"),
            (summary_prompt_path, "summary_prompt_path"),
        ]:
            if not Path(path).is_file():
                raise FileNotFoundError(f"{name}={path!r} does not exist.")
        super().__init__(model_id, device, model_swapping)
        self._max_new_tokens = max_new_tokens
        self._prompt_template = prompt_utils._read_prompt_template(prompt_path)
        self._summary_prompt_template = prompt_utils._read_prompt_template(summary_prompt_path)
        self._batch_size = batch_size
        self._deterministic = deterministic

    def _send_to_device(self, device: torch.device) -> None:
        """Move the underlying model to the specified device.

        ``transformers.Pipeline`` has no ``.to()`` method, so this override
        moves ``self._model.model`` (the ``nn.Module``) directly and updates
        the pipeline's device attribute so inference runs on the right device.

        Args:
            device: Target device to move the model to.
        """
        logging.info(f"Sending model {self._model_id} to device {device}.")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*use another device for inference.*")
            self._model.model.to(device)
        self._model.device = torch.device(device)
        self._device = device

    def _load(self, **kwargs) -> transformers.Pipeline:
        """Load the language model pipeline.

        Initializes a text-generation pipeline with the specified model and
        configures tokenizer settings for LLaMA models.

        Returns:
            The loaded language model.
        """
        # When model_swapping is on, load on CPU: the model only moves to GPU
        # during generate_concept() (via gpu_inference_wrapper) and returns to CPU after.
        load_device = torch.device("cpu") if self._model_swapping else self._device

        pipeline = transformers.pipeline(
            task="text-generation",
            model=self._model_id,
            device=load_device,
            dtype=_TORCH_DTYPE,
        )

        if "llama" in self._model_id:
            assert pipeline.tokenizer is not None
            pipeline.tokenizer.pad_token_id = (
                pipeline.model.config.eos_token_id[0]  # type: ignore
            )
        else:
            raise ValueError(
                f"Unsupported model_id {self._model_id}. Only LLaMA models are supported for now."
            )

        return pipeline

    @gpu_inference_wrapper
    def generate_concept(
        self,
        concept_histories: list[dict[str, float]],
        generation_histories: list[list[tuple[str, float]]],
        top_k: int | None = None,
    ) -> list[tuple[str, str]]:
        """Generate a new concept name with reasoning using the language model.

        Args:
            concept_histories: List of scoreboards, one per neuron.
            generation_histories: List of generation histories, one per neuron.
            top_k: Optional number of top concepts to use in summary mode.
                   If None, uses the full prompt template.

        Returns:
            List of (concept_name, reasoning_text) tuples, one per neuron.
        """
        prompt_template = (
            self._summary_prompt_template if top_k is not None else self._prompt_template
        )
        generation_prompts = [
            prompt_utils.generate_prompt(
                concept_history,
                generation_history,
                prompt_template,
                top_k,
            )
            for concept_history, generation_history in zip(
                concept_histories, generation_histories, strict=True
            )
        ]

        # For LLaMA models, use the tokenizer's chat template to format the prompt.
        tokenizer = self._model.tokenizer
        assert tokenizer is not None
        messages = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": generation_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for generation_prompt in generation_prompts
        ]
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids(_EOT_TAG),
        ]

        effective_batch_size = self._batch_size if self._batch_size > 0 else len(messages)
        logging.info(f"Generating concepts for {len(messages)} neurons.")
        n_batches = (len(messages) + effective_batch_size - 1) // effective_batch_size
        results = []
        for i in tqdm(
            range(0, len(messages), effective_batch_size), total=n_batches, desc="LLM batches"
        ):
            sample_kwargs = (
                {"do_sample": False}
                if self._deterministic
                else {"do_sample": True, "temperature": _TEMPERATURE, "top_p": _TOP_P}
            )
            batch_outputs = self._model(
                messages[i : i + effective_batch_size],
                max_new_tokens=self._max_new_tokens,
                eos_token_id=terminators,
                **sample_kwargs,
            )
            for output in batch_outputs:
                if isinstance(output, list):
                    output = output[0]
                generated_text = output["generated_text"]
                if _ASSISTANT_TAG not in generated_text:
                    logging.warning(
                        f"_ASSISTANT_TAG not found in generated text. "
                        f"Raw output: {generated_text!r}"
                    )
                    results.append((IGNORE_SENTINEL, ""))
                    continue
                response = generated_text.split(_ASSISTANT_TAG)[-1].strip()
                try:
                    if _THINKING_TAG_OPEN not in response or _THINKING_TAG_CLOSE not in response:
                        raise IndexError
                    if _ANSWER_TAG_OPEN not in response:
                        raise IndexError
                    reasoning = (
                        response.split(_THINKING_TAG_OPEN)[1].split(_THINKING_TAG_CLOSE)[0].strip()
                    )
                    answer = response.split(_ANSWER_TAG_OPEN)[1].split(_ANSWER_TAG_CLOSE)[0]
                    # Strip partial closing tag if max_new_tokens cut off before </answer>
                    if "</" in answer:
                        answer = answer.rsplit("</", 1)[0]
                    answer = answer.strip()
                except IndexError:
                    logging.warning(
                        "LLM response missing expected <thinking>/<answer> tags. "
                        f"Raw response: {response!r}"
                    )
                    reasoning = ""
                    answer = IGNORE_SENTINEL
                results.append((answer, reasoning))

        assert len(results) == len(concept_histories) == len(generation_histories)
        return results
