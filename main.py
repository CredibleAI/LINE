"""Main entry point for the LINE pipeline.

This module orchestrates the neuron concept attribution pipeline, which explains
the behavior of individual neurons by generating and refining concept descriptions
using language and image generation models.
"""

import argparse
import logging
import os
import time

import torch

from explained_models import get_model
from line import LINE, ORIGINAL_PROMPT_DIR, RunConfig
from line.pipeline.history import read_params, save_params
from transform_images import get_transform


def parse_neurons(specs: list[str]) -> dict[str, list[int]]:
    """Parse ``--neurons`` values into a layer→neuron-IDs mapping.

    Each spec has the form ``layer:id,id,...``.
    Example: ``["avgpool:0,1,2", "layer4:5,10"]``
    """
    result: dict[str, list[int]] = {}
    for spec in specs:
        layer, _, ids = spec.partition(":")
        if not layer:
            raise ValueError(
                f"Invalid --neurons entry '{spec}'. Expected format: 'layer:id,id,...'"
            )
        if not ids:
            raise ValueError(f"No neuron IDs provided for layer '{layer}'.")
        result[layer] = [int(i) for i in ids.split(",")]
    return result


def get_config() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LINE: neuron concept attribution pipeline.")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run the pipeline on. Select a GPU with CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--t2i_model",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="model_id of the text2image model.",
    )
    parser.add_argument(
        "--explained_model",
        default="resnet18",
        help="model_id of the explained model.",
    )
    parser.add_argument(
        "--language_model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="model_id of the language model.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per LLM response.",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=os.path.join(ORIGINAL_PROMPT_DIR, "prompt_main.txt"),
        help="Path to the prompt template file for LLM concept generation.",
    )
    parser.add_argument(
        "--summary_prompt_path",
        type=str,
        default=os.path.join(ORIGINAL_PROMPT_DIR, "prompt_summary.txt"),
        help="Path to the prompt template file for LLM summary generation.",
    )
    parser.add_argument(
        "--model_swapping",
        action="store_true",
        default=False,
        help=(
            "If set, the models used will be sent to CPU while not performing inference"
            " and sent to GPU only for inference. Useful when the models don't fit into"
            " the VRAM all at once."
        ),
    )
    parser.add_argument(
        "--num_inf_steps",
        type=int,
        default=None,
        help="Number of inference steps in diffusion.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Guidance scale for stable diffusion.",
    )
    parser.add_argument(
        "--num_img",
        type=int,
        default=5,
        help="Number of images to generate per concept per iteration.",
    )
    parser.add_argument(
        "--prompt_text",
        default="A realistic photo of a",
        help="Base prompt prefix used when generating images.",
    )
    parser.add_argument(
        "--t2i_batch_size",
        type=int,
        default=1,
        help=(
            "Batch size for T2I image generation. 0 means no batching (full batch at once),"
            " which needs a lot of VRAM: the initial step generates up to"
            " initial_h_size * num_img images per neuron."
        ),
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=0,
        help="Batch size for LLM inference. 0 means no batching.",
    )
    parser.add_argument(
        "--deterministic_llm",
        action="store_true",
        default=False,
        help=(
            "If set, use greedy decoding for the LLM (do_sample=False) instead of"
            " sampling. Produces reproducible outputs at the cost of diversity."
        ),
    )
    parser.add_argument(
        "--explained_model_batch_size",
        type=int,
        default=0,
        help="Batch size for the explained model forward pass. 0 means no batching.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--control_activations_path",
        default="control_activations",
        help="Path to directory where control activations are stored.",
    )
    parser.add_argument(
        "--imagenet_data_path",
        default=None,
        help=(
            "Path to ImageNet-style control images used to compute control activations."
            " Required when activations have not been pre-computed."
        ),
    )
    parser.add_argument(
        "--h_size",
        type=int,
        default=0,
        help="Maximum scoreboard size during iterative updates. 0 means unbounded.",
    )
    parser.add_argument(
        "--initial_h_size",
        type=int,
        default=10,
        help="Number of concepts to seed the scoreboard with at startup. 0 means use all.",
    )
    parser.add_argument(
        "--best_ratio",
        type=float,
        default=0.5,
        help="Proportion of h_size slots filled with top-scoring concepts.",
    )
    parser.add_argument(
        "--init_pool_ratio",
        type=float,
        default=1.0,
        help=(
            "Fraction (0, 1] of control-concept classes made available to seed the"
            " scoreboard at initialization. 1.0 uses the full bank (default)."
        ),
    )
    parser.add_argument(
        "--neurons",
        type=str,
        nargs="+",
        default=["avgpool:0"],
        help=(
            "Neurons to explain, as space-separated 'layer:id,id,...' entries. "
            "Example: --neurons avgpool:0,1,2 layer4:5,10"
        ),
    )
    parser.add_argument(
        "--n_iters",
        type=int,
        default=10,
        help="Number of iterations.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top concepts passed to the LLM in the final summary step.",
    )
    parser.add_argument(
        "--metric",
        default="AVG_ACTIVATION",
        choices=["AVG_ACTIVATION"],
        help="Scoring metric used to evaluate concepts.",
    )
    parser.add_argument(
        "--save_histories",
        action="store_true",
        default=False,
        help=(
            "If set, generation history and final concept history of the LLM will be"
            " saved to a file."
        ),
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        default=False,
        help="If set, images from each iteration of the pipeline will be saved.",
    )
    parser.add_argument(
        "--save_dir",
        default="results",
        help="Path where pipeline artifacts will be stored.",
    )
    return parser


def main():
    """Run the neuron concept attribution pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = get_config().parse_args()

    neurons = parse_neurons(args.neurons)
    logging.info(f"Neurons to explain: {neurons}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA is not available, falling back to CPU.")
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    transforms = get_transform(args.explained_model)

    explained_model = get_model(
        args.explained_model,
        layers=list(neurons.keys()),
        device=device,
        model_swapping=args.model_swapping,
        transforms=transforms,
    )
    logging.info(f"Explained model '{args.explained_model}' loaded.")

    run_config = RunConfig.from_namespace(args, neurons)
    line = LINE(run_config, explained_model, device)

    # Run the pipeline and measure total runtime
    start_time = time.perf_counter()
    results = line.run_pipeline(seed=args.seed)
    end_time = time.perf_counter()

    # Log final results
    for (layer, neuron_id), concept in results.items():
        logging.info(f"Layer: {layer}, Neuron ID: {neuron_id}, Final Concept: {concept}")

    # Update the saved parameters file with the total runtime of the pipeline
    run_params = read_params(line.save_directory)
    run_params["total_time_seconds"] = end_time - start_time
    save_params(run_params, line.save_directory)


if __name__ == "__main__":
    main()
