"""Sample random neuron IDs for a given model and layer.

Prints a space-separated list of neuron IDs to stdout.
All diagnostic output goes to stderr so stdout can be captured by the caller.
"""

import argparse
import contextlib
import logging
import sys

import numpy as np
import torch

from explained_models import get_model


@contextlib.contextmanager
def _stdout_to_stderr():
    """Redirect stdout to stderr so third-party print() calls don't pollute captured output."""
    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old


def get_layer_size(model_id: str, layer: str, device: torch.device) -> int:
    """Return the number of neurons in `layer` via a dummy forward pass."""
    logging.info(f"Probing layer size: model={model_id} layer={layer} device={device}")
    with _stdout_to_stderr():
        model = get_model(
            model_id,
            layers=[layer],
            device=device,
            model_swapping=False,
            transforms=None,
        )
    dummy = torch.zeros(1, 3, 224, 224)
    activations = model.get_activations(dummy, batch_size=0)
    size = activations[layer].shape[1]
    logging.info(f"Layer '{layer}' has {size} neurons.")
    return size


def sample_neuron_ids(n: int, layer_size: int, seed: int) -> list[int]:
    if n > layer_size:
        logging.warning(
            f"Requested {n} neurons but layer only has {layer_size}; capping to {layer_size}."
        )
        n = layer_size
    logging.info(f"Sampling {n} neuron IDs from {layer_size} (seed={seed}).")
    rng = np.random.default_rng(seed)
    ids = sorted(rng.choice(layer_size, size=n, replace=False).tolist())
    logging.info(f"Sampled neuron IDs: {ids}")
    return ids


def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample random neuron IDs for a model layer.")
    parser.add_argument("--explained_model", required=True)
    parser.add_argument("--layer", required=True, help="Single layer name to probe.")
    parser.add_argument("--n", type=int, required=True, help="Number of random neurons to sample.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="If set, split the sampled IDs into chunks of this size and output only one chunk.",
    )
    parser.add_argument(
        "--chunk_index",
        type=int,
        default=0,
        help="Zero-based index of the chunk to output (requires --chunk_size).",
    )
    parser.add_argument(
        "--just_numbers",
        action="store_true",
        help="If set print only space-separated neuron IDs without the layer name prefix.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_config()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and args.device == "cuda"
        else torch.device("cpu")
    )
    logging.info(f"Using device: {device}")

    layer_size = get_layer_size(args.explained_model, args.layer, device)
    neuron_ids = sample_neuron_ids(args.n, layer_size, args.seed)

    if args.chunk_size is not None:
        start = args.chunk_index * args.chunk_size
        end = start + args.chunk_size
        neuron_ids = neuron_ids[start:end]
        logging.info(f"Chunk {args.chunk_index}: neurons[{start}:{end}] -> {len(neuron_ids)} IDs")

    # Only the spec goes to stdout — captured directly by the shell caller.
    if args.just_numbers:
        print(" ".join(str(i) for i in neuron_ids))
    else:
        print(f"{args.layer}:{','.join(str(i) for i in neuron_ids)}")


if __name__ == "__main__":
    main()
