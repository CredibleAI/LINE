"""Download and organize ImageNet images using the datasets library.

This script downloads the ImageNet validation set from Hugging Face
(ILSVRC/imagenet-1k, a gated dataset: accept its terms on the Hub and set HF_TOKEN),
processes it, and saves images organized into subdirectories by class label.
The label names are processed to create valid directory names.
"""

import argparse
import logging
import os
from pathlib import Path

from datasets import Image, load_dataset

from line import IMAGENET2012_CLASSES

_LABEL_NAMES_SPECIAL_CASES = {
    "maillot, tank suit": "tank_suit",
    "Cardigan, Cardigan Welsh corgi": "cardigan_welsh_corgi",
}


def _process_label_name(label_name: str) -> str:
    """Process ImageNet label name into a valid directory name."""
    if label_name in _LABEL_NAMES_SPECIAL_CASES:
        return _LABEL_NAMES_SPECIAL_CASES[label_name]
    return label_name.split(",")[0].replace(" ", "_").lower()


def main(output_dir: str | Path) -> None:
    """Download and organize images using Hugging Face datasets."""
    output_dir = Path(output_dir)
    hf_token = os.getenv("HF_TOKEN")
    logging.info("Using Hugging Face token: %s", "set" if hf_token else "not set")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set.")

    logging.info("Loading ImageNet validation dataset...")
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", token=hf_token)

    # Cast the image column to NOT decode. This allows us to access the
    # raw bytes and original filenames, avoiding re-compression via PIL.
    dataset = dataset.cast_column("image", Image(decode=False))

    logging.info("Creating class subdirectories.")
    labels_list = list(IMAGENET2012_CLASSES.values())

    # Pre-create all class directories
    for label_name in labels_list:
        dir_name = _process_label_name(label_name)
        (output_dir / dir_name).mkdir(parents=True, exist_ok=True)

    logging.info("Extracting and saving images to disk...")

    for index, row in enumerate(dataset):
        image_data = row["image"]
        label_idx = row["label"]

        dir_name = _process_label_name(labels_list[label_idx])

        # Because of decode=False, image_data is a dict with 'bytes' and 'path'
        filename = Path(image_data["path"]).name if image_data["path"] else f"{index:08d}.JPEG"
        img_path = output_dir / dir_name / filename

        img_path.write_bytes(image_data["bytes"])

    logging.info(f"All images processed and saved successfully to {output_dir}")


def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and organize ImageNet images from Hugging Face."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.getenv("DATA_DIR", "./data"), "ImageNet-1K"),
        help="Output directory (default: $DATA_DIR/ImageNet-1K; DATA_DIR defaults to ./data).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_config()
    main(args.output_dir)
