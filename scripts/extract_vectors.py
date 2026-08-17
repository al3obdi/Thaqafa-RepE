"""Extract contrastive concept vectors for the cultural concept dataset.

Configuration is read from the environment (or a local ``.env`` file, see
``.env.example``) and can be overridden per run on the command line. The
Hugging Face token is only ever read from ``HF_TOKEN``; it is never accepted as
a command line argument, so it cannot leak into shell history or process
listings.

Examples:
    Extract every concept at the middle layer::

        poetry run python scripts/extract_vectors.py --output outputs/vectors

    Extract a single concept at an explicit layer::

        poetry run python scripts/extract_vectors.py \
            --concept diyafa_001 --layer 14 --device cuda
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.data.dataset_builder import DEFAULT_DATASET_PATH
from src.models.rep_engine import DEFAULT_BATCH_SIZE, DEFAULT_MODEL_NAME, CulturalRepE

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("outputs/vectors")
VECTOR_FILE_NAME = "concept_vectors.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments, using the environment for defaults.

    Args:
        argv: Argument list. ``sys.argv[1:]`` is used when omitted.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.environ.get("DATASET_PATH", DEFAULT_DATASET_PATH)),
        help="Path to the cultural concepts JSONL file.",
    )
    parser.add_argument(
        "--concept",
        default=None,
        help="Concept id to extract. Omit to extract every concept in the dataset.",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("BASE_MODEL_NAME", DEFAULT_MODEL_NAME),
        help="Hugging Face identifier of the base model.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "cuda"),
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--dtype",
        default=os.environ.get("DTYPE", "bfloat16"),
        help="Torch dtype for the model weights.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer to read activations from. Defaults to the middle layer.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        help="Number of prompts pushed through the model at once.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Keep the raw mean difference instead of scaling it to unit L2 norm.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)),
        help="Directory the extracted vectors are written to.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the extraction pipeline.

    Args:
        argv: Argument list forwarded to :func:`parse_args`.

    Returns:
        A process exit code: ``0`` on success, ``1`` when nothing could be
        extracted.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    args = parse_args(argv)

    engine = CulturalRepE(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        hf_token=os.environ.get("HF_TOKEN") or None,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
    )
    engine.load_model()

    normalize = not args.no_normalize
    if args.concept:
        engine.extract_vector(concept=args.concept, layer=args.layer, normalize=normalize)
    else:
        engine.extract_all_vectors(layer=args.layer, normalize=normalize)

    if not engine.concept_vectors:
        logger.error("No vectors were extracted; check the dataset at %s", args.dataset)
        return 1

    destination = engine.save_vectors(args.output / VECTOR_FILE_NAME)
    for concept_id, vector in engine.concept_vectors.items():
        logger.info(
            "%s: layer %d, %d dimensions",
            concept_id,
            engine.extraction_layers[concept_id],
            vector.shape[0],
        )
    logger.info("Wrote %d vectors to %s", len(engine.concept_vectors), destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
