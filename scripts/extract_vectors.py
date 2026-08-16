"""Extract concept vectors for every concept in the cultural dataset.

Example:
    poetry run python scripts/extract_vectors.py \
        --dataset data/datasets/cultural_concepts.jsonl \
        --layer -1 \
        --output outputs/vectors
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data.dataset_builder import DEFAULT_DATASET_PATH, load_concepts
from src.models.rep_engine import DEFAULT_MODEL_NAME, CulturalRepE

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Argument list. ``sys.argv[1:]`` is used when omitted.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the cultural concepts JSONL file.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face identifier of the base model.",
    )
    parser.add_argument("--device", default="cuda", help="Device to run the model on.")
    parser.add_argument("--dtype", default="bfloat16", help="Torch dtype for the model weights.")
    parser.add_argument("--layer", type=int, default=-1, help="Layer to read activations from.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vectors"),
        help="Directory the extracted vectors are written to.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the extraction pipeline.

    Args:
        argv: Argument list forwarded to :func:`parse_args`.

    Returns:
        A process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    concepts = load_concepts(args.dataset)
    logger.info("Extracting vectors for %d concepts", len(concepts))

    engine = CulturalRepE(model_name=args.model_name, device=args.device, dtype=args.dtype)
    engine.load_model()

    args.output.mkdir(parents=True, exist_ok=True)
    for concept in concepts:
        engine.extract_vector(
            concept=concept.concept_id,
            examples=concept.all_examples,
            layer=args.layer,
        )
        logger.info("Extracted vector for %s (%s)", concept.concept_id, concept.concept_en)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
