"""Steer generation by injecting a previously extracted concept vector.

Example:
    poetry run python scripts/inject_concepts.py \
        --concept diyafa_001 \
        --strength 1.5 \
        --prompt "What should I do when a guest arrives unannounced?"
"""

from __future__ import annotations

import argparse
import logging

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
    parser.add_argument("--concept", required=True, help="Concept identifier to inject.")
    parser.add_argument("--prompt", required=True, help="Prompt to generate a completion for.")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face identifier of the base model.",
    )
    parser.add_argument("--device", default="cuda", help="Device to run the model on.")
    parser.add_argument("--dtype", default="bfloat16", help="Torch dtype for the model weights.")
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Injection coefficient. Negative values suppress the concept.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Layers to inject into. Defaults to the extraction layer.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run a single steered generation.

    Args:
        argv: Argument list forwarded to :func:`parse_args`.

    Returns:
        A process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    engine = CulturalRepE(model_name=args.model_name, device=args.device, dtype=args.dtype)
    engine.load_model()

    logger.info("Injecting %s at strength %.2f", args.concept, args.strength)
    engine.inject_vector(concept=args.concept, strength=args.strength, layers=args.layers)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
