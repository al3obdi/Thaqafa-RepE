"""Evaluate how injection strength affects cultural alignment.

The script sweeps a range of injection strengths for one concept and reports a
placeholder alignment metric for each. The metric itself is part of the
research phase; the sweep harness around it is stable.

Example:
    poetry run python scripts/evaluate.py --concept diyafa_001 --min -2 --max 2 --steps 9
"""

from __future__ import annotations

import argparse
import logging

from src.models.rep_engine import DEFAULT_MODEL_NAME, CulturalRepE

logger = logging.getLogger(__name__)


def build_strength_grid(minimum: float, maximum: float, steps: int) -> list[float]:
    """Return ``steps`` evenly spaced strengths between ``minimum`` and ``maximum``.

    Args:
        minimum: Lowest injection strength in the sweep.
        maximum: Highest injection strength in the sweep.
        steps: Number of points, must be at least two.

    Returns:
        The strength grid, in ascending order.

    Raises:
        ValueError: If ``steps`` is smaller than two or the range is empty.
    """
    if steps < 2:
        raise ValueError("steps must be at least 2")
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum")

    stride = (maximum - minimum) / (steps - 1)
    return [minimum + stride * index for index in range(steps)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Argument list. ``sys.argv[1:]`` is used when omitted.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept", required=True, help="Concept identifier to evaluate.")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face identifier of the base model.",
    )
    parser.add_argument("--device", default="cuda", help="Device to run the model on.")
    parser.add_argument("--dtype", default="bfloat16", help="Torch dtype for the model weights.")
    parser.add_argument("--min", dest="minimum", type=float, default=-2.0)
    parser.add_argument("--max", dest="maximum", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=9)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the strength sweep.

    Args:
        argv: Argument list forwarded to :func:`parse_args`.

    Returns:
        A process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    grid = build_strength_grid(args.minimum, args.maximum, args.steps)
    logger.info("Evaluating %s over %d strengths", args.concept, len(grid))

    engine = CulturalRepE(model_name=args.model_name, device=args.device, dtype=args.dtype)
    engine.load_model()

    for strength in grid:
        engine.inject_vector(concept=args.concept, strength=strength)
        logger.info("strength=%.2f", strength)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
