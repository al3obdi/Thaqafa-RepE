"""Steer generation by injecting a cultural concept vector.

Vectors are loaded from a file written by ``scripts/extract_vectors.py``, or
extracted on the spot when no file is given. The prompt is then generated twice -
once unsteered and once with the concept injected - so the effect of the
injection is visible side by side rather than in isolation.

Configuration comes from the environment (or a local ``.env``, see
``.env.example``). The Hugging Face token is only ever read from ``HF_TOKEN``;
it is never accepted as a command line argument.

Examples:
    Amplify hospitality using a previously extracted vector::

        poetry run python scripts/inject_concepts.py \
            --concept diyafa_001 --strength 1.5 \
            --vectors outputs/vectors/concept_vectors.pt \
            --prompt "What should I do when a guest arrives unannounced?"

    Suppress the concept instead, extracting the vector on the fly::

        poetry run python scripts/inject_concepts.py \
            --concept wasta_001 --strength -1.5 --prompt "How do I get this job?"
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

from src.data.dataset_builder import DEFAULT_DATASET_PATH
from src.models.rep_engine import DEFAULT_BATCH_SIZE, DEFAULT_MODEL_NAME, CulturalRepE
from src.utils.evaluation import DEFAULT_MAX_NEW_TOKENS, generate_steered

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments, using the environment for defaults.

    Args:
        argv: Argument list. ``sys.argv[1:]`` is used when omitted.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept", required=True, help="Concept id to inject.")
    parser.add_argument("--prompt", required=True, help="Prompt to generate a completion for.")
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
    parser.add_argument(
        "--vectors",
        type=Path,
        default=None,
        help="Vector file from extract_vectors.py. Omit to extract on the fly.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.environ.get("DATASET_PATH", DEFAULT_DATASET_PATH)),
        help="Path to the cultural concepts JSONL file.",
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
        "--batch-size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        help="Number of prompts pushed through the model at once during extraction.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="How many tokens to generate.",
    )
    return parser.parse_args(argv)


def load_vectors(engine: CulturalRepE, path: Path) -> None:
    """Populate an engine's cache from a saved vector file.

    Args:
        engine: Engine to populate.
        path: File written by :meth:`CulturalRepE.save_vectors`.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    engine.concept_vectors.update(payload.get("concept_vectors", {}))
    engine.extraction_layers.update(payload.get("extraction_layers", {}))
    logger.info("Loaded %d vectors from %s", len(engine.concept_vectors), path)


def main(argv: list[str] | None = None) -> int:
    """Generate a prompt with and without the concept injected.

    Args:
        argv: Argument list forwarded to :func:`parse_args`.

    Returns:
        A process exit code.
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

    if args.vectors is not None:
        load_vectors(engine, args.vectors)

    if args.concept not in engine.concept_vectors:
        logger.info("No cached vector for %s; extracting it now", args.concept)
        engine.extract_vector(args.concept)

    baseline = generate_steered(engine, args.prompt, max_new_tokens=args.max_new_tokens)

    with engine.steering(args.concept, strength=args.strength, layers=args.layers):
        logger.info(
            "Injecting %s at strength %+.2f into layer(s) %s",
            args.concept,
            args.strength,
            ", ".join(str(name) for name in engine.active_hook_names),
        )
        steered = generate_steered(engine, args.prompt, max_new_tokens=args.max_new_tokens)

    print(f"\n=== baseline (strength 0.0) ===\n{baseline}")
    print(f"\n=== steered (strength {args.strength:+.2f}) ===\n{steered}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
