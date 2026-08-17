#!/usr/bin/env python3
"""Automated local-to-space extraction pipeline for Thaqafa-RepE.

This script connects to the Hugging Face ZeroGPU Space
(al3obdi/thaqafa-repe-extraction) using ``gradio_client``, submits a
vector-extraction job, waits for completion, and then pulls the results
back from the linked HF Dataset into the local environment.

Usage:
    python scripts/run_space_extraction.py --concepts wasta_001,diyafa_001
    python scripts/run_space_extraction.py --concepts muruah_001 --model allam-ai/ALLaM-1-7b-Instruct
    python scripts/run_space_extraction.py --concepts wasta_001 --dataset al3obdi/thaqafa-repe-vectors --space al3obdi/thaqafa-repe-extraction

The script requires the ``HF_TOKEN`` environment variable to be set.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("space_extraction")

DEFAULT_SPACE = "al3obdi/thaqafa-repe-extraction"
DEFAULT_DATASET = "al3obdi/thaqafa-repe-vectors"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
POLL_INTERVAL = 5  # seconds between status checks
MAX_WAIT = 600  # 10 minutes max wait for a job


def _resolve_token() -> str:
    """Resolve the HF token from the environment.

    Returns:
        The token string.

    Raises:
        SystemExit: If no token is found.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "ERROR: HF_TOKEN environment variable is not set. "
            "Set it with: export HF_TOKEN=hf_your_token",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return token


def _create_client(space_name: str, token: str) -> Any:
    """Create a gradio_client connection to the Space.

    Args:
        space_name: HF Space repository identifier.
        token: Hugging Face access token.

    Returns:
        A configured ``gradio_client.Client`` instance.

    Raises:
        SystemExit: If the connection fails.
    """
    try:
        from gradio_client import Client
    except ImportError as exc:
        print(
            f"ERROR: gradio_client is not installed. Install with: pip install gradio_client\n{exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    space_url = "https://al3obdi-thaqafa-repe-extraction.hf.space"
    try:
        client = Client(space_url, hf_token=token)
        logger.info("Connected to Space: %s", space_name)
        return client
    except Exception as exc:
        print(
            f"ERROR: Failed to connect to Space {space_name}: {exc}\n"
            "The Space may be sleeping or in an error state. "
            "Check: https://huggingface.co/spaces/" + space_name,
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def _submit_job(
    client: Any,
    concept_ids: str,
    model_name: str,
    dataset_name: str,
) -> Any:
    """Submit an extraction job to the Space.

    The Gradio app's first tab has a button click handler at index 0
    that accepts ``(concept_ids, model_name, dataset_name)`` and returns
    a status string.

    Args:
        client: Connected ``gradio_client.Client`` instance.
        concept_ids: Comma-separated concept IDs.
        model_name: Hugging Face model identifier.
        dataset_name: Target HF dataset.

    Returns:
        The job handle from ``client.submit()``.

    Raises:
        SystemExit: If submission fails.
    """
    try:
        job = client.submit(
            fn_index=0,
            inputs=[concept_ids, model_name, dataset_name],
        )
        logger.info("Job submitted for concepts: %s", concept_ids)
        return job
    except Exception as exc:
        print(f"ERROR: Failed to submit extraction job: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _wait_for_job(job: Any) -> str:
    """Poll a submitted job until completion.

    Args:
        job: The job handle returned by ``client.submit()``.

    Returns:
        The job result as a string.

    Raises:
        SystemExit: If the job times out or fails.
    """
    print("Waiting for extraction to complete...", flush=True)
    elapsed = 0
    while elapsed < MAX_WAIT:
        status = job.status()
        # gradio_client Job.status() returns a JobStatus object
        # with a .code attribute (PENDING, IN_PROGRESS, SUCCESS, ERROR)
        code = getattr(status, "code", None)
        if code == "SUCCESS":
            result = job.result()
            print(f"✅ Job completed: {result}", flush=True)
            return str(result)
        if code == "ERROR":
            print(f"❌ Job failed: {status}", file=sys.stderr)
            raise SystemExit(1)
        # Still running
        print(f"  [{elapsed}s] Status: {code}...", flush=True)
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    print(
        f"ERROR: Job timed out after {MAX_WAIT}s. "
        "The Space may be cold-starting or the model is too large.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _load_results(
    concept_ids: list[str],
    dataset_name: str,
    token: str,
) -> dict[str, Any]:
    """Load extracted vectors from the HF Dataset.

    Args:
        concept_ids: List of concept IDs that were extracted.
        dataset_name: Source HF dataset.
        token: Hugging Face access token.

    Returns:
        A dict mapping concept_id to vector info.
    """
    try:
        from src.utils.hf_integration import load_vectors_from_hf
    except ImportError:
        # When running as a standalone script outside the project

        # Try to load from src/ relative to the project root
        project_root = _find_project_root()
        if project_root:
            sys.path.insert(0, str(project_root))
            from src.utils.hf_integration import load_vectors_from_hf
        else:
            print(
                "WARNING: Could not import *** module. "
                "Vectors are in the HF Dataset but not loaded locally.",
                file=sys.stderr,
            )
            return {}

    try:
        vectors = load_vectors_from_hf(
            dataset_name=dataset_name,
            concept_ids=concept_ids,
            token=token,
        )
        return vectors
    except Exception as exc:
        print(f"WARNING: Failed to load vectors from HF Dataset: {exc}", file=sys.stderr)
        return {}


def _find_project_root() -> Path | None:
    """Find the project root by looking for pyproject.toml.

    Returns:
        The project root path, or ``None`` if not found.
    """
    from pathlib import Path

    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return None


def _print_summary(vectors: dict[str, Any]) -> None:
    """Print a summary of loaded vectors.

    Args:
        vectors: Mapping from concept_id to tensor.
    """
    if not vectors:
        print("\n📋 No vectors loaded. Check the HF Dataset for results.")
        return

    print(f"\n{'=' * 60}")
    print(f"📋 Loaded {len(vectors)} vectors from HF Dataset")
    print(f"{'=' * 60}")
    for concept_id, vec in vectors.items():
        shape = tuple(vec.shape) if hasattr(vec, "shape") else "unknown"
        norm = float(vec.norm().item()) if hasattr(vec, "norm") else "unknown"
        print(f"  {concept_id}: shape={shape}, norm={norm:.4f}")
    print(f"{'=' * 60}")
    print("Vectors are ready for steering/probing experiments.")
    print("Use: engine.concept_vectors = vectors")


def main() -> None:
    """Entry point for the CLI script."""
    parser = argparse.ArgumentParser(
        description="Trigger vector extraction on the ZeroGPU Space and pull results locally.",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        required=True,
        help="Comma-separated concept IDs (e.g. wasta_001,diyafa_001)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model identifier (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"HF dataset name (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--space",
        type=str,
        default=DEFAULT_SPACE,
        help=f"HF Space name (default: {DEFAULT_SPACE})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 1. Resolve token
    token = _resolve_token()

    # 2. Parse concept IDs
    concept_ids = [c.strip() for c in args.concepts.split(",") if c.strip()]
    if not concept_ids:
        print("ERROR: No valid concept IDs provided.", file=sys.stderr)
        raise SystemExit(1)

    print("🚀 Thaqafa-RepE Space Extraction")
    print(f"   Concepts: {', '.join(concept_ids)}")
    print(f"   Model: {args.model}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Space: {args.space}")
    print()

    # 3. Connect to Space
    client = _create_client(args.space, token)

    # 4. Submit extraction job
    job = _submit_job(client, args.concepts, args.model, args.dataset)

    # 5. Wait for completion
    _wait_for_job(job)

    # 6. Load results from HF Dataset
    print("\n📥 Loading vectors from HF Dataset...")
    vectors = _load_results(concept_ids, args.dataset, token)

    # 7. Print summary
    _print_summary(vectors)


if __name__ == "__main__":
    main()
