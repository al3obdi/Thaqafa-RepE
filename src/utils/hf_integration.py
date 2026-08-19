"""Hugging Face integration for vector storage and Space synchronisation.

This module bridges the local Thaqafa-RepE engine and the Hugging Face
ecosystem. It provides three capabilities:

1. **Save** extracted concept vectors to a private HF Dataset, serialising
   tensors into a columnar schema that is both human-readable and
   machine-loadable.
2. **Load** vectors from that dataset back into a ``dict[str, torch.Tensor]``,
   either in full or filtered by concept ID.
3. **Sync** with the ZeroGPU extraction Space — check its status and pull
   any new results.

The module never reads ``HF_TOKEN`` from source code. The token is always
fetched from the environment (``HF_TOKEN``) or accepted as an explicit
argument. No token is ever logged or included in error messages.

Example:
    >>> from src.utils.hf_integration import save_vectors_to_hf, load_vectors_from_hf
    >>> vectors = {"diyafa_001": torch.randn(4096)}
    >>> url = save_vectors_to_hf(vectors, metadata={"model_name": "llama-3-8b"})
    >>> loaded = load_vectors_from_hf()
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import torch

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "al3obdi/thaqafa-repe-vectors"
DEFAULT_SPACE = "al3obdi/thaqafa-repe-extraction"

# Dataset schema columns — documented in the dataset README and enforced
# here so that downstream consumers can rely on the shape.
REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "concept_id",
        "concept_ar",
        "concept_en",
        "vector",
        "extraction_layer",
        "model_name",
        "extraction_timestamp",
    }
)


class HFIntegrationError(Exception):
    """Raised when Hugging Face integration operations fail."""


class MissingTokenError(HFIntegrationError):
    """Raised when no Hugging Face token is available."""


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------


def _resolve_token(token: str | None = None) -> str:
    """Resolve the Hugging Face access token.

    The token is never logged, printed, or included in exception messages.
    Only its presence or absence is reported.

    Args:
        token: Explicit token. When ``None``, falls back to the ``HF_TOKEN``
            environment variable.

    Returns:
        The resolved token string.

    Raises:
        MissingTokenError: If no token is provided and ``HF_TOKEN`` is not set.
    """
    resolved = token or os.environ.get("HF_TOKEN")
    if not resolved:
        raise MissingTokenError(
            "No Hugging Face token found. Set the HF_TOKEN environment "
            "variable or pass token= explicitly."
        )
    return resolved


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _tensor_to_list(tensor: torch.Tensor) -> list[float]:
    """Convert a 1-D tensor to a list of Python floats.

    Args:
        tensor: A 1-D tensor of shape ``(d_model,)``.

    Returns:
        A list of floats.

    Raises:
        ValueError: If the tensor is not 1-D.
    """
    if tensor.ndim != 1:
        raise ValueError(
            f"Expected a 1-D tensor, got shape {tuple(tensor.shape)}. "
            "Only single-direction concept vectors are supported."
        )
    return tensor.detach().cpu().to(torch.float32).tolist()


def _list_to_tensor(values: list[float]) -> torch.Tensor:
    """Convert a list of floats back to a 1-D float32 tensor.

    Args:
        values: List of float values.

    Returns:
        A 1-D tensor of shape ``(len(values),)`` in float32.
    """
    return torch.tensor(values, dtype=torch.float32)


def _build_dataset_rows(
    vectors: dict[str, torch.Tensor],
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build HF Dataset rows from a concept-vector mapping.

    Each concept produces one row with the schema defined by
    :data:`REQUIRED_COLUMNS`.

    Args:
        vectors: Mapping from ``concept_id`` to the extracted direction.
        metadata: Optional metadata. Recognised keys:

            - ``model_name`` (str): Model used for extraction.
            - ``extraction_layers`` (dict[str, int]): Layer per concept.
            - ``extraction_timestamp`` (str): ISO-8601 timestamp.
            - ``concept_metadata`` (dict[str, dict]): Per-concept ``concept_ar``
              and ``concept_en`` overrides.

    Returns:
        A list of row dictionaries.

    Raises:
        ValueError: If ``vectors`` is empty.
    """
    if not vectors:
        raise ValueError("No vectors to serialise. The mapping is empty.")

    meta = metadata or {}
    model_name = meta.get("model_name", "unknown")
    extraction_layers: dict[str, int] = meta.get("extraction_layers", {})
    timestamp = meta.get("extraction_timestamp", datetime.now(timezone.utc).isoformat())
    concept_meta: dict[str, dict[str, str]] = meta.get("concept_metadata", {})

    rows: list[dict[str, Any]] = []
    for concept_id, vector in vectors.items():
        cm = concept_meta.get(concept_id, {})
        rows.append(
            {
                "concept_id": concept_id,
                "concept_ar": cm.get("concept_ar", ""),
                "concept_en": cm.get("concept_en", ""),
                "vector": _tensor_to_list(vector),
                "extraction_layer": extraction_layers.get(concept_id, -1),
                "model_name": model_name,
                "extraction_timestamp": timestamp,
            }
        )
    return rows


def _rows_to_vectors(
    rows: list[dict[str, Any]],
    concept_ids: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Convert HF Dataset rows back to a concept-vector mapping.

    Args:
        rows: Dataset rows, each containing at least ``concept_id`` and
            ``vector``.
        concept_ids: Optional filter — only these concepts are loaded.

    Returns:
        A mapping from ``concept_id`` to a 1-D float32 tensor.
    """
    wanted = set(concept_ids) if concept_ids else None
    vectors: dict[str, torch.Tensor] = {}
    for row in rows:
        cid = row["concept_id"]
        if wanted is not None and cid not in wanted:
            continue
        vectors[cid] = _list_to_tensor(row["vector"])
    return vectors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_vectors_to_hf(
    vectors: dict[str, torch.Tensor],
    dataset_name: str = DEFAULT_DATASET,
    metadata: dict[str, Any] | None = None,
    token: str | None = None,
    config_name: str | None = None,
) -> str:
    """Save extracted concept vectors to a Hugging Face Dataset.

    Vectors are serialised to a columnar schema (see :data:`REQUIRED_COLUMNS`)
    and pushed as a parquet-backed dataset. Existing data is overwritten.

    Args:
        vectors: Mapping from ``concept_id`` to the extracted direction.
        dataset_name: Target HF dataset repository, e.g.
            ``"al3obdi/thaqafa-repe-vectors"``.
        metadata: Optional metadata dict. See :func:`_build_dataset_rows`
            for recognised keys.
        token: Hugging Face access token. Defaults to ``HF_TOKEN`` env var.
        config_name: Optional dataset config to push under. Pushing replaces
            the target config wholesale, so use one config per model to keep
            extractions from different models from overwriting each other.

    Returns:
        The URL of the updated dataset.

    Raises:
        MissingTokenError: If no token is available.
        HFIntegrationError: If the upload fails.
        ValueError: If ``vectors`` is empty.
    """
    resolved_token = _resolve_token(token)
    rows = _build_dataset_rows(vectors, metadata)

    logger.info("Saving %d vectors to HF dataset %s", len(rows), dataset_name)

    try:
        from datasets import Dataset

        ds = Dataset.from_list(rows)
        if config_name is not None:
            # A named config scopes the overwrite: pushing model B's vectors
            # under its own config leaves model A's config untouched.
            ds.push_to_hub(
                dataset_name, config_name=config_name, token=resolved_token, private=True
            )
        else:
            ds.push_to_hub(dataset_name, token=resolved_token, private=True)
    except ImportError as exc:
        raise HFIntegrationError(
            "The 'datasets' library is required. Install it with 'pip install datasets'."
        ) from exc
    except Exception as exc:
        raise HFIntegrationError(f"Failed to push vectors to {dataset_name}: {exc}") from exc

    url = f"https://huggingface.co/datasets/{dataset_name}"
    logger.info("Vectors available at %s", url)
    return url


def load_vectors_from_hf(
    dataset_name: str = DEFAULT_DATASET,
    concept_ids: list[str] | None = None,
    token: str | None = None,
    config_name: str | None = None,
) -> dict[str, torch.Tensor]:
    """Load concept vectors from a Hugging Face Dataset.

    Args:
        dataset_name: Source HF dataset repository.
        concept_ids: Optional list of concept IDs to load. ``None`` loads
            everything.
        token: Hugging Face access token. Defaults to ``HF_TOKEN`` env var.
        config_name: Optional dataset config to read, matching the one used
            at save time. ``None`` reads the default config.

    Returns:
        A mapping from ``concept_id`` to a 1-D float32 tensor.

    Raises:
        MissingTokenError: If no token is available.
        HFIntegrationError: If the download or parsing fails.
    """
    resolved_token = _resolve_token(token)

    logger.info("Loading vectors from HF dataset %s", dataset_name)

    try:
        from datasets import load_dataset

        if config_name is not None:
            ds = load_dataset(dataset_name, config_name, token=resolved_token)
        else:
            ds = load_dataset(dataset_name, token=resolved_token)
        # datasets returns a DatasetDict; take the first split
        split_name = list(ds.keys())[0] if hasattr(ds, "keys") else "train"
        rows: list[dict[str, Any]] = [dict(row) for row in ds[split_name]]
    except ImportError as exc:
        raise HFIntegrationError(
            "The 'datasets' library is required. Install it with 'pip install datasets'."
        ) from exc
    except Exception as exc:
        raise HFIntegrationError(f"Failed to load dataset {dataset_name}: {exc}") from exc

    vectors = _rows_to_vectors(rows, concept_ids)
    logger.info("Loaded %d vectors from %s", len(vectors), dataset_name)
    return vectors


def sync_with_space(
    space_name: str = DEFAULT_SPACE,
    token: str | None = None,
) -> dict[str, Any]:
    """Check Space status and sync local vectors with Space results.

    Queries the HF API for the Space's runtime status (stage, hardware,
    last build). If the Space has produced new vectors in its linked
    dataset, they can be pulled with :func:`load_vectors_from_hf`.

    Args:
        space_name: HF Space repository, e.g.
            ``"al3obdi/thaqafa-repe-extraction"``.
        token: Hugging Face access token. Defaults to ``HF_TOKEN`` env var.

    Returns:
        A dictionary with keys:

            - ``space_url`` (str): URL of the Space.
            - ``stage`` (str): Current runtime stage (e.g. ``"RUNNING"``).
            - ``hardware`` (str | None): Requested hardware flavor.
            - ``last_updated`` (str | None): ISO timestamp of last build.
            - ``dataset_url`` (str): URL of the linked dataset.

    Raises:
        MissingTokenError: If no token is available.
        HFIntegrationError: If the API request fails.
    """
    resolved_token = _resolve_token(token)

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=resolved_token)
        info = api.space_info(space_name)
    except ImportError as exc:
        raise HFIntegrationError(
            "The 'huggingface_hub' library is required. "
            "Install it with 'pip install huggingface_hub'."
        ) from exc
    except Exception as exc:
        raise HFIntegrationError(f"Failed to query Space {space_name}: {exc}") from exc

    runtime = getattr(info, "runtime", None)
    stage = getattr(runtime, "stage", "unknown") if runtime else "unknown"
    # runtime.hardware has been a plain string, a dataclass and a dict across
    # huggingface_hub releases; normalise all three without crashing.
    raw_hardware = getattr(runtime, "hardware", None) if runtime else None
    if isinstance(raw_hardware, dict):
        hardware = raw_hardware.get("requested") or raw_hardware.get("current")
    elif raw_hardware is None:
        hardware = None
    else:
        hardware = getattr(raw_hardware, "requested", None) or str(raw_hardware)

    return {
        "space_url": f"https://huggingface.co/spaces/{space_name}",
        "stage": stage,
        "hardware": hardware,
        "last_updated": getattr(info, "last_modified", None),
        "dataset_url": f"https://huggingface.co/datasets/{DEFAULT_DATASET}",
    }


def validate_dataset_schema(
    dataset_name: str = DEFAULT_DATASET,
    token: str | None = None,
) -> bool:
    """Validate that a HF Dataset matches the expected schema.

    Checks that all columns in :data:`REQUIRED_COLUMNS` are present.

    Args:
        dataset_name: HF dataset repository to validate.
        token: Hugging Face access token. Defaults to ``HF_TOKEN`` env var.

    Returns:
        ``True`` if the schema matches.

    Raises:
        MissingTokenError: If no token is available.
        HFIntegrationError: If the dataset cannot be loaded or schema
            validation fails.
    """
    resolved_token = _resolve_token(token)

    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, token=resolved_token, split="train")
        columns = set(ds.column_names)
    except ImportError as exc:
        raise HFIntegrationError(
            "The 'datasets' library is required. Install it with 'pip install datasets'."
        ) from exc
    except Exception as exc:
        raise HFIntegrationError(f"Failed to load dataset {dataset_name}: {exc}") from exc

    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise HFIntegrationError(
            f"Dataset {dataset_name} is missing columns: {sorted(missing)}. "
            f"Expected: {sorted(REQUIRED_COLUMNS)}"
        )
    return True
