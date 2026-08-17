"""Gradio interface for Zero-GPU vector extraction on Hugging Face Spaces.

This Space runs on ZeroGPU (A10G) when hardware is set to ``zero-a10g``.
It provides a web UI for extracting cultural concept vectors using the
Thaqafa-RepE engine and pushing results to a private HF Dataset.

Usage:
    Deploy as a Hugging Face Space. Set ``HF_TOKEN`` as a Space secret
    (Settings → Repository secrets). The Space handles the rest.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATASET = "al3obdi/thaqafa-repe-vectors"
DEFAULT_CONCEPTS = "wasta_001, muruah_001, diyafa_001"

MODEL_CHOICES = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "allam-ai/ALLaM-1-7b-Instruct",
    "core42/jais-13b-chat",
]

# ---------------------------------------------------------------------------
# Lazy imports for heavy dependencies (only needed at extraction time)
# ---------------------------------------------------------------------------


def _get_engine(model_name: str) -> Any:
    """Build a CulturalRepE engine for the given model.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        A configured :class:`CulturalRepE` instance.
    """
    # Add the Space's root to sys.path so ``src`` is importable
    space_root = Path(__file__).resolve().parent
    if str(space_root) not in sys.path:
        sys.path.insert(0, str(space_root))

    from src.models.rep_engine import CulturalRepE

    hf_token = os.environ.get("HF_TOKEN")
    engine = CulturalRepE(
        model_name=model_name,
        device="cuda",
        dtype="bfloat16",
        hf_token=hf_token,
    )
    engine.load_model()
    return engine


def _save_to_hf(
    vectors: dict[str, torch.Tensor],
    model_name: str,
    extraction_layers: dict[str, int],
    dataset_name: str,
) -> str:
    """Push extracted vectors to a HF Dataset.

    Args:
        vectors: Concept vectors keyed by concept_id.
        model_name: Model used for extraction.
        extraction_layers: Layer per concept.
        dataset_name: Target HF dataset repo.

    Returns:
        Dataset URL.
    """
    from src.utils.hf_integration import save_vectors_to_hf

    metadata = {
        "model_name": model_name,
        "extraction_layers": extraction_layers,
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return save_vectors_to_hf(vectors, dataset_name=dataset_name, metadata=metadata)


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------


def extract_vectors(
    concept_ids: str,
    model_name: str,
    dataset_name: str,
    progress: gr.Progress | None = None,
) -> str:
    """Extract vectors for the given concepts and push to HF Dataset.

    Args:
        concept_ids: Comma-separated concept identifiers.
        model_name: Hugging Face model identifier.
        dataset_name: Target HF dataset.
        progress: Gradio progress tracker.

    Returns:
        Status message.
    """
    if progress is None:
        progress = gr.Progress()

    if not concept_ids.strip():
        return "❌ Please provide at least one concept ID."

    ids = [c.strip() for c in concept_ids.split(",") if c.strip()]
    if not ids:
        return "❌ No valid concept IDs found."

    progress(0.1, desc="Loading model...")
    try:
        engine = _get_engine(model_name)
    except Exception as exc:
        return f"❌ Failed to load model: {exc}\n\n{traceback.format_exc()}"

    progress(0.3, desc="Extracting vectors...")
    results: dict[str, torch.Tensor] = {}
    for i, concept_id in enumerate(ids):
        progress(0.3 + 0.5 * (i + 1) / len(ids), desc=f"Extracting {concept_id}...")
        try:
            vec = engine.extract_vector(concept_id)
            results[concept_id] = vec
        except Exception as exc:
            return f"❌ Extraction failed for {concept_id}: {exc}"

    progress(0.85, desc="Saving to HF Dataset...")
    try:
        url = _save_to_hf(
            results,
            model_name,
            engine.extraction_layers,
            dataset_name,
        )
    except Exception as exc:
        return f"❌ Failed to push to HF: {exc}\n\nVectors extracted but not uploaded."

    progress(1.0, desc="Done!")
    return (
        f"✅ Extracted {len(results)} vectors from {model_name}\n"
        f"📊 Pushed to: {url}\n"
        f"🏷️ Concepts: {', '.join(results.keys())}"
    )


def preview_results(dataset_name: str) -> str:
    """Load latest results from the HF Dataset and return as JSON.

    Args:
        dataset_name: HF dataset repo to read.

    Returns:
        JSON string of the dataset contents.
    """
    from src.utils.hf_integration import load_vectors_from_hf

    try:
        vectors = load_vectors_from_hf(dataset_name=dataset_name)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)

    preview: dict[str, Any] = {}
    for concept_id, vec in list(vectors.items())[:5]:
        preview[concept_id] = {
            "shape": list(vec.shape),
            "norm": float(torch.linalg.vector_norm(vec).item()),
            "first_5": vec[:5].tolist(),
        }
    return json.dumps(preview, indent=2, ensure_ascii=False)


def download_json(dataset_name: str) -> str:
    """Download all vectors from HF Dataset as a local JSON file.

    Args:
        dataset_name: HF dataset repo.

    Returns:
        Path to the downloaded JSON file.
    """
    from src.utils.hf_integration import load_vectors_from_hf

    vectors = load_vectors_from_hf(dataset_name=dataset_name)
    output: dict[str, Any] = {}
    for concept_id, vec in vectors.items():
        output[concept_id] = {
            "vector": vec.tolist(),
            "shape": list(vec.shape),
            "norm": float(torch.linalg.vector_norm(vec).item()),
        }

    out_path = "/tmp/thaqafa_vectors.json"
    Path(out_path).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    return out_path


def push_to_hf(dataset_name: str) -> str:
    """Trigger a push of any local vectors to HF Dataset.

    Args:
        dataset_name: Target HF dataset.

    Returns:
        Status message.
    """
    # In the Space context, extraction already pushes. This is a
    # convenience button for re-syncing.
    return f"ℹ️ Vectors are pushed automatically after extraction.\nDataset: {dataset_name}"


# ---------------------------------------------------------------------------
# Build the Gradio interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="Thaqafa-RepE Vector Extraction") as demo:
    gr.Markdown(
        """
        # 🧠 Thaqafa-RepE — Zero-GPU Vector Extraction

        Extract cultural concept vectors from LLMs using Representation Engineering.
        Results are pushed to a private Hugging Face Dataset.
        """
    )

    with gr.Tab("Extract Vectors"):
        with gr.Row():
            concept_input = gr.Textbox(
                label="Concept IDs (comma-separated)",
                value=DEFAULT_CONCEPTS,
                info="e.g. wasta_001, muruah_001, diyafa_001",
            )
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=MODEL_CHOICES[0],
                label="Model",
            )
            dataset_input = gr.Textbox(
                label="HF Dataset",
                value=DEFAULT_DATASET,
            )
        extract_btn = gr.Button("🚀 Extract & Push", variant="primary")
        extract_output = gr.Textbox(label="Status", lines=6)

    with gr.Tab("Preview Results"):
        preview_btn = gr.Button("📊 Load Latest Results")
        preview_output = gr.Code(label="Vector Preview (JSON)", language="json")

    with gr.Tab("Download"):
        download_btn = gr.Button("⬇️ Download as JSON")
        download_file = gr.File(label="Downloaded File")
        push_btn = gr.Button("📤 Push to HF Dataset")
        push_output = gr.Textbox(label="Push Status", lines=2)

    # Wire up events
    extract_btn.click(
        extract_vectors,
        inputs=[concept_input, model_dropdown, dataset_input],
        outputs=extract_output,
    )
    preview_btn.click(
        preview_results,
        inputs=[dataset_input],
        outputs=preview_output,
    )
    download_btn.click(
        download_json,
        inputs=[dataset_input],
        outputs=download_file,
    )
    push_btn.click(
        push_to_hf,
        inputs=[dataset_input],
        outputs=push_output,
    )

if __name__ == "__main__":
    demo.launch()
