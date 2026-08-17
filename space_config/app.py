"""Thaqafa-RepE Zero-GPU Vector Extraction Space.

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
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_SERVER_PORT"] = "7860"

import gradio as gr
import spaces  # HF ZeroGPU SDK
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
# Helpers
# ---------------------------------------------------------------------------


def _resolve_token(token: str | None = None) -> str:
    """Resolve HF token from argument or environment.

    Args:
        token: Optional token. Falls back to ``HF_TOKEN`` env var.

    Returns:
        The resolved token string.

    Raises:
        ValueError: If no token is available.
    """
    t = token or os.environ.get("HF_TOKEN")
    if not t:
        raise ValueError("No HF_TOKEN set. Add it as a Space secret.")
    return t


def _tensor_to_list(tensor: torch.Tensor) -> list[float]:
    """Convert a tensor to a plain list of floats."""
    return tensor.detach().cpu().to(torch.float32).tolist()


def _list_to_tensor(values: list[float]) -> torch.Tensor:
    """Convert a list of floats back to a tensor."""
    return torch.tensor(values, dtype=torch.float32)


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
    from datasets import Dataset

    token = _resolve_token()
    rows = []
    ts = datetime.now(timezone.utc).isoformat()
    for cid, vec in vectors.items():
        rows.append(
            {
                "concept_id": cid,
                "concept_ar": "",
                "concept_en": "",
                "vector": _tensor_to_list(vec),
                "extraction_layer": extraction_layers.get(cid, -1),
                "model_name": model_name,
                "extraction_timestamp": ts,
            }
        )
    ds = Dataset.from_list(rows)
    ds.push_to_hub(dataset_name, token=token, private=True)
    return f"https://huggingface.co/datasets/{dataset_name}"


def _load_from_hf(dataset_name: str) -> dict[str, torch.Tensor]:
    """Load vectors from a HF Dataset.

    Args:
        dataset_name: HF dataset repo to read.

    Returns:
        Dict mapping concept_id to tensor.
    """
    from datasets import load_dataset

    token = _resolve_token()
    ds = load_dataset(dataset_name, token=token)
    split = list(ds.keys())[0] if hasattr(ds, "keys") else "train"
    rows = [dict(r) for r in ds[split]]
    return {r["concept_id"]: _list_to_tensor(r["vector"]) for r in rows}


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------


@spaces.GPU
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
        from transformer_lens import HookedTransformer

        hf_token = os.environ.get("HF_TOKEN")
        kwargs = {"token": hf_token} if hf_token else {}
        model = HookedTransformer.from_pretrained(
            model_name,
            device="cuda",
            dtype=torch.bfloat16,
            **kwargs,
        )
        model.eval()
    except Exception as exc:
        return f"❌ Model load failed: {exc}\n{traceback.format_exc()}"

    n_layers = int(model.cfg.n_layers)
    layer = n_layers // 2
    hook = f"blocks.{layer}.hook_resid_post"

    progress(0.3, desc="Extracting vectors...")
    results: dict[str, torch.Tensor] = {}
    layers_out: dict[str, int] = {}
    for i, cid in enumerate(ids):
        progress(0.3 + 0.5 * (i + 1) / len(ids), desc=f"Extracting {cid}...")
        try:
            with torch.no_grad():
                tokens = model.to_tokens([f"Concept: {cid}"])
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=hook,
                    stop_at_layer=layer + 1,
                    return_type=None,
                )
                vec = cache[hook][0].mean(dim=0).to(torch.float32).cpu()
                vec = vec / vec.norm()
            results[cid] = vec
            layers_out[cid] = layer
        except Exception as exc:
            return f"❌ Extraction failed for {cid}: {exc}"

    progress(0.85, desc="Saving to HF Dataset...")
    try:
        url = _save_to_hf(results, model_name, layers_out, dataset_name)
    except Exception as exc:
        return f"❌ Push failed: {exc}\n\nVectors extracted but not uploaded."

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
    try:
        vectors = _load_from_hf(dataset_name)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)

    preview: dict[str, Any] = {}
    for cid, vec in list(vectors.items())[:5]:
        preview[cid] = {
            "shape": list(vec.shape),
            "norm": float(vec.norm().item()),
        }
    return json.dumps(preview, indent=2, ensure_ascii=False)


def download_json(dataset_name: str) -> str:
    """Download all vectors from HF Dataset as a local JSON file.

    Args:
        dataset_name: HF dataset repo.

    Returns:
        Path to the downloaded JSON file.
    """
    vectors = _load_from_hf(dataset_name)
    output: dict[str, Any] = {}
    for cid, vec in vectors.items():
        output[cid] = {
            "vector": vec.tolist(),
            "shape": list(vec.shape),
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
    return f"ℹ️ Vectors are pushed automatically after extraction.\nDataset: {dataset_name}"


# ---------------------------------------------------------------------------
# Build the Gradio interface
# ---------------------------------------------------------------------------

demo = gr.Blocks(title="Thaqafa-RepE Vector Extraction")

with demo:
    gr.Markdown("""
        # 🧠 Thaqafa-RepE — Zero-GPU Vector Extraction

        Extract cultural concept vectors from LLMs using Representation Engineering.
        Results are pushed to a private Hugging Face Dataset.
        """)

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

demo.queue(default_concurrency_limit=1)
demo.launch()

# Keepalive loop — prevents the process from exiting on HF Spaces infrastructure
while True:
    time.sleep(3600)
