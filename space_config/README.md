---
title: Thaqafa-RepE Vector Extraction
emoji: "🧠"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.20.0"
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
tags:
  - representation-engineering
  - arabic
  - cultural-alignment
  - interpretability
---

# Thaqafa-RepE — Zero-GPU Vector Extraction Space

This Hugging Face Space provides a web interface for extracting cultural concept
vectors from large language models using Representation Engineering (RepE).

## Architecture

This Space is part of a **hybrid Zero-GPU strategy**:

- **Development & Testing** — Local (CPU): Code development, unit tests, lightweight evaluation.
- **Vector Extraction** — **This Space** (ZeroGPU A10G): Heavy GPU extraction from Llama-3-8B or Jais-13B.
- **Vector Storage** — [HF Dataset](https://huggingface.co/datasets/al3obdi/thaqafa-repe-vectors): Private dataset for extracted vectors.

## How It Works

1. **Extract Vectors Tab**: Select concepts and a model, then click "Extract & Push".
   The Space loads the model on ZeroGPU, extracts contrastive mean-difference vectors,
   and pushes results to the private HF Dataset.

2. **Preview Results Tab**: Load the latest vectors from the dataset and inspect
   their shapes and norms.

3. **Download Tab**: Download all vectors as a JSON file for local use.

## Integration with Main Repository

The main repository ([Thaqafa-RepE](https://github.com/al3obdi/Thaqafa-RepE))
provides the ``CulturalRepE`` engine and ``hf_integration`` utilities.

After extraction, load vectors locally:

```python
from src.utils.hf_integration import load_vectors_from_hf

vectors = load_vectors_from_hf("al3obdi/thaqafa-repe-vectors")
```

## Setup

1. Set ``HF_TOKEN`` as a Space secret (Settings → Repository secrets)
2. Set hardware to ZeroGPU (``zero-a10g``) in Settings → Hardware
3. The Space will build automatically

## Notes

- Uses Gradio SDK 6.20.0 with Python 3.12 for ZeroGPU compatibility.
- The ``@spaces.GPU`` decorator ensures dynamic GPU allocation.
- A keepalive loop prevents the process from exiting on the HF Spaces infrastructure.

## License

MIT