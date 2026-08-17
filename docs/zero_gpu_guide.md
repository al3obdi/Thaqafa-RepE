# Zero-GPU Strategy Guide

This guide explains how Thaqafa-RepE uses a hybrid local/cloud architecture
to run GPU-heavy experiments without owning a GPU.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LOCAL (CPU)                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Development  │  │ Unit Tests   │  │ Evaluation   │              │
│  │ & Debugging  │  │ (CPU-only)   │  │ (Lightweight)│              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                    sync_with_space()
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                  HF ZERO-GPU SPACE (A10G)                            │
│  ┌──────────────────────────────────────────────────┐               │
│  │  Gradio Interface (app.py)                        │               │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐          │               │
│  │  │ Extract │  │ Preview │  │Download │          │               │
│  │  │ Vectors │  │ Results │  │   JSON  │          │               │
│  │  └─────────┘  └─────────┘  └─────────┘          │               │
│  └──────────────────────────────────────────────────┘               │
│  ┌──────────────────────────────────────────────────┐               │
│  │  CulturalRepE Engine (transformer_lens)           │               │
│  │  Llama-3-8B-Instruct / Jais-13B                   │               │
│  └──────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                    push_to_hub()
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                  HF PRIVATE DATASET                                  │
│  ┌──────────────────────────────────────────────────┐               │
│  │  al3obdi/thaqafa-repe-vectors                     │               │
│  │  ┌────────┬────────┬────────┬────────┐           │               │
│  │  │concept │ vector │ layer  │ model  │           │               │
│  │  │  _id   │ (list) │  (int) │ (str)  │           │               │
│  │  └────────┴────────┴────────┴────────┘           │               │
│  └──────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

## Step-by-Step: Extract Vectors via the Space

### 1. Access the Space

Go to: https://al3obdi-thaqafa-repe-extraction.hf.space

> **Note**: The Space requires ZeroGPU hardware. If you see a "payment required"
> error, the Space may be on `cpu-basic` — change it to `zero-a10g` in
> Settings → Hardware (requires HF PRO).

### 2. Extract Vectors

1. Open the **"Extract Vectors"** tab
2. Enter concept IDs (comma-separated), e.g.: `wasta_001, muruah_001, diyafa_001`
3. Select a model from the dropdown (Llama-3-8B-Instruct is default)
4. Click **"🚀 Extract & Push"**
5. Wait for the progress bar to complete — vectors are automatically pushed to the HF Dataset

### 3. Preview Results

1. Switch to the **"Preview Results"** tab
2. Click **"📊 Load Latest Results"**
3. Inspect the JSON preview showing vector shapes, norms, and first 5 values

### 4. Download Vectors

1. Switch to the **"Download"** tab
2. Click **"⬇️ Download as JSON"**
3. The file `thaqafa_vectors.json` will download

## Loading Vectors Locally

After extraction via the Space, load the vectors on your local machine:

```python
import os
os.environ["HF_TOKEN"] = "your_token_here"

from src.utils.hf_integration import load_vectors_from_hf

# Load all vectors
vectors = load_vectors_from_hf("al3obdi/thaqafa-repe-vectors")

# Load specific concepts only
vectors = load_vectors_from_hf(
    "al3obdi/thaqafa-repe-vectors",
    concept_ids=["diyafa_001", "wasta_001"],
)

# Use them with the engine
from src.models.rep_engine import CulturalRepE

engine = CulturalRepE(model_name="meta-llama/Meta-Llama-3-8B-Instruct")
engine.concept_vectors = vectors  # inject loaded vectors
```

Or use the engine's built-in method:

```python
engine = CulturalRepE(model_name="meta-llama/Meta-Llama-3-8B-Instruct")
engine.load_vectors_from_hf()  # loads into engine.concept_vectors
```

## Checking Space Status

```python
from src.utils.hf_integration import sync_with_space

status = sync_with_space()
print(f"Stage: {status['stage']}")
print(f"Hardware: {status['hardware']}")
print(f"Space URL: {status['space_url']}")
```

## Saving Vectors to HF (from local)

If you extract vectors locally (e.g., on a machine with a GPU):

```python
from src.utils.hf_integration import save_vectors_to_hf

url = save_vectors_to_hf(
    vectors=engine.concept_vectors,
    dataset_name="al3obdi/thaqafa-repe-vectors",
    metadata={
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "extraction_layers": engine.extraction_layers,
    },
)
print(f"Pushed to: {url}")
```

## Troubleshooting

### "Cannot POST" error when creating the Space

This means the HF API endpoint has changed. Use the `huggingface_hub` Python
library instead:

```python
from huggingface_hub import HfApi
api = HfApi(token="your_token")
api.create_repo("al3obdi/thaqafa-repe-extraction", repo_type="space", space_sdk="gradio")
```

### ZeroGPU "Payment Required" (402)

You have reached the 10-Space ZeroGPU limit. Either:
- Delete an unused ZeroGPU Space, or
- Set the Space to `cpu-basic` temporarily (no GPU, but the UI still works)

### "No Hugging Face token found"

Set the `HF_TOKEN` environment variable:

```bash
export HF_TOKEN=hf_your_token_here
```

Or pass it explicitly:

```python
save_vectors_to_hf(vectors, token="hf_your_token_here")
```

### Dataset is empty after extraction

Check that:
1. The Space's `HF_TOKEN` secret has write access to the dataset
2. The dataset name in the Space UI matches `al3obdi/thaqafa-repe-vectors`
3. The extraction actually completed (check the status output in the UI)

### Vectors have wrong shape after loading

The dataset stores vectors as lists of floats. They are always loaded as
1-D `float32` tensors. If you need a different dtype, cast after loading:

```python
vectors = load_vectors_from_hf()
for k, v in vectors.items():
    vectors[k] = v.to(torch.bfloat16)
```

## Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `concept_id` | string | Stable identifier (e.g. `wasta_001`) |
| `concept_ar` | string | Arabic concept name |
| `concept_en` | string | English concept name |
| `vector` | list[float] | The concept direction |
| `extraction_layer` | int | Layer the vector was extracted from |
| `model_name` | string | Model used for extraction |
| `extraction_timestamp` | string | ISO-8601 timestamp |