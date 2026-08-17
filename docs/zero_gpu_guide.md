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

## Automated Extraction from the Command Line

The `scripts/run_space_extraction.py` script automates the entire flow:
connect to the Space, submit an extraction job, wait for completion,
and pull the results back locally.

### Prerequisites

```bash
# Set your HF token
export HF_TOKEN=hf_your_token_here

# Install gradio_client (included in project dependencies)
pip install gradio_client
```

### Usage

```bash
# Extract specific concepts
python scripts/run_space_extraction.py --concepts wasta_001,diyafa_001

# Use a different model
python scripts/run_space_extraction.py --concepts muruah_001 \
  --model allam-ai/ALLaM-1-7b-Instruct

# Custom dataset and space
python scripts/run_space_extraction.py --concepts wasta_001 \
  --dataset al3obdi/thaqafa-repe-vectors \
  --space al3obdi/thaqafa-repe-extraction
```

### What the Script Does

1. Resolves `HF_TOKEN` from the environment
2. Connects to the ZeroGPU Space via `gradio_client.Client`
3. Submits an extraction job (concept IDs + model name)
4. Polls every 5 seconds until the job completes (max 10 minutes)
5. Loads the extracted vectors from the HF Dataset
6. Prints a summary of loaded vectors (concept ID, shape, norm)

### Using `extract_via_space()` in Python

For programmatic access, use the `CulturalRepE.extract_via_space()` method:

```python
from src.models.rep_engine import CulturalRepE

engine = CulturalRepE(model_name="meta-llama/Meta-Llama-3-8B-Instruct")

# Trigger extraction on the Space and load results into engine.concept_vectors
vectors = engine.extract_via_space(
    concept_ids=["wasta_001", "diyafa_001", "muruah_001"],
)

# Now use the vectors for steering
with engine.steering("diyafa_001", strength=2.0):
    output = engine.model.generate("A guest arrives at your home", max_new_tokens=50)
```

### Using in a Jupyter Notebook

```python
import os
os.environ["HF_TOKEN"] = "hf_your_token_here"

from src.models.rep_engine import CulturalRepE

engine = CulturalRepE()
vectors = engine.extract_via_space(concept_ids=["wasta_001"])
print(f"Loaded {len(vectors)} vectors")
for cid, vec in vectors.items():
    print(f"  {cid}: shape={vec.shape}, norm={vec.norm():.4f}")
```

## Generating Paper Results

The `scripts/generate_paper_results.py` script orchestrates the full experimental pipeline:

### Prerequisites

```bash
export HF_TOKEN=hf_your_token_here
pip install gradio_client matplotlib
```

### Usage

```bash
# Extract all 3 seed concepts and generate all outputs
python scripts/generate_paper_results.py --concepts wasta_001,muruah_001,diyafa_001

# Use a specific model
python scripts/generate_paper_results.py --concepts wasta_001 \
  --model meta-llama/Meta-Llama-3-8B-Instruct

# Custom output directory
python scripts/generate_paper_results.py --concepts wasta_001 \
  --output-dir outputs/paper_results
```

### What the Script Does

1. **Extraction**: Triggers vector extraction on the ZeroGPU Space via `extract_via_space()`
2. **Layer sweep**: Runs linear probes across all layers, identifies best layer per concept
3. **Steering sweep**: Measures effect (KL) vs cost (fluency loss) across strength grid
4. **Baseline comparison**: Compares steering against prompt-based baselines
5. **Report**: Generates `RESULTS_SUMMARY.md` with LaTeX-ready snippets

### Output Files

- `outputs/paper_results/vectors.json` - Extracted vectors
- `outputs/paper_results/layer_sweep.csv` - Probe accuracy per layer
- `outputs/paper_results/steering_sweep.csv` - KL and loss per strength
- `outputs/paper_results/baseline_comparison.csv` - Steering vs prompting
- `outputs/paper_results/figures/*.png` - Layer sweep and steering plots
- `outputs/paper_results/RESULTS_SUMMARY.md` - Markdown report with LaTeX snippets

### Workflow: Results to Paper

1. Run the script: `python scripts/generate_paper_results.py --concepts wasta_001,muruah_001,diyafa_001`
2. Open `outputs/paper_results/RESULTS_SUMMARY.md`
3. Search for `\todo` markers in `docs/research_paper/main.tex`
4. Copy the LaTeX-ready tables from the Markdown into the corresponding `\todo` locations
5. Replace placeholder figures with `figures/layer_sweep.png` and `figures/effect_vs_cost.png`
6. Rebuild: `cd docs/research_paper && ./build.sh`

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