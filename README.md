# Thaqafa-RepE

**Representation Engineering for Arab Cultural Concepts in Large Language Models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/al3obdi/Thaqafa-RepE/actions/workflows/ci.yml/badge.svg)](https://github.com/al3obdi/Thaqafa-RepE/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> *Thaqafa* (ثقافة) means "culture" in Arabic. This project asks whether that
> culture is legible inside a language model's activations, and whether it can
> be steered.

---

## Research Motivation

Alignment research has largely been written in English, by and for a narrow slice
of the world. When a model is asked to reason about *wasta* (الواسطة), *muru'ah*
(المروءة) or the obligations of *diyafa* (الضيافة), it tends to fall back on
translated approximations that flatten the concept into its nearest Western
analogue — nepotism, chivalry, hospitality — and loses the social weight that
makes the original meaningful.

Representation engineering (RepE) offers a way to study this directly. Instead
of probing behaviour through prompts alone, RepE reads the model's internal
activations, isolates the direction that encodes a concept, and then adds that
direction back into the forward pass to steer generation. Thaqafa-RepE applies
this method to Arab cultural concepts in order to answer three questions:

1. **Legibility** — Are Arab cultural concepts linearly represented in the
   residual stream of open-weight instruction-tuned models?
2. **Locality** — At which layers do these representations emerge, and are
   Arabic and English descriptions of the same concept encoded in a shared
   direction?
3. **Controllability** — Can injecting a concept vector make a model's answers
   more culturally grounded without degrading factuality or fluency?

The long-term goal is a reproducible, auditable toolkit for cultural alignment
that other under-represented cultures can adapt, not a single benchmark score.

## Key Features

- **Bilingual concept dataset** — every concept carries Arabic and English
  names, examples, and a note on the cultural context it belongs to.
- **Concept vector extraction** — read activations at any layer and derive a
  reading vector per concept (`CulturalRepE.extract_vector`).
- **Steered generation** — inject a concept vector at a chosen strength and set
  of layers, with negative strengths for suppression
  (`CulturalRepE.inject_vector`).
- **Evaluation harness** — sweep injection strength and plot the effect on
  alignment metrics.
- **Research-grade tooling** — Poetry-pinned dependencies, pre-commit hooks,
  type hints throughout, and CI running ruff, black and pytest.

> **Status: scaffolding.** The extraction and injection algorithms are typed,
> documented stubs that raise `NotImplementedError`. The data layer, plotting
> utilities and CLI wiring around them are functional.

## Installation

Requires Python 3.10–3.12 and [Poetry](https://python-poetry.org/docs/#installation).

```bash
git clone https://github.com/al3obdi/Thaqafa-RepE.git
cd Thaqafa-RepE

# Install runtime and development dependencies
poetry install

# Install the git hooks (black, isort, ruff, mypy, nbstripout)
poetry run pre-commit install

# Copy the environment template and fill in your own values
cp .env.example .env
```

`.env` is git-ignored. Never commit tokens or credentials — put your Hugging
Face token in `.env` only, and leave `.env.example` empty.

Verify the installation:

```bash
poetry run pytest
poetry run ruff check .
```

## Usage

### Load the concept dataset

```python
from src.data.dataset_builder import load_concepts, filter_by_category

concepts = load_concepts("data/datasets/cultural_concepts.jsonl")
ethical = filter_by_category(concepts, "ethical")

for concept in ethical:
    print(concept.concept_ar, "-", concept.concept_en)
```

### Extract a concept vector

```python
from src.models.rep_engine import CulturalRepE

engine = CulturalRepE(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    device="cuda",
    dtype="bfloat16",
)
engine.load_model()

vector = engine.extract_vector(
    concept="diyafa_001",
    examples=["أكرم ضيافته لمدة ثلاثة أيام", "He hosted him generously for three days"],
    layer=-1,
)
```

### Steer generation

```python
# Amplify the concept
engine.inject_vector(concept="diyafa_001", strength=1.5, layers=[12, 14, 16])

# Suppress it
engine.inject_vector(concept="diyafa_001", strength=-1.5)
```

### Command line

```bash
poetry run python scripts/extract_vectors.py --layer -1 --output outputs/vectors
poetry run python scripts/inject_concepts.py --concept diyafa_001 --strength 1.5 \
    --prompt "What should I do when a guest arrives unannounced?"
poetry run python scripts/evaluate.py --concept diyafa_001 --min -2 --max 2 --steps 9
```

## Project Structure

```text
Thaqafa-RepE/
├── .github/workflows/ci.yml       # Lint and test on every push and PR
├── notebooks/                     # Exploratory research notebooks
│   ├── 01_data_collection.ipynb   # Build and inspect the concept dataset
│   ├── 02_vector_extraction.ipynb # Extract vectors, sweep layers
│   └── 03_concept_injection.ipynb # Steer generation, sweep strengths
├── scripts/                       # Reproducible CLI entry points
│   ├── extract_vectors.py
│   ├── inject_concepts.py
│   └── evaluate.py
├── src/
│   ├── data/dataset_builder.py    # CulturalConcept dataclass and loaders
│   ├── models/rep_engine.py       # CulturalRepE: extraction and injection
│   └── utils/visualization.py     # Layer sweeps, similarity heatmaps
├── data/
│   ├── raw/                       # Untracked source material
│   ├── processed/                 # Untracked intermediate artefacts
│   └── datasets/                  # Tracked, curated concept dataset (JSONL)
├── tests/                         # pytest suite
├── docs/research_paper/           # Paper drafts and figures
├── .env.example                   # Configuration template (no secrets)
├── .pre-commit-config.yaml
├── pyproject.toml                 # Poetry dependencies and tool config
└── CONTRIBUTING.md
```

Only `data/datasets/` is version-controlled. `data/raw/` and `data/processed/`
are ignored so that large or licence-restricted corpora never enter git history.

## Dataset Schema

Each line of `data/datasets/cultural_concepts.jsonl` is one JSON object:

| Field | Type | Description |
| --- | --- | --- |
| `concept_id` | string | Stable identifier, e.g. `wasta_001` |
| `concept_ar` | string | Concept name in Arabic |
| `concept_en` | string | Transliteration and English gloss |
| `category` | string | `social`, `ethical`, `cultural`, ... |
| `description` | string | One-sentence definition |
| `examples_ar` | string[] | Arabic sentences expressing the concept |
| `examples_en` | string[] | English sentences expressing the concept |
| `cultural_context` | string | Where and why the concept matters |
| `sentiment` | string | `positive`, `negative` or `mixed` |

## Roadmap

- [x] **Phase 0 — Scaffolding.** Repository structure, tooling, CI, dataset schema.
- [ ] **Phase 1 — Data collection.** Expand to 100+ concepts with contrastive
      examples, reviewed by native speakers across dialect regions.
- [ ] **Phase 2 — Vector extraction.** Implement activation reading, layer
      sweeps and linear probes; report where concepts become separable.
- [ ] **Phase 3 — Concept injection.** Implement steering hooks and measure the
      strength/coherence trade-off.
- [ ] **Phase 4 — Evaluation.** Human evaluation of cultural grounding, plus
      automatic checks for factuality and fluency regressions.
- [ ] **Phase 5 — Publication.** Release the dataset, vectors and paper.

## Citation

A paper is in preparation. Until then, please cite the repository:

```bibtex
@software{thaqafa_repe_2026,
  title  = {Thaqafa-RepE: Representation Engineering for Arab Cultural Concepts},
  author = {{Thaqafa-RepE Contributors}},
  year   = {2026},
  url    = {https://github.com/al3obdi/Thaqafa-RepE}
}
```

## Contributing

Contributions are welcome — especially new cultural concepts and native-speaker
review of existing entries. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Released under the [MIT License](LICENSE).
