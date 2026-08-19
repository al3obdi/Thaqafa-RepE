# Contributing to Thaqafa-RepE

Thank you for your interest in this project. Contributions of code, data and
cultural review are all welcome. This document explains how to get set up and
what we expect from a contribution.

All code, comments, docstrings and commit messages must be written in English.
Dataset content is bilingual by design: Arabic fields hold Arabic text, English
fields hold English text.

## Getting Set Up

```bash
git clone https://github.com/al3obdi/Thaqafa-RepE.git
cd Thaqafa-RepE
poetry install
poetry run pre-commit install
cp .env.example .env
```

Run the checks before you push:

```bash
poetry run pytest
poetry run ruff check .
poetry run black --check .
poetry run mypy
```

## Reporting Bugs

Open an issue with:

1. **What happened** and **what you expected** instead.
2. A **minimal reproduction** — the smallest script or notebook cell that shows
   the problem.
3. Your **environment**: OS, Python version, `poetry show torch transformers`
   output, and GPU model if relevant.
4. The **full traceback** in a fenced code block.

Search existing issues first; a comment on an open issue is more useful than a
duplicate.

## Suggesting Features

Open an issue describing the research question the feature would help answer,
not just the implementation. Include:

- The problem or gap you are trying to close.
- How you would know the feature worked (a metric, a plot, an experiment).
- Any prior work in the interpretability or cultural alignment literature.

## Adding New Cultural Concepts

This is the most valuable contribution, and the one that most needs native
speaker judgement.

Dataset contributions are licensed under CC BY-SA 4.0 (see
`data/datasets/README.md`); by submitting concepts you agree to that.

1. Append one line per concept to `data/datasets/cultural_concepts.jsonl` —
   one JSON object per line, no trailing commas, UTF-8 encoded.
2. Use the full schema; every field in the README's schema table is required.
3. Choose a `concept_id` of the form `<transliteration>_<3 digits>`, for example
   `karam_001`. Identifiers must be unique.
4. Provide at least three `examples_ar` and three `examples_en`. Examples should
   be natural sentences, not dictionary definitions.
5. Write `cultural_context` in a way that a reader outside the culture can
   follow: where the concept is used, who it binds, and what it costs to
   violate it.
6. Set `sentiment` honestly. Contested concepts should be `mixed` rather than
   flattered into `positive`.
7. Note the dialect or region in `cultural_context` when a concept is not
   pan-Arab.

Validate your addition before opening a pull request:

```bash
poetry run python -c "from src.data.dataset_builder import load_concepts; print(len(load_concepts('data/datasets/cultural_concepts.jsonl')))"
```

Concepts that touch religion, politics or gender norms are in scope, but they
must be described descriptively rather than prescriptively, and will be
reviewed by more than one native speaker before merging.

## Pull Request Process

1. Branch from `main` using a descriptive name: `feature/...`, `fix/...`,
   `data/...` or `docs/...`.
2. Keep each pull request focused on one change. Data additions and code
   changes belong in separate pull requests.
3. Write commit messages in the [Conventional Commits](https://www.conventionalcommits.org/)
   style: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`.
4. Add or update tests for any behaviour you change.
5. Make sure CI passes — ruff, black and pytest all run on every pull request.
6. Fill in the pull request description: what changed, why, and how you
   verified it.
7. Never commit secrets, tokens, model weights or large raw corpora. `.env`,
   `checkpoints/`, `outputs/` and `data/raw/` are git-ignored for this reason.

## Code Style

- **Formatting**: `black` with a line length of 100. `isort` with the black
  profile for import ordering.
- **Linting**: `ruff`, configured in `pyproject.toml`.
- **Typing**: `mypy` with `disallow_untyped_defs`. Every public function needs
  full type hints.
- **Docstrings**: Google style, with `Args:`, `Returns:` and `Raises:` sections
  on anything non-trivial.
- **Notebooks**: outputs are stripped automatically by `nbstripout` on commit.
  Keep exploratory work in `notebooks/` and promote anything reusable into
  `src/`.

The pre-commit hooks enforce most of this for you. If a hook rewrites a file,
stage the result and commit again.

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). The short
version: be respectful, assume good faith, discuss cultural concepts
descriptively rather than prescriptively, and treat native-speaker review
as authoritative on cultural content.
