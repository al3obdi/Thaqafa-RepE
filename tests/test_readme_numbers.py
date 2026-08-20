"""The README's summary table must agree with the committed artefacts.

The table is written by hand, which is how it fell out of step once already:
a seed fix changed the gpt2 layer sweep and the README kept quoting the old
figures. Numbers a reader meets first are the ones most worth pinning, so each
cell is recomputed here from the CSVs under ``results/`` and compared.

If a rerun legitimately changes a number, this test fails and the README is
updated in the same commit as the artefact - which is the point.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = PROJECT_ROOT / "README.md"
RESULTS = PROJECT_ROOT / "results"

MIN_READABLE_PROBE = 0.70
"""Probe quality below which a causal point is not counted, per the README."""


def _rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of row dicts."""
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _best_layer_rows(run: Path) -> list[dict[str, str]]:
    """Return each concept's highest-scoring layer from the sweep."""
    best: dict[str, dict[str, str]] = {}
    for row in _rows(run / "layer_sweep.csv"):
        concept = row["concept_id"]
        if concept not in best or float(row["probe_score"]) > float(best[concept]["probe_score"]):
            best[concept] = row
    return list(best.values())


def _causal_counts(path: Path, effect_key: str, probe_key: str) -> tuple[int, int]:
    """Count points through a usable probe, and how many show the effect."""
    readable = [row for row in _rows(path) if float(row[probe_key]) >= MIN_READABLE_PROBE]
    return sum(1 for row in readable if float(row[effect_key]) > 0), len(readable)


def _readme_row(label: str) -> list[str]:
    """Return the README summary-table cells for one run.

    Args:
        label: Text identifying the run's row, e.g. ``"pilot_gpt2"``.

    Returns:
        The row's cells, stripped.

    Raises:
        AssertionError: If the row is not in the table.
    """
    for line in README.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and f"results/{label}/" in line:
            return [cell.strip() for cell in line.strip("|").split("|")]
    raise AssertionError(f"No README summary row for {label}")


RUNS = [directory.name for directory in sorted(RESULTS.iterdir()) if directory.is_dir()]


@pytest.mark.parametrize("run", RUNS)
class TestReadmeMatchesArtefacts:
    """Every cell in the summary table, against the CSV behind it."""

    def test_significant_concept_count(self, run: str) -> None:
        """The headline "readable above chance" fraction."""
        best = _best_layer_rows(RESULTS / run)
        significant = sum(1 for row in best if float(row["p_value"]) < 0.05)

        assert _readme_row(run)[1] == f"{significant} / {len(best)}"

    def test_mean_balanced_accuracy(self, run: str) -> None:
        """Quoted to three decimals, so it must round to the same string."""
        best = _best_layer_rows(RESULTS / run)
        mean = sum(float(row["probe_score"]) for row in best) / len(best)

        assert _readme_row(run)[2] == f"{mean:.3f}"

    def test_amplification_counts(self, run: str) -> None:
        """Points where adding the direction beat a matched-norm random one."""
        hits, total = _causal_counts(
            RESULTS / run / "causal_readback.csv", "lift_over_random", "probe_accuracy"
        )

        assert _readme_row(run)[3] == f"{hits} / {total}"

    def test_suppression_counts(self, run: str) -> None:
        """Points where subtracting it beat a matched-norm random one."""
        hits, total = _causal_counts(
            RESULTS / run / "suppression.csv",
            "drop_beyond_random",
            "probe_balanced_accuracy",
        )

        assert _readme_row(run)[4] == f"{hits} / {total}"


def test_every_committed_run_appears_in_the_readme() -> None:
    """A run nobody links to is a run nobody reads."""
    linked = {run for run in RUNS if f"results/{run}/" in README.read_text(encoding="utf-8")}
    assert linked == set(RUNS)


def test_the_readme_states_the_probe_threshold_it_counts_by() -> None:
    """The causal fractions are meaningless without it."""
    text = README.read_text(encoding="utf-8")
    assert re.search(r"0\.70 balanced accuracy", text)
