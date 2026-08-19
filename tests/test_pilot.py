"""Tests for the reproducible pilot runner.

The runner's job is to turn a model into a directory of results that a reader
can check. These tests are therefore about the artefacts: that every file is
written, that the numbers in the report are the numbers in the CSVs, and that
the manifest says what produced them.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_pilot  # noqa: E402
from src.models.rep_engine import RESID_POST_HOOK, CulturalRepE  # noqa: E402
from tests.helpers import DATASET_PATH, SteerableFakeModel  # noqa: E402


class PilotFakeModel(SteerableFakeModel):
    """A fake that can be probed, steered and generated from at once.

    :class:`SteerableFakeModel` returns the same activation for every prompt,
    which would leave the contrastive difference at exactly zero. Here one
    dimension carries the prompt's token content, so probes see variance and
    extraction produces a non-degenerate vector, while the remaining dimensions
    still carry the depth-dependent norm relative steering calibrates against.
    """

    def run_with_cache(
        self,
        tokens: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[None, dict[str, torch.Tensor]]:
        """Return depth-scaled activations whose first dimension varies by text."""
        _, cache = super().run_with_cache(tokens, **kwargs)
        name = next(iter(cache))
        activations = cache[name].clone()
        activations[..., 0] = activations[..., 0] + tokens.to(torch.float32) * 0.1
        return None, {name: activations}


def make_pilot_engine() -> CulturalRepE:
    """Build an engine backed by :class:`PilotFakeModel`."""
    engine = CulturalRepE(
        model_name="dummy/pilot",
        device="cpu",
        dtype="float32",
        dataset_path=DATASET_PATH,
    )
    model = PilotFakeModel(n_layers=3, d_model=4, d_vocab=8)
    engine.model = model  # type: ignore[assignment]
    engine.tokenizer = model.tokenizer
    return engine


@pytest.fixture
def engine() -> CulturalRepE:
    """An engine wired to the pilot fake."""
    return make_pilot_engine()


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of row dicts."""
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestLoadConceptNames:
    """Display names come from the dataset, not a hand-maintained table."""

    def test_covers_every_concept_in_the_dataset(self) -> None:
        """Every entry gets a name, so no run can label one by its raw id."""
        names = run_pilot.load_concept_names(DATASET_PATH)
        line_count = sum(
            1 for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        assert len(names) == line_count

    def test_maps_ids_to_english_names(self) -> None:
        """The value is the readable name an instruction template can use."""
        names = run_pilot.load_concept_names(DATASET_PATH)
        assert names["wasta_001"] == "Wasta/Nepotism"

    def test_empty_dataset_is_an_error(self, tmp_path: Path) -> None:
        """Silently running over zero concepts would produce empty results."""
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="No concepts"):
            run_pilot.load_concept_names(empty)


class TestParseArgs:
    """The defaults are part of what makes a run reproducible."""

    def test_defaults_are_norm_relative(self) -> None:
        """A fixed absolute grid is not comparable across layers."""
        args = run_pilot.parse_args([])
        assert args.model == run_pilot.DEFAULT_MODEL
        assert [float(s) for s in args.strengths.split(",")] == list(run_pilot.DEFAULT_STRENGTHS)

    def test_default_strengths_include_the_unsteered_reference(self) -> None:
        """Zero has to be in the grid, measured through the same code path."""
        assert 0.0 in run_pilot.DEFAULT_STRENGTHS

    def test_overrides_are_honoured(self) -> None:
        """Every knob the manifest records must be settable."""
        args = run_pilot.parse_args(
            ["--model", "gpt2-medium", "--seed", "7", "--strengths", "0.0,0.5", "--no-baselines"]
        )
        assert args.model == "gpt2-medium"
        assert args.seed == 7
        assert args.strengths == "0.0,0.5"
        assert args.no_baselines is True


class TestWriteCsv:
    """Results are read by other tools, so the header contract matters."""

    def test_writes_a_header_and_rows(self, tmp_path: Path) -> None:
        """A reader must be able to open the file with DictReader."""
        path = tmp_path / "nested" / "out.csv"
        run_pilot._write_csv(path, ["a", "b"], [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        assert _read_csv(path) == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        """A run must not fail on its last step because a directory is absent."""
        path = tmp_path / "deep" / "deeper" / "out.csv"
        run_pilot._write_csv(path, ["a"], [{"a": 1}])
        assert path.exists()


class TestLayerSweep:
    """The sweep decides which layer everything downstream uses."""

    def test_covers_every_layer_of_every_concept(self, engine: CulturalRepE) -> None:
        """A partial sweep would make the "best" layer an artefact of coverage."""
        rows, best = run_pilot.run_layer_sweep(engine, ["wasta_001", "diyafa_001"])
        n_layers = engine.model.cfg.n_layers  # type: ignore[union-attr]
        assert len(rows) == 2 * n_layers
        assert set(best) == {"wasta_001", "diyafa_001"}

    def test_reports_dispersion_alongside_accuracy(self, engine: CulturalRepE) -> None:
        """An accuracy without its spread invites over-reading a small sample."""
        rows, _ = run_pilot.run_layer_sweep(engine, ["wasta_001"])
        assert all("probe_std" in row for row in rows)
        assert all("chance_accuracy" in row for row in rows)

    def test_best_layer_is_one_that_was_probed(self, engine: CulturalRepE) -> None:
        """The chosen layer has to appear in the rows backing it."""
        rows, best = run_pilot.run_layer_sweep(engine, ["wasta_001"])
        probed = {row["layer"] for row in rows}
        assert best["wasta_001"] in probed


class TestSteeringSweep:
    """Strengths are fractions of the residual norm, not raw coefficients."""

    def test_one_row_per_concept_and_strength(self, engine: CulturalRepE) -> None:
        """Missing points would silently shorten the curve."""
        engine.extract_vector("wasta_001", layer=1)
        rows = run_pilot.run_steering_sweep(
            engine, ["wasta_001"], (-0.2, 0.0, 0.2), ["A guest arrives."]
        )
        assert [row["relative_strength"] for row in rows] == [-0.2, 0.0, 0.2]

    def test_zero_strength_has_no_effect(self, engine: CulturalRepE) -> None:
        """Injecting a zero offset is a no-op, so it is the honest reference."""
        engine.extract_vector("wasta_001", layer=1)
        rows = run_pilot.run_steering_sweep(engine, ["wasta_001"], (0.0,), ["A guest arrives."])
        assert rows[0]["effect_kl"] == pytest.approx(0.0, abs=1e-6)

    def test_records_the_layer_it_injected_into(self, engine: CulturalRepE) -> None:
        """A KL value is meaningless without knowing where it was applied."""
        engine.extract_vector("wasta_001", layer=2)
        rows = run_pilot.run_steering_sweep(engine, ["wasta_001"], (0.2,), ["A guest arrives."])
        assert rows[0]["layer"] == 2


class TestBaselines:
    """Steering is only interesting next to what prompting already does."""

    def test_includes_the_unprompted_control(self, engine: CulturalRepE, tmp_path: Path) -> None:
        """Without a control, neither arm has anything to be compared against."""
        engine.extract_vector("wasta_001", layer=1)
        rows = run_pilot.run_baselines(
            engine, ["wasta_001"], {"wasta_001": "Wasta"}, ["A guest arrives."], 0.2, 4, tmp_path
        )
        conditions = {row["condition"] for row in rows}
        assert any("neutral" in str(name) for name in conditions)
        assert len(conditions) > 1

    def test_saves_generations_for_blind_rating(self, engine: CulturalRepE, tmp_path: Path) -> None:
        """Fluency cannot judge cultural grounding; humans need the text."""
        engine.extract_vector("wasta_001", layer=1)
        run_pilot.run_baselines(
            engine, ["wasta_001"], {"wasta_001": "Wasta"}, ["A guest arrives."], 0.2, 4, tmp_path
        )
        payload = json.loads((tmp_path / "generations" / "wasta_001.json").read_text())
        assert payload["relative_strength"] == 0.2
        assert payload["conditions"]


class TestWriteReport:
    """The report is what a reader sees first, so it must not overstate."""

    def _manifest(self) -> dict[str, Any]:
        """A manifest with known values to assert against."""
        return {
            "model": {"name": "gpt2", "device": "cpu", "dtype": "float32"},
            "git": {"commit": "abcdef0123456789", "branch": "main", "dirty": False},
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "seed": 42,
            "dataset": {"sha256": "0" * 64},
        }

    def _rows(self) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
        """Minimal layer, best-layer and steering data."""
        layer_rows = [
            {
                "concept_id": "wasta_001",
                "layer": 0,
                "probe_accuracy": 0.5,
                "probe_std": 0.1,
                "chance_accuracy": 0.5,
                "lift_over_chance": 0.0,
                "n_samples": 12,
            },
            {
                "concept_id": "wasta_001",
                "layer": 1,
                "probe_accuracy": 0.75,
                "probe_std": 0.2,
                "chance_accuracy": 0.5,
                "lift_over_chance": 0.25,
                "n_samples": 12,
            },
        ]
        steering_rows = [
            {
                "concept_id": "wasta_001",
                "relative_strength": 0.2,
                "layer": 1,
                "effect_kl": 0.4321,
                "mean_loss": 5.5,
                "perplexity": 244.7,
            }
        ]
        return layer_rows, {"wasta_001": 1}, steering_rows

    def test_states_the_model_commit_and_seed(self, tmp_path: Path) -> None:
        """A table of numbers with no provenance invites being quoted alone."""
        layer_rows, best, steering_rows = self._rows()
        path = run_pilot.write_report(
            tmp_path, self._manifest(), layer_rows, best, steering_rows, []
        )
        text = path.read_text(encoding="utf-8")
        assert "gpt2" in text
        assert "abcdef012345" in text
        assert "42" in text

    def test_reports_the_best_layer_row(self, tmp_path: Path) -> None:
        """The headline accuracy must be the one at the chosen layer."""
        layer_rows, best, steering_rows = self._rows()
        path = run_pilot.write_report(
            tmp_path, self._manifest(), layer_rows, best, steering_rows, []
        )
        text = path.read_text(encoding="utf-8")
        assert "0.750" in text
        assert "+0.250" in text

    def test_carries_the_limitations(self, tmp_path: Path) -> None:
        """Small samples and unreviewed entries have to travel with the numbers."""
        layer_rows, best, steering_rows = self._rows()
        path = run_pilot.write_report(
            tmp_path, self._manifest(), layer_rows, best, steering_rows, []
        )
        text = path.read_text(encoding="utf-8")
        assert "Limitations" in text
        assert "native-speaker" in text

    def test_flags_a_dirty_working_tree(self, tmp_path: Path) -> None:
        """Results from uncommitted code must say so on their face."""
        manifest = self._manifest()
        manifest["git"]["dirty"] = True
        layer_rows, best, steering_rows = self._rows()
        path = run_pilot.write_report(tmp_path, manifest, layer_rows, best, steering_rows, [])
        assert "dirty tree" in path.read_text(encoding="utf-8")

    def test_omits_the_baseline_section_when_it_did_not_run(self, tmp_path: Path) -> None:
        """An empty table would read as "prompting did nothing"."""
        layer_rows, best, steering_rows = self._rows()
        path = run_pilot.write_report(
            tmp_path, self._manifest(), layer_rows, best, steering_rows, []
        )
        assert "against prompting" not in path.read_text(encoding="utf-8")


class TestMain:
    """End to end: a directory a reader can pick up and check."""

    @pytest.fixture(autouse=True)
    def _patch_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Run the pipeline against the fake instead of downloading a model."""
        built = make_pilot_engine()
        monkeypatch.setattr(built, "load_model", lambda: built.model)
        monkeypatch.setattr(run_pilot, "CulturalRepE", lambda **_kwargs: built)

    def test_writes_every_artefact(self, tmp_path: Path) -> None:
        """Manifest, CSVs and report all have to land together."""
        exit_code = run_pilot.main(
            [
                "--model",
                "dummy/pilot",
                "--concepts",
                "wasta_001,diyafa_001",
                "--strengths",
                "0.0,0.2",
                "--output-dir",
                str(tmp_path / "run"),
                "--no-baselines",
            ]
        )
        out = tmp_path / "run"
        assert exit_code == 0
        assert (out / "manifest.json").exists()
        assert (out / "layer_sweep.csv").exists()
        assert (out / "steering_sweep.csv").exists()
        assert (out / "README.md").exists()

    def test_manifest_pins_the_run(self, tmp_path: Path) -> None:
        """Seed, dataset hash and the strength grid all belong on disk."""
        run_pilot.main(
            [
                "--model",
                "dummy/pilot",
                "--concepts",
                "wasta_001",
                "--strengths",
                "0.0,0.2",
                "--seed",
                "7",
                "--output-dir",
                str(tmp_path / "run"),
                "--no-baselines",
            ]
        )
        manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
        assert manifest["seed"] == 7
        assert manifest["settings"]["relative_strengths"] == [0.0, 0.2]
        assert manifest["settings"]["baselines_run"] is False
        assert len(manifest["dataset"]["sha256"]) == 64

    def test_defaults_to_every_concept_in_the_dataset(self, tmp_path: Path) -> None:
        """Hand-listing concepts is how a sweep quietly loses coverage."""
        run_pilot.main(
            [
                "--model",
                "dummy/pilot",
                "--strengths",
                "0.0",
                "--output-dir",
                str(tmp_path / "run"),
                "--no-baselines",
            ]
        )
        manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
        assert len(manifest["concepts"]) == len(run_pilot.load_concept_names(DATASET_PATH))

    def test_unknown_concept_fails_loudly(self, tmp_path: Path) -> None:
        """A typo must not silently shrink the experiment."""
        with pytest.raises(SystemExit, match="Unknown concept"):
            run_pilot.main(
                [
                    "--model",
                    "dummy/pilot",
                    "--concepts",
                    "not_a_concept_999",
                    "--output-dir",
                    str(tmp_path / "run"),
                    "--no-baselines",
                ]
            )

    def test_extracts_at_the_layer_the_probe_chose(self, tmp_path: Path) -> None:
        """Extraction must follow the sweep, not a layer chosen by theory."""
        run_pilot.main(
            [
                "--model",
                "dummy/pilot",
                "--concepts",
                "wasta_001",
                "--strengths",
                "0.2",
                "--output-dir",
                str(tmp_path / "run"),
                "--no-baselines",
            ]
        )
        out = tmp_path / "run"
        layer_rows = _read_csv(out / "layer_sweep.csv")
        best = max(layer_rows, key=lambda r: float(r["probe_accuracy"]))
        steering_rows = _read_csv(out / "steering_sweep.csv")
        assert steering_rows[0]["layer"] == best["layer"]

    def test_baselines_produce_a_comparison_table(self, tmp_path: Path) -> None:
        """The prompting arm is the point of comparison, not an optional extra."""
        run_pilot.main(
            [
                "--model",
                "dummy/pilot",
                "--concepts",
                "wasta_001",
                "--strengths",
                "0.2",
                "--max-new-tokens",
                "4",
                "--output-dir",
                str(tmp_path / "run"),
            ]
        )
        out = tmp_path / "run"
        rows = _read_csv(out / "baseline_comparison.csv")
        assert len(rows) > 1
        assert "against prompting" in (out / "README.md").read_text(encoding="utf-8")
        assert (out / "generations" / "wasta_001.json").exists()

    def test_is_deterministic_for_a_fixed_seed(self, tmp_path: Path) -> None:
        """Two runs of the same command must produce the same numbers."""
        argv = [
            "--model",
            "dummy/pilot",
            "--concepts",
            "wasta_001",
            "--strengths",
            "0.0,0.2",
            "--seed",
            "11",
            "--no-baselines",
        ]
        run_pilot.main([*argv, "--output-dir", str(tmp_path / "a")])
        run_pilot.main([*argv, "--output-dir", str(tmp_path / "b")])
        assert _read_csv(tmp_path / "a" / "layer_sweep.csv") == _read_csv(
            tmp_path / "b" / "layer_sweep.csv"
        )
        assert _read_csv(tmp_path / "a" / "steering_sweep.csv") == _read_csv(
            tmp_path / "b" / "steering_sweep.csv"
        )


def test_resid_hook_name_is_the_one_the_fake_answers() -> None:
    """Guards the fake against a rename of the hook the engine reads."""
    assert RESID_POST_HOOK.format(layer=0) == "blocks.0.hook_resid_post"
