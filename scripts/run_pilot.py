#!/usr/bin/env python3
"""Run the Thaqafa-RepE pilot experiment end to end and write it to disk.

This is the reproducible entry point behind every number the paper reports.
It loads a model locally, then for each concept:

1. probes every layer to find where the concept is linearly readable,
2. extracts the concept vector at the layer that scored highest,
3. sweeps norm-relative injection strengths, recording effect (KL) and cost
   (continuation loss),
4. compares steering against prompt-engineering baselines.

Everything it writes is accompanied by ``manifest.json``, which pins the
commit, the seed, the dataset hash and the installed package versions, so a
reader can tell whether a rerun is comparable. Nothing here estimates,
extrapolates or fills in a missing measurement: a phase that cannot run fails
loudly instead of producing a plausible-looking number.

Usage:
    python scripts/run_pilot.py --model gpt2 --output-dir results/pilot_gpt2
    python scripts/run_pilot.py --model gpt2 --concepts wasta_001,diyafa_001
    python scripts/run_pilot.py --model gpt2 --no-baselines   # skip generation
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_builder import load_concepts  # noqa: E402
from src.models.rep_engine import EVALUATION_PROMPTS, CulturalRepE  # noqa: E402
from src.utils.baselines import compare_steering_vs_prompting  # noqa: E402
from src.utils.evaluation import evaluate_steering  # noqa: E402
from src.utils.probes import best_layer, sweep_layers_with_probe  # noqa: E402
from src.utils.provenance import build_manifest, set_global_seed  # noqa: E402

logger = logging.getLogger("pilot")

DEFAULT_MODEL = "gpt2"
DEFAULT_OUTPUT_DIR = "results/pilot_gpt2"
DEFAULT_SEED = 42
DEFAULT_STRENGTHS: tuple[float, ...] = (-0.4, -0.2, 0.0, 0.2, 0.4)
"""Injection coefficients as a fraction of the layer's mean residual norm.

Absolute coefficients are not comparable across layers or models, because the
residual stream grows in norm with depth. See ``calibrate_layer_norms``.
"""

DEFAULT_MAX_NEW_TOKENS = 24
DEFAULT_BASELINE_STRENGTH = 0.2


def load_concept_names(dataset_path: Path | str) -> dict[str, str]:
    """Map every concept id in the dataset to its English display name.

    Reading the names from the dataset rather than a hard-coded table means a
    concept added to the JSONL is immediately usable, and no run can silently
    label a concept by its raw identifier because someone forgot to extend a
    dict.

    Args:
        dataset_path: JSONL concept dataset.

    Returns:
        A mapping from concept id to display name, in file order.

    Raises:
        ValueError: If the dataset holds no entries.
    """
    concepts = load_concepts(dataset_path)
    if not concepts:
        raise ValueError(f"No concepts found in {dataset_path}")
    return {concept.concept_id: concept.concept_en for concept in concepts}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write rows to a CSV with a header, creating parent directories.

    Args:
        path: Destination file.
        fieldnames: Column order.
        rows: Records to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s (%d rows)", path, len(rows))


def run_layer_sweep(
    engine: CulturalRepE,
    concept_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Probe every layer for every concept.

    Args:
        engine: Engine with a loaded model.
        concept_ids: Concepts to probe.

    Returns:
        A ``(rows, best_layers)`` pair, where rows are ready for CSV and
        ``best_layers`` maps each concept to its highest-scoring layer.
    """
    rows: list[dict[str, Any]] = []
    best_layers: dict[str, int] = {}
    for concept_id in concept_ids:
        logger.info("probing layers for %s", concept_id)
        results = sweep_layers_with_probe(engine, concept_id)
        for result in results.values():
            rows.append(
                {
                    "concept_id": concept_id,
                    "layer": result.layer,
                    "probe_accuracy": round(result.accuracy, 6),
                    "probe_std": round(result.std, 6),
                    "chance_accuracy": round(result.chance, 6),
                    "lift_over_chance": round(result.lift_over_chance, 6),
                    "n_samples": result.n_samples,
                }
            )
        best_layers[concept_id] = best_layer(results)
        logger.info(
            "%s: best layer %d (acc %.3f, chance %.3f)",
            concept_id,
            best_layers[concept_id],
            results[best_layers[concept_id]].accuracy,
            results[best_layers[concept_id]].chance,
        )
    return rows, best_layers


def run_steering_sweep(
    engine: CulturalRepE,
    concept_ids: list[str],
    strengths: tuple[float, ...],
    prompts: list[str],
) -> list[dict[str, Any]]:
    """Sweep norm-relative injection strengths for every concept.

    Args:
        engine: Engine with the concept vectors already extracted.
        concept_ids: Concepts to steer.
        strengths: Coefficients as fractions of the layer's residual norm.
        prompts: Texts to score under each strength.

    Returns:
        Rows ready for CSV, one per concept and strength.
    """
    rows: list[dict[str, Any]] = []
    for concept_id in concept_ids:
        logger.info("steering sweep for %s", concept_id)
        results = evaluate_steering(
            engine,
            concept_id,
            prompts,
            strengths=list(strengths),
            generate=False,
            measure_effect=True,
            strength_mode="relative",
        )
        for strength, result in results.items():
            rows.append(
                {
                    "concept_id": concept_id,
                    "relative_strength": strength,
                    "layer": result.layers[0] if result.layers else -1,
                    "effect_kl": round(result.effect_kl, 6),
                    "mean_loss": round(result.mean_loss, 6),
                    "perplexity": round(result.perplexity, 6),
                }
            )
    return rows


def run_baselines(
    engine: CulturalRepE,
    concept_ids: list[str],
    concept_names: dict[str, str],
    prompts: list[str],
    strength: float,
    max_new_tokens: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Compare steering against prompt-engineering baselines and save outputs.

    Args:
        engine: Engine with the concept vectors already extracted.
        concept_ids: Concepts to compare.
        concept_names: Display names used by the instruction templates.
        prompts: Questions every condition answers.
        strength: Norm-relative injection coefficient for the steering arm.
        max_new_tokens: Tokens to generate per prompt.
        output_dir: Root directory; generations land under ``generations/``.

    Returns:
        Rows ready for CSV, one per concept and condition.
    """
    rows: list[dict[str, Any]] = []
    generations_dir = output_dir / "generations"
    generations_dir.mkdir(parents=True, exist_ok=True)

    for concept_id in concept_ids:
        logger.info("baseline comparison for %s", concept_id)
        comparison = compare_steering_vs_prompting(
            engine,
            concept_id,
            concept_names[concept_id],
            prompts,
            strength=strength,
            max_new_tokens=max_new_tokens,
            measure_effect=True,
            strength_mode="relative",
        )
        for row in comparison.rows():
            rows.append({"concept_id": concept_id, **row})

        payload = {
            "concept_id": concept_id,
            "relative_strength": strength,
            "steering_effect_kl": comparison.steering_effect_kl,
            "conditions": {
                name: result.generations for name, result in comparison.conditions.items()
            },
        }
        path = generations_dir / f"{concept_id}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def write_report(
    output_dir: Path,
    manifest: dict[str, Any],
    layer_rows: list[dict[str, Any]],
    best_layers: dict[str, int],
    steering_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> Path:
    """Write the human-readable summary of a pilot run.

    The report states the model and commit in its first lines, because a table
    of accuracies that does not say what produced it invites being quoted out of
    context.

    Args:
        output_dir: Where to write ``README.md``.
        manifest: The run manifest.
        layer_rows: Layer sweep rows.
        best_layers: Best layer per concept.
        steering_rows: Steering sweep rows.
        baseline_rows: Baseline comparison rows.

    Returns:
        The path written.
    """
    model = manifest["model"]["name"]
    commit = manifest["git"]["commit"][:12]
    dirty = " (dirty tree)" if manifest["git"]["dirty"] else ""

    lines = [
        f"# Pilot results: `{model}`",
        "",
        f"- **Model**: `{model}` on {manifest['model']['device']} "
        f"({manifest['model']['dtype']})",
        f"- **Commit**: `{commit}`{dirty}",
        f"- **Run (UTC)**: {manifest['timestamp_utc']}",
        f"- **Seed**: {manifest['seed']}",
        f"- **Dataset SHA-256**: `{manifest['dataset']['sha256'][:16]}...`",
        "",
        "Regenerate with:",
        "",
        "```bash",
        f"python scripts/run_pilot.py --model {model} "
        f"--output-dir {output_dir.as_posix()} --seed {manifest['seed']}",
        "```",
        "",
        "## 1. Where each concept is linearly readable",
        "",
        "Cross-validated logistic-regression accuracy on residual activations,",
        "against the majority-class floor. `+/-` is the standard deviation across",
        "folds; on this many exemplars it is wide, so treat the ranking as a",
        "direction to investigate rather than a result.",
        "",
        "| Concept | Best layer | Accuracy | Chance | Lift |",
        "|---|---|---|---|---|",
    ]

    by_concept: dict[str, list[dict[str, Any]]] = {}
    for row in layer_rows:
        by_concept.setdefault(str(row["concept_id"]), []).append(row)

    for concept_id, layer in best_layers.items():
        row = next(r for r in by_concept[concept_id] if r["layer"] == layer)
        lines.append(
            f"| `{concept_id}` | {layer} | {row['probe_accuracy']:.3f} "
            f"+/- {row['probe_std']:.3f} | {row['chance_accuracy']:.3f} | "
            f"{row['lift_over_chance']:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Steering: effect against cost",
            "",
            "Strength is a fraction of the layer's mean residual norm, so the",
            "same number means the same relative intervention at every layer.",
            "`effect_kl` is the KL divergence between the steered and unsteered",
            "next-token distributions; `mean_loss` is the model's own",
            "cross-entropy on the prompts, which rises as steering damages",
            "fluency.",
            "",
            "| Concept | Strength | Layer | Effect (KL) | Loss |",
            "|---|---|---|---|---|",
        ]
    )
    for row in steering_rows:
        lines.append(
            f"| `{row['concept_id']}` | {row['relative_strength']:+.2f} | "
            f"{row['layer']} | {row['effect_kl']:.4f} | {row['mean_loss']:.4f} |"
        )

    if baseline_rows:
        lines.extend(
            [
                "",
                "## 3. Steering against prompting",
                "",
                "`mean_continuation_loss` is scored by the *unmodified* model, so",
                "it measures damage, not cultural grounding. Which condition is",
                "more culturally appropriate is not decided here; the generations",
                "under `generations/` are kept so raters can judge them blind.",
                "",
                "| Concept | Condition | Cont. loss | Extra input tokens |",
                "|---|---|---|---|",
            ]
        )
        for row in baseline_rows:
            lines.append(
                f"| `{row['concept_id']}` | {row['condition']} | "
                f"{float(row['mean_continuation_loss']):.4f} | "
                f"{row['extra_input_tokens']} |"
            )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Small exemplar sets mean wide confidence intervals; no claim here",
            "  is statistically established.",
            "- The exemplar and contrast sets are not the same size, so the",
            "  majority-class floor sits above 0.5. Compare each accuracy to the",
            "  `Chance` column in the table above, never to 0.5.",
            "- Most of the concept entries are still awaiting native-speaker",
            "  review (`review_status` in the dataset).",
            "- A model with little Arabic capability can only validate that the",
            "  pipeline measures what it claims to; it cannot support a claim",
            "  about Arab cultural concepts.",
            "",
        ]
    )

    path = output_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", path)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to load locally.")
    parser.add_argument("--device", default="cpu", help="Device to run on.")
    parser.add_argument("--dtype", default="float32", help="Numeric precision.")
    parser.add_argument(
        "--concepts",
        default=None,
        help="Comma-separated concept ids. Defaults to every concept in the dataset.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Where to write results.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Global random seed.")
    parser.add_argument(
        "--strengths",
        default=",".join(str(s) for s in DEFAULT_STRENGTHS),
        help="Comma-separated norm-relative injection coefficients.",
    )
    parser.add_argument(
        "--baseline-strength",
        type=float,
        default=DEFAULT_BASELINE_STRENGTH,
        help="Norm-relative coefficient for the steering arm of the baseline table.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Tokens generated per prompt in the baseline comparison.",
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip the generation-based baseline comparison, which dominates runtime.",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the pilot and write every artefact.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    set_global_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = CulturalRepE(model_name=args.model, device=args.device, dtype=args.dtype)
    concept_names = load_concept_names(engine.dataset_path)
    if args.concepts:
        concept_ids = [c.strip() for c in args.concepts.split(",") if c.strip()]
        unknown = [c for c in concept_ids if c not in concept_names]
        if unknown:
            raise SystemExit(f"Unknown concept ids: {', '.join(unknown)}")
    else:
        concept_ids = list(concept_names)
    strengths = tuple(float(s) for s in args.strengths.split(",") if s.strip())

    manifest = build_manifest(
        experiment=output_dir.name,
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        dataset_path=engine.dataset_path,
        concepts=concept_ids,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        repo_root=PROJECT_ROOT,
        extra={
            "relative_strengths": list(strengths),
            "baseline_relative_strength": args.baseline_strength,
            "max_new_tokens": args.max_new_tokens,
            "baselines_run": not args.no_baselines,
            "evaluation_prompts": list(EVALUATION_PROMPTS),
        },
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("loading %s on %s", args.model, args.device)
    engine.load_model()

    layer_rows, best_layers = run_layer_sweep(engine, concept_ids)
    _write_csv(
        output_dir / "layer_sweep.csv",
        [
            "concept_id",
            "layer",
            "probe_accuracy",
            "probe_std",
            "chance_accuracy",
            "lift_over_chance",
            "n_samples",
        ],
        layer_rows,
    )

    # Extract at the layer the probe picked, not at a layer chosen by theory.
    for concept_id, layer in best_layers.items():
        engine.extract_vector(concept_id, layer=layer)

    steering_rows = run_steering_sweep(engine, concept_ids, strengths, list(EVALUATION_PROMPTS))
    _write_csv(
        output_dir / "steering_sweep.csv",
        ["concept_id", "relative_strength", "layer", "effect_kl", "mean_loss", "perplexity"],
        steering_rows,
    )

    baseline_rows: list[dict[str, Any]] = []
    if not args.no_baselines:
        baseline_rows = run_baselines(
            engine,
            concept_ids,
            concept_names,
            list(EVALUATION_PROMPTS),
            args.baseline_strength,
            args.max_new_tokens,
            output_dir,
        )
        _write_csv(
            output_dir / "baseline_comparison.csv",
            [
                "concept_id",
                "condition",
                "mean_continuation_loss",
                "extra_input_tokens",
                "n_generations",
            ],
            baseline_rows,
        )

    write_report(output_dir, manifest, layer_rows, best_layers, steering_rows, baseline_rows)
    logger.info("pilot complete: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
