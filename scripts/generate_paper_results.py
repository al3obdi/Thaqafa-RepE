#!/usr/bin/env python3
"""Generate structured experimental results for the Thaqafa-RepE paper.

This script orchestrates the full experimental pipeline:

1. **Extraction**: Trigger vector extraction on the ZeroGPU Space via
   ``CulturalRepE.extract_via_space()`` and save vectors locally.
2. **Layer sweep**: Run linear probes across all layers to find where
   each concept is most readable.
3. **Steering sweep**: Measure effect (KL divergence) vs cost (fluency loss)
   across a grid of injection strengths.
4. **Baseline comparison**: Compare steering against prompt-based baselines.
5. **Report**: Write a Markdown summary with LaTeX-ready snippets.

All outputs land in ``outputs/paper_results/``.

Usage:
    python scripts/generate_paper_results.py --concepts wasta_001,muruah_001,diyafa_001
    python scripts/generate_paper_results.py --concepts wasta_001 --model meta-llama/Meta-Llama-3-8B-Instruct
    python scripts/generate_paper_results.py --concepts wasta_001 --output-dir outputs/paper_results
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("paper_results")

DEFAULT_CONCEPTS = ["wasta_001", "muruah_001", "diyafa_001"]
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/paper_results"
STRENGTH_GRID = [-2.0, -1.0, 0.0, 1.0, 2.0]

CONCEPT_NAMES: dict[str, str] = {
    "wasta_001": "Wasta (intercession)",
    "muruah_001": "Muruah (manly virtue)",
    "diyafa_001": "Diyafa (hospitality)",
}

EVALUATION_PROMPTS: list[str] = [
    "A guest arrives at your home unexpectedly.",
    "You are asked to recommend a friend for a job.",
    "Describe what it means to be a generous host.",
    "A stranger needs help finding direction.",
    "Your cousin asks you to intervene on their behalf.",
]


def _ensure_output_dir(output_dir: str) -> Path:
    """Create the output directory structure.

    Args:
        output_dir: Root output path.

    Returns:
        The resolved Path for the output directory.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)
    (root / "generations").mkdir(exist_ok=True)
    return root


def _save_vectors_json(
    vectors: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Save extracted vectors to JSON.

    Args:
        vectors: Mapping from concept_id to tensor.
        output_dir: Output directory.

    Returns:
        Path to the saved file.
    """
    out: dict[str, Any] = {}
    for cid, vec in vectors.items():
        if hasattr(vec, "tolist"):
            out[cid] = vec.tolist()
        else:
            out[cid] = list(vec)
        out[f"{cid}_shape"] = list(getattr(vec, "shape", (len(vec),)))

    path = output_dir / "vectors.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("Saved vectors to %s", path)
    return path


def _save_layer_sweep_csv(
    sweep_data: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> Path:
    """Save layer sweep results to CSV.

    Args:
        sweep_data: Per-concept list of probe result dicts.
        output_dir: Output directory.

    Returns:
        Path to the CSV file.
    """
    path = output_dir / "layer_sweep.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["concept_id", "layer", "probe_accuracy", "chance_accuracy"]
        )
        writer.writeheader()
        for concept_id, rows in sweep_data.items():
            for row in rows:
                writer.writerow(
                    {
                        "concept_id": concept_id,
                        "layer": row["layer"],
                        "probe_accuracy": row["accuracy"],
                        "chance_accuracy": row["chance"],
                    }
                )
    logger.info("Saved layer sweep to %s", path)
    return path


def _save_steering_sweep_csv(
    sweep_data: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> Path:
    """Save steering sweep results to CSV.

    Args:
        sweep_data: Per-concept list of steering result dicts.
        output_dir: Output directory.

    Returns:
        Path to the CSV file.
    """
    path = output_dir / "steering_sweep.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["concept_id", "strength", "effect_kl", "mean_loss"])
        writer.writeheader()
        for concept_id, rows in sweep_data.items():
            for row in rows:
                writer.writerow(
                    {
                        "concept_id": concept_id,
                        "strength": row["strength"],
                        "effect_kl": row["effect_kl"],
                        "mean_loss": row["mean_loss"],
                    }
                )
    logger.info("Saved steering sweep to %s", path)
    return path


def _save_baseline_comparison_csv(
    comparison_data: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> Path:
    """Save baseline comparison results to CSV.

    Args:
        comparison_data: Per-concept list of condition result dicts.
        output_dir: Output directory.

    Returns:
        Path to the CSV file.
    """
    path = output_dir / "baseline_comparison.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "concept_id",
                "condition",
                "mean_continuation_loss",
                "extra_input_tokens",
                "n_generations",
            ],
        )
        writer.writeheader()
        for concept_id, rows in comparison_data.items():
            for row in rows:
                writer.writerow(
                    {
                        "concept_id": concept_id,
                        "condition": row["condition"],
                        "mean_continuation_loss": row["mean_continuation_loss"],
                        "extra_input_tokens": row["extra_input_tokens"],
                        "n_generations": row["n_generations"],
                    }
                )
    logger.info("Saved baseline comparison to %s", path)
    return path


def _save_generations(
    comparison_data: dict[str, dict[str, dict[str, str]]],
    output_dir: Path,
) -> None:
    """Save generation outputs per concept and condition.

    Args:
        comparison_data: Per-concept, per-condition generation dicts.
        output_dir: Output directory.
    """
    gen_dir = output_dir / "generations"
    gen_dir.mkdir(exist_ok=True)
    for concept_id, conditions in comparison_data.items():
        for condition_name, gens in conditions.items():
            safe_name = condition_name.replace(":", "_").replace("/", "_")
            path = gen_dir / f"{concept_id}_{safe_name}.txt"
            with open(path, "w") as f:
                for prompt, gen in gens.items():
                    f.write(f"PROMPT: {prompt}\n")
                    f.write(f"OUTPUT: {gen}\n")
                    f.write("---\n")
    logger.info("Saved generations to %s", gen_dir)


def _generate_layer_sweep_plot(
    sweep_data: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> Path | None:
    """Generate a layer sweep plot.

    Args:
        sweep_data: Per-concept layer sweep data.
        output_dir: Output directory.

    Returns:
        Path to the plot, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    for concept_id, rows in sweep_data.items():
        layers = [r["layer"] for r in rows]
        accs = [r["accuracy"] for r in rows]
        ax.plot(layers, accs, marker="o", label=concept_id)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe Accuracy")
    ax.set_title("Layer Sweep: Linear Probe Accuracy by Concept")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = output_dir / "figures" / "layer_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved layer sweep plot to %s", path)
    return path


def _generate_steering_plot(
    sweep_data: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> Path | None:
    """Generate an effect vs cost plot.

    Args:
        sweep_data: Per-concept steering sweep data.
        output_dir: Output directory.

    Returns:
        Path to the plot, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    for concept_id, rows in sweep_data.items():
        strengths = [r["strength"] for r in rows]
        effect = [r["effect_kl"] for r in rows]
        ax.plot(strengths, effect, marker="s", label=f"{concept_id} (KL)")

    ax.set_xlabel("Injection Strength")
    ax.set_ylabel("KL Divergence (effect)")
    ax.set_title("Steering Sweep: Effect vs Strength")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = output_dir / "figures" / "effect_vs_cost.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved steering plot to %s", path)
    return path


def _generate_markdown_report(
    layer_data: dict[str, list[dict[str, Any]]],
    steering_data: dict[str, list[dict[str, Any]]],
    baseline_data: dict[str, list[dict[str, Any]]],
    best_layers: dict[str, int],
    output_dir: Path,
) -> Path:
    """Generate a Markdown results summary.

    Args:
        layer_data: Per-concept layer sweep results.
        steering_data: Per-concept steering sweep results.
        baseline_data: Per-concept baseline comparison results.
        best_layers: Per-concept best layer index.
        output_dir: Output directory.

    Returns:
        Path to the Markdown report.
    """
    lines = [
        "# Thaqafa-RepE Results Summary",
        "",
        f"Generated: {Path(output_dir).name}",
        "",
        "## 1. Best Layers by Concept",
        "",
        "| Concept | Best Layer | Max Accuracy | Chance |",
        "|---------|-----------|-------------|--------|",
    ]

    for concept_id, layer in best_layers.items():
        rows = layer_data.get(concept_id, [])
        if rows:
            best_row = max(rows, key=lambda r: r["accuracy"])
            lines.append(
                f"| {concept_id} | {layer} | {best_row['accuracy']:.4f} | {best_row['chance']:.4f} |"
            )
        else:
            lines.append(f"| {concept_id} | {layer} | N/A | N/A |")

    lines.extend(
        [
            "",
            "## 2. Steering Sweep (Effect vs Cost)",
            "",
            "| Concept | Strength | Effect KL | Mean Loss |",
            "|---------|----------|-----------|-----------|",
        ]
    )

    for concept_id, rows in steering_data.items():
        for row in rows:
            lines.append(
                f"| {concept_id} | {row['strength']:.1f} | {row['effect_kl']:.4f} | {row['mean_loss']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## 3. Optimal Strength (Knee of Curve)",
            "",
            "| Concept | Optimal Strength | Rationale |",
            "|---------|-----------------|-----------|",
        ]
    )

    for concept_id, rows in steering_data.items():
        if not rows:
            continue
        # Find the strength where effect_kl / mean_loss ratio is highest
        best = max(
            rows,
            key=lambda r: abs(r["effect_kl"]) / max(abs(r["mean_loss"]), 1e-8),
        )
        lines.append(
            f"| {concept_id} | {best['strength']:.1f} | "
            f"Max effect/cost ratio (KL={best['effect_kl']:.4f}, loss={best['mean_loss']:.4f}) |"
        )

    lines.extend(
        [
            "",
            "## 4. Baseline Comparison",
            "",
            "| Concept | Condition | Mean Cont. Loss | Extra Tokens | N Gens |",
            "|---------|-----------|----------------|-------------|--------|",
        ]
    )

    for concept_id, rows in baseline_data.items():
        for row in rows:
            lines.append(
                f"| {concept_id} | {row['condition']} | "
                f"{row['mean_continuation_loss']:.4f} | "
                f"{row['extra_input_tokens']} | {row['n_generations']} |"
            )

    lines.extend(
        [
            "",
            "## 5. LaTeX-Ready Snippets",
            "",
            "### Layer sweep table",
            "",
            "```latex",
            r"\begin{tabular}{lccc}",
            r"  \textbf{Concept} & \textbf{Best Layer} & \textbf{Accuracy} & \textbf{Chance} \\\\",
        ]
    )

    for concept_id, layer in best_layers.items():
        rows = layer_data.get(concept_id, [])
        if rows:
            best_row = max(rows, key=lambda r: r["accuracy"])
            lines.append(
                f"  {concept_id} & {layer} & {best_row['accuracy']:.3f} & {best_row['chance']:.3f} \\\\"
            )

    lines.extend(
        [
            r"\end{tabular}",
            "```",
            "",
            "### Steering sweep table",
            "",
            "```latex",
            r"\begin{tabular}{lcrr}",
            r"  \textbf{Concept} & \textbf{Strength} & \textbf{KL} & \textbf{Loss} \\\\",
        ]
    )

    for concept_id, rows in steering_data.items():
        for row in rows:
            lines.append(
                f"  {concept_id} & {row['strength']:.1f} & {row['effect_kl']:.4f} & {row['mean_loss']:.4f} \\\\"
            )

    lines.extend(
        [
            r"\end{tabular}",
            "```",
            "",
            "## 6. Summary Statistics",
            "",
        ]
    )

    n_concepts = len(best_layers)
    n_layers_total = sum(len(v) for v in layer_data.values())
    n_strengths_total = sum(len(v) for v in steering_data.values())
    lines.append(f"- Concepts evaluated: {n_concepts}")
    lines.append(f"- Total layer probes: {n_layers_total}")
    lines.append(f"- Total steering evaluations: {n_strengths_total}")
    lines.append("")
    lines.append("## How to Use These Results")
    lines.append("")
    lines.append("1. Open `docs/research_paper/main.tex`")
    lines.append("2. Search for `\\todo` markers")
    lines.append("3. Copy the relevant LaTeX snippets above into the corresponding sections")
    lines.append("4. Replace placeholder figures with:")
    lines.append("   - `figures/layer_sweep.png` for the probe accuracy plot")
    lines.append("   - `figures/effect_vs_cost.png` for the steering sweep plot")
    lines.append("5. Rebuild: `cd docs/research_paper && ./build.sh`")

    path = output_dir / "RESULTS_SUMMARY.md"
    path.write_text("\n".join(lines))
    logger.info("Saved results summary to %s", path)
    return path


def main() -> None:
    """Entry point for the paper results generation script."""
    parser = argparse.ArgumentParser(
        description="Generate structured experimental results for the Thaqafa-RepE paper.",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default=",".join(DEFAULT_CONCEPTS),
        help=f"Comma-separated concept IDs (default: {','.join(DEFAULT_CONCEPTS)})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model identifier (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    concept_ids = [c.strip() for c in args.concepts.split(",") if c.strip()]
    if not concept_ids:
        print("ERROR: No valid concept IDs provided.", file=sys.stderr)
        raise SystemExit(1)

    output_dir = _ensure_output_dir(args.output_dir)

    print("=" * 60)
    print("  Thaqafa-RepE Paper Results Generation")
    print("=" * 60)
    print(f"  Concepts: {', '.join(concept_ids)}")
    print(f"  Model: {args.model}")
    print(f"  Output: {output_dir}")
    print(f"  Strengths: {STRENGTH_GRID}")
    print("=" * 60)
    print()

    # Ensure src is importable
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.models.rep_engine import CulturalRepE

    engine = CulturalRepE(model_name=args.model)

    # Run the full experiment pipeline
    results = engine.run_full_experiment(
        concept_ids=concept_ids,
        output_dir=str(output_dir),
    )

    print()
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Vectors saved: {results.get('vectors_saved', False)}")
    print(f"  Layer sweep rows: {sum(len(v) for v in results.get('layer_sweep', {}).values())}")
    print(
        f"  Steering sweep rows: {sum(len(v) for v in results.get('steering_sweep', {}).values())}"
    )
    print(
        f"  Baseline comparison rows: {sum(len(v) for v in results.get('baseline_comparison', {}).values())}"
    )
    print(f"  Best layers: {results.get('best_layers', {})}")
    print(f"  Markdown report: {results.get('markdown_report', 'N/A')}")
    print("=" * 60)
    print()
    print("Next: open outputs/paper_results/RESULTS_SUMMARY.md")
    print("      copy LaTeX snippets into docs/research_paper/main.tex")


if __name__ == "__main__":
    main()
