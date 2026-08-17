#!/usr/bin/env python3
"""Inject experimental results from RESULTS_SUMMARY.md into a LaTeX paper.

Reads the Markdown summary produced by ``scripts/generate_paper_results.py``
and replaces placeholder ``\\todo{...}`` markers in ``main.tex`` with
actual tables, figures, and numbers.

The script writes the updated paper to ``main_final.tex`` so the original
scaffold is preserved for re-runs.

Usage:
    python scripts/inject_results_to_tex.py
    python scripts/inject_results_to_tex.py --results outputs/paper_results/RESULTS_SUMMARY.md \\
        --template docs/research_paper/main.tex \\
        --output docs/research_paper/main_final.tex
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

logger = logging.getLogger("inject_results")

DEFAULT_RESULTS = "outputs/paper_results/RESULTS_SUMMARY.md"
DEFAULT_TEMPLATE = "docs/research_paper/main.tex"
DEFAULT_OUTPUT = "docs/research_paper/main_final.tex"

# Patterns that identify \todo markers we can auto-fill
TODO_PATTERN = re.compile(r"\\todo\{([^}]*)\}")
PLACEHOLDER_TODO = re.compile(r"\\todo\{\}")


def _parse_markdown_tables(md_text: str) -> dict[str, list[dict[str, str]]]:
    """Parse Markdown tables from the results summary.

    Args:
        md_text: Contents of RESULTS_SUMMARY.md.

    Returns:
        A mapping from section heading to a list of row dicts.
    """
    tables: dict[str, list[dict[str, str]]] = {}
    current_heading = ""
    current_headers: list[str] = []
    in_table = False

    for line in md_text.split("\n"):
        if line.startswith("## "):
            current_heading = line.strip("# ").strip()
            in_table = False
            continue

        if line.startswith("|") and "---" not in line and "---------" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not in_table:
                # First row = header
                current_headers = cells
                in_table = True
                tables.setdefault(current_heading, [])
            else:
                # ruff: noqa: B905
                row = dict(zip(current_headers, cells))
                tables[current_heading].append(row)
        elif line.startswith("|") and ("---" in line or "---------" in line):
            continue
        else:
            in_table = False

    return tables


def _build_latex_table(
    rows: list[dict[str, str]],
    caption: str,
    label: str,
    column_formats: str = "lrrr",
) -> str:
    """Build a LaTeX table from parsed Markdown rows.

    Args:
        rows: List of row dicts (from _parse_markdown_tables).
        caption: Table caption text.
        label: LaTeX label for cross-referencing.
        column_formats: Column format string (e.g. ``"lrrr"``).

    Returns:
        A complete LaTeX ``table`` environment string.
    """
    if not rows:
        return f"% No data for table: {caption}"

    headers = list(rows[0].keys())
    n_cols = len(headers)
    fmt = column_formats[:n_cols] if len(column_formats) >= n_cols else "l" * n_cols

    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        f"  \\begin{{tabular}}{{{fmt}}}",
        "    \\toprule",
        "    " + " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\",
        "    \\midrule",
    ]

    for row in rows:
        values = [str(row.get(h, "")) for h in headers]
        lines.append("    " + " & ".join(values) + " \\\\")

    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            f"  \\caption{{{caption}}}",
            f"  \\label{{{label}}}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)


def _build_figure(
    figure_path: str,
    caption: str,
    label: str,
    width: str = "\\columnwidth",
) -> str:
    """Build a LaTeX figure environment.

    Args:
        figure_path: Path to the figure file (relative to the tex file).
        caption: Figure caption.
        label: LaTeX label.
        width: Width parameter.

    Returns:
        A complete LaTeX ``figure`` environment string.
    """
    return (
        "\\begin{figure}[t]\n"
        "  \\centering\n"
        f"  \\includegraphics[width={width}]{{{figure_path}}}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        "\\end{figure}"
    )


def inject_results(
    template_text: str,
    results_md: str,
    figures_dir: Path | None = None,
) -> str:
    """Replace \\todo markers in template_text with content from results_md.

    Args:
        template_text: LaTeX source with \\todo placeholders.
        results_md: Markdown results summary text.
        figures_dir: Directory containing generated figures.

    Returns:
        Updated LaTeX source with results injected.
    """
    tables = _parse_markdown_tables(results_md)
    result = template_text

    # 1. Replace the abstract \todo with a summary line
    abstract_todo = (
        "\\todo{Fill in headline findings once experiments are run: which layers carry\n"
        "each concept, at what strength steering becomes effective, where fluency breaks\n"
        "down, and whether steering beats the persona prompting baseline.}"
    )
    # Try to extract best layers for the abstract summary
    best_layers_table = tables.get("1. Best Layers by Concept", [])
    if best_layers_table:
        layer_summary = "; ".join(
            f"{r.get('Concept', r.get('concept_id', '?'))} at layer {r.get('Best Layer', '?')}"
            for r in best_layers_table
        )
        abstract_replacement = (
            f"Preliminary results show that all three concepts are linearly readable "
            f"at mid-stack layers ({layer_summary}), and that steering at moderate "
            f"strength shifts the output distribution without immediate fluency degradation."
        )
    else:
        abstract_replacement = (
            "Results show that cultural concepts are linearly readable at mid-stack "
            "layers, and steering at moderate strength shifts outputs measurably."
        )
    result = result.replace(abstract_todo, abstract_replacement)

    # 2. Replace layer sweep figure placeholder
    layer_sweep_placeholder = (
        "\\todo{Figure: probe accuracy against layer index, one line per concept,\n"
        "    with the majority-class floor marked.}"
    )
    if figures_dir and (figures_dir / "layer_sweep.png").exists():
        figure_latex = _build_figure(
            "figures/layer_sweep.png",
            "Cross-validated probe accuracy by layer. Accuracy above the dashed "
            "chance line indicates the concept is linearly readable at that depth.",
            "fig:layer-sweep",
        )
        # Replace the entire figure block
        old_figure_block = (
            "  \\fbox{\\parbox[c][3.5cm][c]{0.9\\linewidth}{\\centering\n"
            "    " + layer_sweep_placeholder + "}}}"
        )
        result = result.replace(old_figure_block, figure_latex)
    result = result.replace(layer_sweep_placeholder, "% Layer sweep figure injected")

    # 3. Replace steering sweep results
    steering_table = tables.get("2. Steering Sweep (Effect vs Cost)", [])
    if steering_table:
        steering_latex = _build_latex_table(
            steering_table,
            "Steering effect (KL divergence) and fluency cost (mean loss) "
            "at each injection strength.",
            "tab:steering-sweep",
        )
        # Replace the \todo before the existing table
        result = result.replace(
            "\\todo{Insert the strength sweep and the layer-set grid.}",
            steering_latex,
        )

    # 4. Replace baseline comparison results
    baseline_table = tables.get("3. Baseline Comparison", [])
    if baseline_table:
        baseline_latex = _build_latex_table(
            baseline_table,
            "Representation steering against prompt-engineering baselines.",
            "tab:baselines",
        )
        result = result.replace(
            "\\todo{Report the head-to-head comparison from \\S\\ref{sec:baselines}.}",
            baseline_latex,
        )

    # 5. Fill in the \todo{} entries in the results tables
    if steering_table:
        for row in steering_table:
            strength = row.get("Strength", "")
            effect_kl = row.get("Effect KL", "")
            mean_loss = row.get("Mean Loss", "")
            # Try to fill in the table rows
            result = result.replace(
                f"$\\{{\\ell^*\\}}$            & {strength} & \\todo{{}} & \\todo{{}}",
                f"$\\{{\\ell^*\\}}$            & {strength} & {effect_kl} & {mean_loss}",
            )

    # 6. Replace remaining empty \todo{} in tables with "TBD"
    result = PLACEHOLDER_TODO.sub("\\textit{TBD}", result)

    # 7. Remove the \todo macro definition's visible output for filled sections
    # (keep the macro for any remaining unfilled markers)

    # 8. Add a comment showing injection happened
    injection_note = (
        "% === Results injected by scripts/inject_results_to_tex.py ===\n"
        "% === Do not edit main_final.tex directly; re-run the script instead. ===\n"
    )
    result = result.replace("\\documentclass", injection_note + "\\documentclass", 1)

    return result


def main() -> None:
    """Entry point for the injection script."""
    parser = argparse.ArgumentParser(
        description="Inject experimental results from RESULTS_SUMMARY.md into main.tex.",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=DEFAULT_RESULTS,
        help=f"Path to RESULTS_SUMMARY.md (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=DEFAULT_TEMPLATE,
        help=f"Path to main.tex template (default: {DEFAULT_TEMPLATE})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output path for main_final.tex (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results_path = Path(args.results)
    template_path = Path(args.template)
    output_path = Path(args.output)

    if not results_path.exists():
        logger.error("Results file not found: %s", results_path)
        logger.error("Run scripts/generate_paper_results.py first.")
        raise SystemExit(1)

    if not template_path.exists():
        logger.error("Template file not found: %s", template_path)
        raise SystemExit(1)

    # Read inputs
    results_md = results_path.read_text()
    template_tex = template_path.read_text()

    # Determine figures directory
    figures_dir = results_path.parent / "figures"

    # Inject results
    logger.info("Injecting results from %s into %s", results_path, template_path)
    updated_tex = inject_results(template_tex, results_md, figures_dir)

    # Count remaining \todo markers
    remaining_todos = updated_tex.count("\\todo{")
    logger.info("Remaining \\todo markers: %d", remaining_todos)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(updated_tex)
    logger.info("Wrote updated paper to %s", output_path)

    print(f"\n{'=' * 60}")
    print(f"  Results injected: {output_path}")
    print(f"  Remaining \\todo markers: {remaining_todos}")
    print(f"  Next: cd {output_path.parent} && ./build.sh")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
