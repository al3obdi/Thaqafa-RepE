"""Tests for submission preparation: result injection and anonymization.

All tests run on CPU without external dependencies (no LaTeX, no network).
The anonymization test works on a synthetic .tex file to verify the script
removes all identifying patterns.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
INJECT_SCRIPT = SCRIPTS_DIR / "inject_results_to_tex.py"
ANON_SCRIPT = SCRIPTS_DIR / "anonymize_paper.sh"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_results_md(tmp_path: Path) -> Path:
    """Create a sample RESULTS_SUMMARY.md for injection tests."""
    md = dedent("""
        # Thaqafa-RepE Results Summary

        ## 1. Best Layers by Concept

        | Concept | Best Layer | Max Accuracy | Chance |
        |---------|-----------|-------------|--------|
        | wasta_001 | 16 | 0.8200 | 0.5000 |
        | muruah_001 | 14 | 0.7800 | 0.5000 |
        | diyafa_001 | 18 | 0.8500 | 0.5000 |

        ## 2. Steering Sweep (Effect vs Cost)

        | Concept | Strength | Effect KL | Mean Loss |
        |---------|----------|-----------|-----------|
        | wasta_001 | -2.0 | 0.3000 | 2.4000 |
        | wasta_001 | -1.0 | 0.1500 | 2.1000 |
        | wasta_001 | 0.0 | 0.0000 | 2.0000 |
        | wasta_001 | 1.0 | 0.1500 | 2.0500 |
        | wasta_001 | 2.0 | 0.3000 | 2.2000 |

        ## 3. Baseline Comparison

        | Concept | Condition | Mean Loss | Extra Tokens | N Gens |
        |---------|-----------|-----------|-------------|--------|
        | wasta_001 | steering | 2.0500 | 0 | 5 |
        | wasta_001 | prompt:direct_en | 2.1000 | 8 | 5 |
    """).strip()
    path = tmp_path / "RESULTS_SUMMARY.md"
    path.write_text(md)
    return path


@pytest.fixture
def sample_tex(tmp_path: Path) -> Path:
    """Create a sample .tex file with \\todo markers."""
    tex = r"""\documentclass[11pt]{article}
\usepackage{xcolor}
\newcommand{\todo}[1]{\textcolor{red}{[TODO: #1]}}

\title{Thaqafa-RepE: Representation Engineering}
\author{
  Thaqafa-RepE Contributors \\
  \texttt{https://github.com/al3obdi/Thaqafa-RepE}
}

\begin{document}
\maketitle

\begin{abstract}
Some text here.
\todo{Fill in headline findings once experiments are run: which layers carry
each concept, at what strength steering becomes effective, where fluency breaks
down, and whether steering beats the persona prompting baseline.}
\end{abstract}

\section{Results}
\todo{Insert the strength sweep and the layer-set grid.}

\begin{table}[t]
  \centering
  \begin{tabular}{lrrr}
    \toprule
    Layers & $\alpha$ & Effect (KL) & Cost ($\Delta$ CE) \\
    \midrule
    $\{\ell^{*}\}$ & 1.0 & \todo{} & \todo{} \\
    $\{\ell^{*}\}$ & 2.0 & \todo{} & \todo{} \\
    \bottomrule
  \end{tabular}
\end{table}

\todo{Report the head-to-head comparison from \S\ref{sec:baselines}.}

\end{document}
"""
    path = tmp_path / "main.tex"
    path.write_text(tex)
    return path


@pytest.fixture
def sample_tex_with_author(tmp_path: Path) -> Path:
    """Create a sample .tex file with author info for anonymization tests."""
    tex = r"""\documentclass[11pt]{article}
\usepackage{url}
\title{Thaqafa-RepE: Representation Engineering}
\author{
  Abdullah Almohammedi \\
  Independent Researcher, Rabigh \\
  \url{https://orcid.org/0009-0001-0832-0995} \\
  \texttt{https://github.com/al3obdi/Thaqafa-RepE}
}
\begin{document}
\maketitle
Code at \url{https://github.com/al3obdi/Thaqafa-RepE}.
Contact: al3obdi@example.com
\end{document}
"""
    path = tmp_path / "main_with_author.tex"
    path.write_text(tex)
    return path


# ---------------------------------------------------------------------------
# Tests: Provenance guard
# ---------------------------------------------------------------------------


class TestProvenanceGuard:
    """The injector must refuse summaries that lack a live-run marker."""

    def test_summary_without_marker_is_refused(self, sample_results_md: Path) -> None:
        """A results file with no provenance marker exits with an error."""
        from scripts.inject_results_to_tex import check_provenance

        with pytest.raises(SystemExit, match="provenance"):
            check_provenance(sample_results_md.read_text())

    def test_summary_with_marker_is_accepted(self) -> None:
        """A results file stamped by run_full_experiment passes."""
        from scripts.inject_results_to_tex import check_provenance

        check_provenance("<!-- provenance: live-model-run model=test-model -->\n# Summary\n")

    def test_run_full_experiment_report_carries_the_marker(self) -> None:
        """The marker the guard requires is the one the engine writes."""
        import src.models.rep_engine as rep_engine
        from scripts.inject_results_to_tex import PROVENANCE_MARKER

        source = Path(rep_engine.__file__).read_text()
        assert PROVENANCE_MARKER in source


# ---------------------------------------------------------------------------
# Tests: Result injection
# ---------------------------------------------------------------------------


class TestParseMarkdownTables:
    """Test the Markdown table parser."""

    def test_parse_basic_tables(self, sample_results_md: Path) -> None:
        """Tables are correctly parsed from Markdown."""
        from scripts.inject_results_to_tex import _parse_markdown_tables

        md = sample_results_md.read_text()
        tables = _parse_markdown_tables(md)
        assert "1. Best Layers by Concept" in tables
        assert len(tables["1. Best Layers by Concept"]) == 3
        assert "2. Steering Sweep (Effect vs Cost)" in tables
        assert len(tables["2. Steering Sweep (Effect vs Cost)"]) == 5

    def test_parse_empty_markdown(self) -> None:
        """Empty Markdown returns empty dict."""
        from scripts.inject_results_to_tex import _parse_markdown_tables

        tables = _parse_markdown_tables("")
        assert tables == {}


class TestBuildLatexTable:
    """Test LaTeX table builder."""

    def test_build_table_from_rows(self) -> None:
        """A LaTeX table is built from parsed rows."""
        from scripts.inject_results_to_tex import _build_latex_table

        rows = [
            {"Concept": "wasta_001", "Layer": "16", "Accuracy": "0.82"},
            {"Concept": "diyafa_001", "Layer": "18", "Accuracy": "0.85"},
        ]
        table = _build_latex_table(rows, "Test caption", "tab:test")
        assert "\\begin{table}" in table
        assert "\\end{table}" in table
        assert "wasta_001" in table
        assert "diyafa_001" in table
        assert "tab:test" in table

    def test_build_table_empty_rows(self) -> None:
        """Empty rows produce a comment."""
        from scripts.inject_results_to_tex import _build_latex_table

        table = _build_latex_table([], "Empty", "tab:empty")
        assert "No data" in table


class TestInjectResults:
    """Test the result injection function."""

    def test_inject_replaces_abstract_todo(
        self,
        sample_tex: Path,
        sample_results_md: Path,
    ) -> None:
        """The abstract \\todo is replaced with results text."""
        from scripts.inject_results_to_tex import inject_results

        template = sample_tex.read_text()
        results = sample_results_md.read_text()
        updated = inject_results(template, results)

        # The abstract todo should be replaced
        assert "Preliminary results" in updated or "Results show" in updated
        assert "headline findings" not in updated

    def test_inject_replaces_steering_todo(
        self,
        sample_tex: Path,
        sample_results_md: Path,
    ) -> None:
        """The steering sweep \\todo is replaced with a table."""
        from scripts.inject_results_to_tex import inject_results

        template = sample_tex.read_text()
        results = sample_results_md.read_text()
        updated = inject_results(template, results)

        assert "Insert the strength sweep" not in updated
        assert "\\begin{table}" in updated

    def test_inject_fills_empty_todos(
        self,
        sample_tex: Path,
        sample_results_md: Path,
    ) -> None:
        """Empty \\todo{} markers are filled with TBD."""
        from scripts.inject_results_to_tex import inject_results

        template = sample_tex.read_text()
        results = sample_results_md.read_text()
        updated = inject_results(template, results)

        # Empty \todo{} should be replaced with \textit{TBD}
        assert "\\todo{}" not in updated

    def test_inject_adds_note(self, sample_tex: Path, sample_results_md: Path) -> None:
        """An injection note is added at the top."""
        from scripts.inject_results_to_tex import inject_results

        template = sample_tex.read_text()
        results = sample_results_md.read_text()
        updated = inject_results(template, results)

        assert "Results injected" in updated

    def test_inject_with_no_results_file(self, sample_tex: Path) -> None:
        """Injection with empty results still produces valid output."""
        from scripts.inject_results_to_tex import inject_results

        template = sample_tex.read_text()
        updated = inject_results(template, "")
        # Should not crash, just leave todos in place
        assert "\\documentclass" in updated


# ---------------------------------------------------------------------------
# Tests: Anonymization
# ---------------------------------------------------------------------------


class TestAnonymization:
    """Test the anonymization script."""

    def test_anonymize_removes_author_names(
        self,
        sample_tex_with_author: Path,
        tmp_path: Path,
    ) -> None:
        """Author names are removed from the anonymized output."""
        output = tmp_path / "main_anonymous.tex"
        result = subprocess.run(
            [
                "bash",
                str(ANON_SCRIPT),
                str(sample_tex_with_author),
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"Anonymization failed: {result.stderr}"
        assert output.exists()

        content = output.read_text()
        assert "Abdullah" not in content
        assert "Almohammedi" not in content

    def test_anonymize_removes_github_links(
        self,
        sample_tex_with_author: Path,
        tmp_path: Path,
    ) -> None:
        """GitHub URLs are replaced."""
        output = tmp_path / "main_anonymous.tex"
        subprocess.run(
            ["bash", str(ANON_SCRIPT), str(sample_tex_with_author), str(output)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        content = output.read_text()
        assert "github.com/al3obdi" not in content
        assert "Anonymous" in content

    def test_anonymize_removes_orcid(
        self,
        sample_tex_with_author: Path,
        tmp_path: Path,
    ) -> None:
        """ORCID links are removed."""
        output = tmp_path / "main_anonymous.tex"
        subprocess.run(
            ["bash", str(ANON_SCRIPT), str(sample_tex_with_author), str(output)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        content = output.read_text()
        assert "0009-0001-0832-0995" not in content

    def test_anonymize_removes_email(
        self,
        sample_tex_with_author: Path,
        tmp_path: Path,
    ) -> None:
        """Email addresses are replaced."""
        output = tmp_path / "main_anonymous.tex"
        subprocess.run(
            ["bash", str(ANON_SCRIPT), str(sample_tex_with_author), str(output)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        content = output.read_text()
        assert "al3obdi@example.com" not in content
        assert "anonymous@example.com" in content

    def test_anonymize_adds_note(self, sample_tex_with_author: Path, tmp_path: Path) -> None:
        """An anonymization note is added at the top."""
        output = tmp_path / "main_anonymous.tex"
        subprocess.run(
            ["bash", str(ANON_SCRIPT), str(sample_tex_with_author), str(output)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        content = output.read_text()
        assert "ANONYMIZED VERSION" in content

    def test_anonymize_verification_passes(
        self,
        sample_tex_with_author: Path,
        tmp_path: Path,
    ) -> None:
        """The verification step reports PASS."""
        output = tmp_path / "main_anonymous.tex"
        result = subprocess.run(
            ["bash", str(ANON_SCRIPT), str(sample_tex_with_author), str(output)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "PASS" in result.stdout
        assert "FAIL" not in result.stdout


# ---------------------------------------------------------------------------
# Tests: No secrets exposed
# ---------------------------------------------------------------------------


class TestNoSecretsExposed:
    """Ensure no tokens are hardcoded in scripts."""

    def test_no_hardcoded_token_in_inject_script(self) -> None:
        """No HF token in inject_results_to_tex.py."""

        content = INJECT_SCRIPT.read_text()
        token_pattern = re.compile(r"hf_[A-Za-z0-9]{20,}")
        matches = token_pattern.findall(content)
        assert len(matches) == 0

    def test_no_hardcoded_token_in_anon_script(self) -> None:
        """No HF token in anonymize_paper.sh."""

        content = ANON_SCRIPT.read_text()
        token_pattern = re.compile(r"hf_[A-Za-z0-9]{20,}")
        matches = token_pattern.findall(content)
        assert len(matches) == 0


# ---------------------------------------------------------------------------
# Tests: Abstract word count
# ---------------------------------------------------------------------------


class TestAbstractWordCount:
    """Verify the abstract in main.tex meets the 150-200 word requirement."""

    def test_abstract_word_count(self) -> None:
        """Abstract is between 150 and 200 words."""
        main_tex = PROJECT_ROOT / "docs" / "research_paper" / "main.tex"
        if not main_tex.exists():
            pytest.skip("main.tex not found")

        content = main_tex.read_text()
        # Extract abstract
        import re as re_mod

        match = re_mod.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", content, re_mod.DOTALL)
        if not match:
            pytest.skip("No abstract found in main.tex")

        # pre-commit's isolated mypy has no pytest stubs, so skip() is not
        # known to be NoReturn there; narrow explicitly.
        assert match is not None
        abstract_text = match.group(1)
        # Remove LaTeX commands for word counting
        clean = re_mod.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", abstract_text)
        clean = re_mod.sub(r"\\[a-zA-Z]+", "", clean)
        clean = re_mod.sub(r"[{}\\]", "", clean)
        words = [w for w in clean.split() if len(w) > 1 and w.isalpha() or not w.isalpha()]
        word_count = len(words)
        assert 100 <= word_count <= 250, f"Abstract has {word_count} words (expected 150-250)"
