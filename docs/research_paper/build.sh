#!/usr/bin/env bash
#
# Build the Thaqafa-RepE paper.
#
# Runs the standard four-pass LaTeX cycle: a first pdflatex to emit the .aux
# with its citation keys, bibtex to resolve them against references.bib, then
# two more pdflatex passes so that citation numbers and cross-references
# converge.
#
# Usage:
#   ./build.sh              Build main.pdf with pdflatex
#   ./build.sh --xe         Build with xelatex (needed to render Arabic script)
#   ./build.sh --clean      Remove build artefacts and exit
#
# Requires a TeX distribution on PATH. On Debian or Ubuntu:
#   sudo apt-get install texlive-latex-base texlive-latex-recommended \
#                        texlive-bibtex-extra
# Add texlive-xetex and a font with Arabic coverage for --xe.

set -euo pipefail

readonly DOC="main"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ARTEFACTS=(aux bbl blg log out toc synctex.gz)

cd "$SCRIPT_DIR"

clean() {
    local extension
    for extension in "${ARTEFACTS[@]}"; do
        rm -f "${DOC}.${extension}"
    done
}

engine="pdflatex"
for argument in "$@"; do
    case "$argument" in
        --clean)
            clean
            echo "Removed build artefacts."
            exit 0
            ;;
        --xe)
            engine="xelatex"
            ;;
        -h|--help)
            sed -n '3,18p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: ${argument}" >&2
            echo "Try '$(basename "${BASH_SOURCE[0]}") --help'." >&2
            exit 2
            ;;
    esac
done

if ! command -v "$engine" >/dev/null 2>&1; then
    echo "Error: ${engine} not found on PATH." >&2
    echo "Install a TeX distribution first; see the header of this script." >&2
    exit 1
fi

if ! command -v bibtex >/dev/null 2>&1; then
    echo "Error: bibtex not found on PATH." >&2
    exit 1
fi

echo "==> Pass 1/4: ${engine}"
"$engine" -interaction=nonstopmode -halt-on-error "${DOC}.tex" >/dev/null

echo "==> Pass 2/4: bibtex"
bibtex "${DOC}" >/dev/null

echo "==> Pass 3/4: ${engine}"
"$engine" -interaction=nonstopmode -halt-on-error "${DOC}.tex" >/dev/null

echo "==> Pass 4/4: ${engine}"
"$engine" -interaction=nonstopmode -halt-on-error "${DOC}.tex" >/dev/null

if [[ ! -f "${DOC}.pdf" ]]; then
    echo "Error: ${DOC}.pdf was not produced. Inspect ${DOC}.log." >&2
    exit 1
fi

echo
echo "Built ${SCRIPT_DIR}/${DOC}.pdf"

remaining=$(grep -c '\\todo{' "${DOC}.tex" || true)
if [[ "$remaining" -gt 0 ]]; then
    echo "Note: ${remaining} \\todo markers remain in ${DOC}.tex."
fi
