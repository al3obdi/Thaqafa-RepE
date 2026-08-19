#!/usr/bin/env bash
# Anonymize a LaTeX paper for double-blind review.
#
# Strips author names, affiliations, ORCID links, and GitHub URLs.
# Outputs an anonymized version that can be submitted without revealing identity.
#
# Usage:
#   ./scripts/anonymize_paper.sh
#   ./scripts/anonymize_paper.sh docs/research_paper/main.tex docs/research_paper/main_anonymous.tex

set -euo pipefail

INPUT="${1:-docs/research_paper/main.tex}"
OUTPUT="${2:-docs/research_paper/main_anonymous.tex}"

if [ ! -f "$INPUT" ]; then
    echo "ERROR: Input file not found: $INPUT" >&2
    exit 1
fi

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT")"

# Build anonymized version using Python for portability
python3 - "$INPUT" "$OUTPUT" << 'PYEOF'
import re
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

with open(input_path, "r") as f:
    content = f.read()

# 1. Replace the author block. The block contains nested braces
# (\texttt{...}), so a [^}]* regex stops at the first closing brace and
# leaves a stray "}" behind, breaking the LaTeX. Walk the braces instead.
def replace_author_block(text: str) -> str:
    marker = "\\author{"
    start = text.find(marker)
    if start == -1:
        return text
    depth = 0
    i = start + len(marker) - 1  # position of the opening brace
    for i in range(start + len(marker) - 1, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
    replacement = (
        "\\author{Anonymous Author(s)\\\\ \\\\ Submitted for double-blind review}"
    )
    return text[:start] + replacement + text[i + 1:]

content = replace_author_block(content)

# 2. Remove GitHub URLs
content = content.replace(
    "https://github.com/al3obdi/Thaqafa-RepE",
    "Anonymous (code released upon acceptance)",
)

# 3. Remove ORCID links
content = re.sub(r"\\url\{https://orcid\.org/[0-9-]*\}", "", content)

# 4. Remove affiliation lines with Rabigh / Independent Researcher
content = re.sub(r"\\\\.*[Rr]abigh.*\\\\", r"\\\\", content)
content = re.sub(r"\\\\.*[Ii]ndependent [Rr]esearcher.*\\\\", r"\\\\", content)

# 5. Remove personal names and identifiers
content = re.sub(r"al3obdi", "anonymous", content, flags=re.IGNORECASE)
content = re.sub(r"Almohammedi", "anonymous", content, flags=re.IGNORECASE)
content = re.sub(r"Abdullah", "anonymous", content, flags=re.IGNORECASE)
content = content.replace("0009-0001-0832-0995", "0000-0000-0000-0000")

# 6. Remove email addresses
content = re.sub(
    r"[a-zA-Z0-9._%+-]*@[a-zA-Z0-9.-]*\.[a-zA-Z]{2,}",
    "anonymous@example.com",
    content,
)

# 7. Add anonymization note at top
note = "% === ANONYMIZED VERSION — do not include identifying information ===\n"
content = note + content

with open(output_path, "w") as f:
    f.write(content)

print("Anonymized paper written to: " + output_path)
PYEOF

echo ""
echo "=== Verification ==="
echo "Checking for identifying information..."
ISSUES=0

for pattern in "al3obdi" "Almohammedi" "Abdullah" "0009-0001-0832-0995" "Rabigh" "github.com/al3obdi"; do
    if grep -q "$pattern" "$OUTPUT" 2>/dev/null; then
        echo "  WARNING: Found '$pattern' in output!" >&2
        ISSUES=$((ISSUES + 1))
    fi
done

if [ "$ISSUES" -eq 0 ]; then
    echo "  PASS: No identifying information found."
else
    echo "  FAIL: $ISSUES identifying pattern(s) found!" >&2
    exit 1
fi

# The identity grep alone cannot tell whether the rewrite broke the LaTeX,
# so compile the anonymized file when a TeX toolchain is available.
if command -v pdflatex >/dev/null 2>&1; then
    echo ""
    echo "=== Compile check ==="
    OUTDIR="$(dirname "$OUTPUT")"
    BASENAME="$(basename "$OUTPUT" .tex)"
    if (cd "$OUTDIR" && pdflatex -interaction=nonstopmode -halt-on-error \
            "$BASENAME.tex" >/dev/null 2>&1); then
        echo "  PASS: $BASENAME.tex compiles."
        rm -f "$OUTDIR/$BASENAME".{aux,log,out,pdf}
    else
        echo "  FAIL: $BASENAME.tex does not compile. Inspect $OUTDIR/$BASENAME.log" >&2
        exit 1
    fi
else
    echo "  NOTE: pdflatex not found; skipping the compile check."
fi
