# Research paper

Scaffold for the Thaqafa-RepE paper. Every section is present with an outline of
what belongs in it; the numbers are not.

## Building

```bash
./build.sh            # pdflatex + bibtex + two more passes -> main.pdf
./build.sh --xe       # xelatex, needed to typeset Arabic script
./build.sh --clean    # remove .aux/.bbl/.log and friends
```

Requires a TeX distribution. On Debian or Ubuntu the minimal set is:

```bash
sudo apt-get install texlive-latex-base texlive-latex-recommended \
                     texlive-bibtex-extra
```

Add `texlive-xetex` and a font with Arabic coverage for `--xe`.

## Placeholders

Unwritten content is marked `\todo{...}` and renders in red. `build.sh` prints
how many remain. Before submitting, either resolve them all or silence the
macro by uncommenting the `\renewcommand{\todo}[1]{}` line near the top of
`main.tex` — but note that silencing it hides gaps rather than filling them.

## Switching to a conference template

The preamble is deliberately minimal so the scaffold compiles on a bare TeX
install; it is not a submission-ready style. Section structure, labels and
citations carry over to either template unchanged.

**ACL.** Download `acl.sty` and `acl_natbib.bst` from the ACL style files
repository into this directory, then replace the preamble with:

```latex
\documentclass[11pt]{article}
\usepackage[review]{acl}   % drop the [review] option for the camera-ready
```

and set `\bibliographystyle{acl_natbib}`. ACL requires the Limitations section
(already present) and an Ethics Statement (already stubbed).

**NeurIPS.** Drop in `neurips_2025.sty` and use
`\usepackage[preprint]{neurips_2025}`. NeurIPS wants the checklist appendix,
which this scaffold does not include.

## Arabic script

Body text transliterates Arabic terms via `\arabterm{transliteration}{gloss}`,
which is what `pdflatex` renders. To typeset the Arabic script itself, build
with `--xe` and add `polyglossia` plus an Arabic-capable font to the preamble.

## Files

| File | Purpose |
| --- | --- |
| `main.tex` | The paper. |
| `references.bib` | Bibliography. Verify every field against the published version before submission. |
| `build.sh` | Four-pass build script. |
| `figures/` | Generated plots. Create it when the first figure exists. |

Build artefacts (`main.pdf`, `*.aux`, `*.bbl`, ...) are git-ignored: the PDF is
regenerated from source, so committing it only produces merge conflicts.

## Filling in Results

The paper contains `\todo{...}` markers wherever real experimental numbers go.
`scripts/run_pilot.py` produces every artefact those numbers come from,
alongside a `manifest.json` recording what produced them.

### Workflow

1. **Run the experiment script**:

   ```bash
   python scripts/run_pilot.py --model gpt2 --output-dir results/pilot_gpt2 --seed 42
   ```

2. **Open the results summary**:

   ```bash
   cat results/pilot_gpt2/README.md
   ```

   This file contains:
   - Best layers per concept (table)
   - Steering sweep: effect KL vs fluency loss (table)
   - Steering effect against fluency cost, per strength
   - Steering against prompt-engineering baselines
   - Arabic/English direction agreement, against a mismatched control
   - The causal read-back and suppression checks, each against a
     matched-norm random control

3. **Fill in `\todo` markers in `main.tex`**:

   - Take the numbers from that run's CSVs, not from a transcription of
     them, and cite the run directory so a reader can find the manifest
   - Replace `\todo{}` entries in the results tables with those numbers
   - Carry the caveat next to each number: the reported layer was chosen on
     the same data as its p-value, and a lift measured through a probe near
     chance means nothing

4. **Rebuild**:

   ```bash
   cd docs/research_paper && ./build.sh
   ```

   Check that the `\todo` count printed by `build.sh` has decreased.
