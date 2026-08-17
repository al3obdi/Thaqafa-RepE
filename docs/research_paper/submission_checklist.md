# ACL/EMNLP 2026 Submission Checklist

This checklist must be completed before submitting the paper for double-blind
review. Tick each item and verify before submission.

## Paper Structure

- [ ] Abstract is 150–200 words
- [ ] Introduction clearly states the 3 main contributions
- [ ] Methodology section covers extraction, steering, and probing
- [ ] Experimental Setup reports models, dataset, metrics, and baselines
- [ ] Results section covers all four research questions
- [ ] Limitations section is comprehensive and honest
- [ ] Conclusion restates findings and future work
- [ ] Ethics Statement addresses cultural essentialism and dual-use
- [ ] Reproducibility Statement includes commit hash and compute details

## Anonymization (Double-Blind)

- [ ] Author names and affiliations removed
- [ ] GitHub URLs replaced with "Anonymous (code released upon acceptance)"
- [ ] ORCID links removed
- [ ] No acknowledgments mentioning specific grants or names
- [ ] No identifying information in file metadata (PDF author field)
- [ ] `scripts/anonymize_paper.sh` run and verified
- [ ] Verification script reports PASS (no identifying patterns found)

## Figures and Tables

- [ ] All figures have descriptive captions
- [ ] Figure references in text use `\ref{}`
- [ ] Tables use `booktabs` style (no vertical lines)
- [ ] Layer sweep figure included (or placeholder marked)
- [ ] Steering sweep figure included (or placeholder marked)
- [ | Baseline comparison table included
- [ ] All numbers in tables are from actual experiments (not \todo markers)

## References

- [ ] All citations in text have corresponding BibTeX entries
- [ ] All BibTeX entries verified against published versions
- [ ] No broken `\citep{}` or `\citet{}` references
- [ ] Bibliography style matches target conference (plainnat for base, acl_natbib for ACL)

## Reproducibility

- [ ] Code repository commit hash recorded in Reproducibility Statement
- [ ] Random seeds documented
- [ ] Hardware and compute time reported
- [ ] `pyproject.toml` lockfile or `requirements.txt` included
- [ ] `scripts/generate_paper_results.py` command documented in paper
- [ ] Dataset file (`data/datasets/cultural_concepts.jsonl`) included in repo

## Ethics

- [ ] Consent and compensation for native-speaker annotators described
- [ ] Cultural essentialism risk discussed (§Limitations + Ethics Statement)
- [ ] Dual-use concern addressed (suppression is as easy as amplification)
- [ ] Decision to label contested concepts as `mixed` is explained
- [ ] No personally identifiable information in the dataset

## Final Build

- [ ] `cd docs/research_paper && ./build.sh` completes without errors
- [ ] PDF generated successfully
- [ ] Page count within limit (8 pages for long, 4 for short — check CFP)
- [ ] No overfull/underfull hbox warnings in critical sections
- [ ] All `\todo` markers resolved or silenced

## Supplementary Material

- [ ] Supplementary material directory structure prepared
- [ ] Additional examples and full generation outputs included
- [ ] Anonymized in the same way as the main paper

## Pre-Submission Commands

```bash
# 1. Generate results
python scripts/generate_paper_results.py --concepts wasta_001,muruah_001,diyafa_001

# 2. Inject results into LaTeX
python scripts/inject_results_to_tex.py

# 3. Anonymize
./scripts/anonymize_paper.sh

# 4. Build
cd docs/research_paper && ./build.sh

# 5. Verify
grep -c '\\todo' main_anonymous.tex  # should be 0 or very few
```