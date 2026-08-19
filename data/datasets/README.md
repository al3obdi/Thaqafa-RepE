# Cultural concepts dataset

`cultural_concepts.jsonl` - one JSON object per line, one Arab cultural
concept per object. The schema is documented in the repository README
(field table) and enforced by `src/data/dataset_builder.py`.

## License

The **dataset** (this directory) is released under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), separately
from the repository's MIT code license. Attribute "Thaqafa-RepE
Contributors" and share adaptations under the same terms.

## Provenance and review

- Entries are authored by contributors and reviewed by native speakers;
  the review requirement is part of [CONTRIBUTING.md](../../CONTRIBUTING.md).
- Every entry carries a `review_status` field: `"reviewed"` (native-speaker
  approved) or `"pending_native_review"` (drafted - by anyone, human or
  machine - and awaiting that approval). Results built on unreviewed entries
  must say so. Machine-drafted entries are always submitted as
  `pending_native_review`; only a native speaker may flip the flag.
- Contested concepts (e.g. *wasta*) carry `sentiment: "mixed"` by policy.
  Flattening a contested practice into `positive`/`negative` is treated as
  a data bug.
- Dialect coverage is currently uneven and pan-Arab phrasing is preferred;
  region-specific concepts must say so in `cultural_context`.

## Contrastive minimal pairs

Each entry may carry `contrast_ar`/`contrast_en`: sentences structurally
close to the exemplars but with the concept absent (for *diyafa*'s
"he hosted him generously for three days", a minimal pair is "he booked him
a hotel room and sent the address"). Extraction and probing prefer these
over the generic neutral bank, because a minimal pair cancels topic and
register, not merely "being an ordinary sentence". Write contrasts that
stay in the exemplar's world: same setting, concept removed - not its
moral opposite, which would extract "virtue vs vice" rather than the
concept itself.

## Known limitations

Twelve concepts with six exemplars and four contrasts per language: enough
for pilot probes, still far from statistical sufficiency. The Phase 5
target remains 100+ concepts, all `reviewed`, with dialect coverage beyond
MSA and Gulf. Nine of the twelve current entries are machine-drafted and
marked `pending_native_review`.
