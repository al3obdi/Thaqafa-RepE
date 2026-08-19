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

- Entries are authored by contributors and reviewed by native speakers
  before merging; the review requirement is part of
  [CONTRIBUTING.md](../../CONTRIBUTING.md).
- Contested concepts (e.g. *wasta*) carry `sentiment: "mixed"` by policy.
  Flattening a contested practice into `positive`/`negative` is treated as
  a data bug.
- Dialect coverage is currently uneven and pan-Arab phrasing is preferred;
  region-specific concepts must say so in `cultural_context`.

## Known limitations

Three seed concepts with two exemplars each: enough to exercise the
pipeline, far too small for statistical claims. The Phase 5 target is 100+
concepts with at least three exemplars per language and curated
minimal-pair negatives.
