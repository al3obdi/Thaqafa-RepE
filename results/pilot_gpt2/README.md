# Pilot results: `gpt2`

- **Model**: `gpt2` on cpu (float32)
- **Commit**: `4784e46d48ac`
- **Run (UTC)**: 2026-08-20T05:27:26+00:00
- **Seed**: 42
- **Dataset SHA-256**: `52a52e5705f38566...`

Regenerate with:

```bash
python scripts/run_pilot.py --model gpt2 --output-dir results/pilot_gpt2 --seed 42
```

## 1. Where each concept is linearly readable

Cross-validated **balanced accuracy** - the mean of the per-class
recalls - for a logistic regression on residual activations. Balanced
rather than raw because the exemplar and contrast sets are different
sizes: raw accuracy would hand a probe that learned nothing the class
ratio (`Majority` below) for free, while balanced accuracy discounts
that strategy to 0.5 whatever the ratio. `+/-` is the standard
deviation across folds; on this many exemplars it is wide, so treat the
ranking as a direction to investigate rather than a result.

`p` is a permutation p-value: the labels were shuffled
200 times and the whole cross-validation rerun, and `p`
is the share of shufflings that scored at least as well. It answers
"could this have come from a labelling unrelated to the
activations?" - not "is the probe reading the concept", which a
keyword shared by the exemplars would also satisfy.

**The layer in this table was chosen by the same data the p-value is
computed on.** Every layer was probed and the best one kept, so these
p-values are optimistic, and no correction is applied for having done
that twelve times over. Read the table as a ranking of where to look,
not as a set of independent hypothesis tests. `layer_sweep.csv` holds
every layer, so the selection can be redone.

| Concept | Best layer | Balanced acc. | Chance | Lift | p | Majority |
|---|---|---|---|---|---|---|
| `wasta_001` | 4 | 0.500 +/- 0.105 | 0.500 | +0.000 | 0.542 | 0.600 |
| `muruah_001` | 5 | 0.700 +/- 0.245 | 0.500 | +0.200 | 0.124 | 0.600 |
| `diyafa_001` | 11 | 0.850 +/- 0.200 | 0.500 | +0.350 | 0.015 | 0.600 |
| `karam_001` | 4 | 0.800 +/- 0.187 | 0.500 | +0.300 | 0.020 | 0.600 |
| `sharaf_001` | 2 | 0.800 +/- 0.187 | 0.500 | +0.300 | 0.030 | 0.600 |
| `sabr_001` | 0 | 0.633 +/- 0.187 | 0.500 | +0.133 | 0.219 | 0.600 |
| `silat_rahim_001` | 2 | 0.833 +/- 0.183 | 0.500 | +0.333 | 0.015 | 0.600 |
| `jiwar_001` | 2 | 0.667 +/- 0.211 | 0.500 | +0.167 | 0.204 | 0.600 |
| `shura_001` | 0 | 0.833 +/- 0.091 | 0.500 | +0.333 | 0.015 | 0.600 |
| `majlis_001` | 1 | 0.683 +/- 0.238 | 0.500 | +0.183 | 0.134 | 0.600 |
| `fazaa_001` | 2 | 0.767 +/- 0.226 | 0.500 | +0.267 | 0.065 | 0.600 |
| `hayaa_001` | 0 | 0.850 +/- 0.122 | 0.500 | +0.350 | 0.015 | 0.600 |

## 2. Steering: effect against cost

Strength is a fraction of the layer's mean residual norm, so the
same number means the same relative intervention at every layer.
`effect_kl` is the KL divergence between the steered and unsteered
next-token distributions; `mean_loss` is the model's own
cross-entropy on the prompts, which rises as steering damages
fluency.

| Concept | Strength | Layer | Effect (KL) | Loss |
|---|---|---|---|---|
| `wasta_001` | -0.40 | 4 | 2.6524 | 6.4336 |
| `wasta_001` | -0.20 | 4 | 0.6837 | 5.1078 |
| `wasta_001` | +0.00 | 4 | 0.0000 | 4.5341 |
| `wasta_001` | +0.20 | 4 | 0.7236 | 5.0244 |
| `wasta_001` | +0.40 | 4 | 3.4659 | 6.5206 |
| `muruah_001` | -0.40 | 5 | 0.3929 | 5.1749 |
| `muruah_001` | -0.20 | 5 | 0.1699 | 4.5579 |
| `muruah_001` | +0.00 | 5 | 0.0000 | 4.5341 |
| `muruah_001` | +0.20 | 5 | 0.8009 | 5.0608 |
| `muruah_001` | +0.40 | 5 | 4.9386 | 6.9172 |
| `diyafa_001` | -0.40 | 11 | 1.4355 | 5.7075 |
| `diyafa_001` | -0.20 | 11 | 0.2286 | 4.7447 |
| `diyafa_001` | +0.00 | 11 | 0.0000 | 4.5341 |
| `diyafa_001` | +0.20 | 11 | 0.0908 | 4.9066 |
| `diyafa_001` | +0.40 | 11 | 0.3262 | 5.6804 |
| `karam_001` | -0.40 | 4 | 0.5907 | 5.3414 |
| `karam_001` | -0.20 | 4 | 0.2511 | 4.6763 |
| `karam_001` | +0.00 | 4 | 0.0000 | 4.5341 |
| `karam_001` | +0.20 | 4 | 0.5902 | 4.7718 |
| `karam_001` | +0.40 | 4 | 1.5241 | 6.2720 |
| `sharaf_001` | -0.40 | 2 | 4.8321 | 7.1639 |
| `sharaf_001` | -0.20 | 2 | 3.7466 | 6.5806 |
| `sharaf_001` | +0.00 | 2 | 0.0000 | 4.5341 |
| `sharaf_001` | +0.20 | 2 | 0.2076 | 4.5837 |
| `sharaf_001` | +0.40 | 2 | 1.4792 | 5.4082 |
| `sabr_001` | -0.40 | 0 | 1.4294 | 4.9082 |
| `sabr_001` | -0.20 | 0 | 0.0200 | 4.6115 |
| `sabr_001` | +0.00 | 0 | 0.0000 | 4.5341 |
| `sabr_001` | +0.20 | 0 | 0.0740 | 4.5072 |
| `sabr_001` | +0.40 | 0 | 0.2055 | 4.5835 |
| `silat_rahim_001` | -0.40 | 2 | 4.5882 | 7.2186 |
| `silat_rahim_001` | -0.20 | 2 | 3.6427 | 6.6770 |
| `silat_rahim_001` | +0.00 | 2 | 0.0000 | 4.5341 |
| `silat_rahim_001` | +0.20 | 2 | 0.2266 | 4.6271 |
| `silat_rahim_001` | +0.40 | 2 | 1.4808 | 5.4727 |
| `jiwar_001` | -0.40 | 2 | 4.9310 | 7.2251 |
| `jiwar_001` | -0.20 | 2 | 3.8365 | 6.6369 |
| `jiwar_001` | +0.00 | 2 | 0.0000 | 4.5341 |
| `jiwar_001` | +0.20 | 2 | 0.2355 | 4.5438 |
| `jiwar_001` | +0.40 | 2 | 1.4674 | 5.2196 |
| `shura_001` | -0.40 | 0 | 0.1532 | 4.5517 |
| `shura_001` | -0.20 | 0 | 0.0300 | 4.5101 |
| `shura_001` | +0.00 | 0 | 0.0000 | 4.5341 |
| `shura_001` | +0.20 | 0 | 0.0339 | 4.5704 |
| `shura_001` | +0.40 | 0 | 0.1953 | 4.6192 |
| `majlis_001` | -0.40 | 1 | 3.5335 | 6.5916 |
| `majlis_001` | -0.20 | 1 | 3.1136 | 6.5671 |
| `majlis_001` | +0.00 | 1 | 0.0000 | 4.5341 |
| `majlis_001` | +0.20 | 1 | 0.3763 | 4.6639 |
| `majlis_001` | +0.40 | 1 | 1.4691 | 7.0850 |
| `fazaa_001` | -0.40 | 2 | 4.7823 | 7.1799 |
| `fazaa_001` | -0.20 | 2 | 3.8058 | 6.6436 |
| `fazaa_001` | +0.00 | 2 | 0.0000 | 4.5341 |
| `fazaa_001` | +0.20 | 2 | 0.2193 | 4.6570 |
| `fazaa_001` | +0.40 | 2 | 1.6156 | 5.7845 |
| `hayaa_001` | -0.40 | 0 | 1.2928 | 5.6411 |
| `hayaa_001` | -0.20 | 0 | 0.0260 | 4.4739 |
| `hayaa_001` | +0.00 | 0 | 0.0000 | 4.5341 |
| `hayaa_001` | +0.20 | 0 | 0.0353 | 4.5867 |
| `hayaa_001` | +0.40 | 0 | 0.1670 | 4.6304 |

## 3. Steering against prompting

`mean_continuation_loss` is scored by the *unmodified* model, so
it measures damage, not cultural grounding. Which condition is
more culturally appropriate is not decided here, and cannot be:
that judgement needs native-speaker raters. Every condition's
text is kept under `generations/` so it can be inspected, but at
this model scale the continuations are largely degenerate
repetition and are not yet worth putting in front of raters.

| Concept | Condition | Cont. loss | Extra input tokens |
|---|---|---|---|
| `wasta_001` | prompt:neutral | 1.9440 | 0 |
| `wasta_001` | prompt:direct_en | 1.9562 | 8 |
| `wasta_001` | prompt:persona_en | 1.8919 | 18 |
| `wasta_001` | steering@+0.20 | 1.5444 | 0 |
| `muruah_001` | prompt:neutral | 1.9440 | 0 |
| `muruah_001` | prompt:direct_en | 2.3420 | 8 |
| `muruah_001` | prompt:persona_en | 2.0933 | 18 |
| `muruah_001` | steering@+0.20 | 2.2067 | 0 |
| `diyafa_001` | prompt:neutral | 1.9440 | 0 |
| `diyafa_001` | prompt:direct_en | 2.2746 | 9 |
| `diyafa_001` | prompt:persona_en | 1.8504 | 19 |
| `diyafa_001` | steering@+0.20 | 1.8750 | 0 |
| `karam_001` | prompt:neutral | 1.9440 | 0 |
| `karam_001` | prompt:direct_en | 2.0604 | 8 |
| `karam_001` | prompt:persona_en | 1.8553 | 18 |
| `karam_001` | steering@+0.20 | 2.2187 | 0 |
| `sharaf_001` | prompt:neutral | 1.9440 | 0 |
| `sharaf_001` | prompt:direct_en | 2.1128 | 8 |
| `sharaf_001` | prompt:persona_en | 1.8429 | 18 |
| `sharaf_001` | steering@+0.20 | 2.1555 | 0 |
| `sabr_001` | prompt:neutral | 1.9440 | 0 |
| `sabr_001` | prompt:direct_en | 2.1128 | 9 |
| `sabr_001` | prompt:persona_en | 1.8724 | 19 |
| `sabr_001` | steering@+0.20 | 1.9833 | 0 |
| `silat_rahim_001` | prompt:neutral | 1.9440 | 0 |
| `silat_rahim_001` | prompt:direct_en | 2.1128 | 10 |
| `silat_rahim_001` | prompt:persona_en | 1.9196 | 20 |
| `silat_rahim_001` | steering@+0.20 | 2.1562 | 0 |
| `jiwar_001` | prompt:neutral | 1.9440 | 0 |
| `jiwar_001` | prompt:direct_en | 2.0752 | 9 |
| `jiwar_001` | prompt:persona_en | 1.9236 | 19 |
| `jiwar_001` | steering@+0.20 | 2.2024 | 0 |
| `shura_001` | prompt:neutral | 1.9440 | 0 |
| `shura_001` | prompt:direct_en | 2.1584 | 8 |
| `shura_001` | prompt:persona_en | 2.0393 | 18 |
| `shura_001` | steering@+0.20 | 1.9798 | 0 |
| `majlis_001` | prompt:neutral | 1.9440 | 0 |
| `majlis_001` | prompt:direct_en | 2.1548 | 9 |
| `majlis_001` | prompt:persona_en | 1.8530 | 19 |
| `majlis_001` | steering@+0.20 | 1.9980 | 0 |
| `fazaa_001` | prompt:neutral | 1.9440 | 0 |
| `fazaa_001` | prompt:direct_en | 2.1698 | 10 |
| `fazaa_001` | prompt:persona_en | 2.0424 | 20 |
| `fazaa_001` | steering@+0.20 | 2.1555 | 0 |
| `hayaa_001` | prompt:neutral | 1.9440 | 0 |
| `hayaa_001` | prompt:direct_en | 2.0604 | 10 |
| `hayaa_001` | prompt:persona_en | 1.9737 | 20 |
| `hayaa_001` | steering@+0.20 | 1.9751 | 0 |

## 4. Do the Arabic and English exemplars find the same direction?

`aligned` is the cosine between a concept's Arabic-only and
English-only directions. On its own it means nothing: two
directions at the same layer can be similar because the layer
has a dominant axis. `mismatched` is the same measurement
against the *other* concepts' English directions, and
`separation` is the gap. Only the gap carries information.

| Concept | Layer | Aligned | Mismatched | Separation |
|---|---|---|---|---|
| `diyafa_001` | 2 | +0.395 | -0.419 | +0.814 |
| `karam_001` | 2 | +0.955 | +0.767 | +0.188 |
| `majlis_001` | 2 | +0.928 | +0.748 | +0.180 |
| `sharaf_001` | 2 | +0.950 | +0.771 | +0.179 |
| `fazaa_001` | 2 | +0.922 | +0.750 | +0.172 |
| `hayaa_001` | 2 | +0.899 | +0.732 | +0.167 |
| `silat_rahim_001` | 2 | +0.828 | +0.671 | +0.157 |
| `sabr_001` | 2 | +0.860 | +0.713 | +0.148 |
| `shura_001` | 2 | +0.843 | +0.772 | +0.071 |
| `muruah_001` | 2 | +0.229 | +0.197 | +0.032 |
| `jiwar_001` | 2 | -0.294 | -0.252 | -0.042 |
| `wasta_001` | 2 | -0.859 | -0.755 | -0.104 |

## Limitations

- Small exemplar sets mean wide confidence intervals; no claim here
  is statistically established.
- The reported layer was selected on the same data as the p-value
  beside it, and nothing corrects for having probed every layer of
  every concept. A confirmatory result would need the layer fixed
  in advance, or a correction, or concepts held out.
- A small p-value says the labelling is unlikely to be unrelated to
  the activations. It does not say the probe found the concept
  rather than a word the exemplars happen to share.
- Most of the concept entries are still awaiting native-speaker
  review (`review_status` in the dataset).
- A model with little Arabic capability can only validate that the
  pipeline measures what it claims to; it cannot support a claim
  about Arab cultural concepts.
