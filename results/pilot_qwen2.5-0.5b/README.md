# Pilot results: `Qwen/Qwen2.5-0.5B`

- **Model**: `Qwen/Qwen2.5-0.5B` on cpu (float32)
- **Commit**: `f5bc90a2a521`
- **Run (UTC)**: 2026-08-20T05:57:09+00:00
- **Seed**: 42
- **Dataset SHA-256**: `52a52e5705f38566...`

Regenerate with:

```bash
python scripts/run_pilot.py --model Qwen/Qwen2.5-0.5B --output-dir results/pilot_qwen2.5-0.5b --seed 42
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
| `wasta_001` | 14 | 0.817 +/- 0.097 | 0.500 | +0.317 | 0.040 | 0.600 |
| `muruah_001` | 6 | 1.000 +/- 0.000 | 0.500 | +0.500 | <0.005 | 0.600 |
| `diyafa_001` | 5 | 1.000 +/- 0.000 | 0.500 | +0.500 | <0.005 | 0.600 |
| `karam_001` | 6 | 1.000 +/- 0.000 | 0.500 | +0.500 | <0.005 | 0.600 |
| `sharaf_001` | 8 | 1.000 +/- 0.000 | 0.500 | +0.500 | <0.005 | 0.600 |
| `sabr_001` | 14 | 1.000 +/- 0.000 | 0.500 | +0.500 | <0.005 | 0.600 |
| `silat_rahim_001` | 14 | 0.867 +/- 0.194 | 0.500 | +0.367 | 0.015 | 0.600 |
| `jiwar_001` | 9 | 0.950 +/- 0.100 | 0.500 | +0.450 | <0.005 | 0.600 |
| `shura_001` | 8 | 1.000 +/- 0.000 | 0.500 | +0.500 | <0.005 | 0.600 |
| `majlis_001` | 3 | 0.783 +/- 0.194 | 0.500 | +0.283 | 0.060 | 0.600 |
| `fazaa_001` | 8 | 1.000 +/- 0.000 | 0.500 | +0.500 | <0.005 | 0.600 |
| `hayaa_001` | 12 | 1.000 +/- 0.000 | 0.500 | +0.500 | <0.005 | 0.600 |

## 2. Steering: effect against cost

Strength is a fraction of the layer's mean residual norm, so the
same number means the same relative intervention at every layer.
`effect_kl` is the KL divergence between the steered and unsteered
next-token distributions; `mean_loss` is the model's own
cross-entropy on the prompts, which rises as steering damages
fluency.

| Concept | Strength | Layer | Effect (KL) | Loss |
|---|---|---|---|---|
| `wasta_001` | -0.40 | 14 | 4.9820 | 8.8911 |
| `wasta_001` | -0.20 | 14 | 1.3006 | 6.1092 |
| `wasta_001` | +0.00 | 14 | 0.0000 | 4.1370 |
| `wasta_001` | +0.20 | 14 | 1.4907 | 5.2175 |
| `wasta_001` | +0.40 | 14 | 8.1747 | 11.3444 |
| `muruah_001` | -0.40 | 6 | 9.7874 | 9.5147 |
| `muruah_001` | -0.20 | 6 | 6.5798 | 8.1390 |
| `muruah_001` | +0.00 | 6 | 0.0000 | 4.1370 |
| `muruah_001` | +0.20 | 6 | 1.1143 | 4.4143 |
| `muruah_001` | +0.40 | 6 | 1.8875 | 6.6170 |
| `diyafa_001` | -0.40 | 5 | 6.0697 | 8.4204 |
| `diyafa_001` | -0.20 | 5 | 1.5033 | 4.8396 |
| `diyafa_001` | +0.00 | 5 | 0.0000 | 4.1370 |
| `diyafa_001` | +0.20 | 5 | 7.9725 | 9.3106 |
| `diyafa_001` | +0.40 | 5 | 9.3283 | 10.4123 |
| `karam_001` | -0.40 | 6 | 10.5795 | 9.5982 |
| `karam_001` | -0.20 | 6 | 6.5289 | 8.1813 |
| `karam_001` | +0.00 | 6 | 0.0000 | 4.1370 |
| `karam_001` | +0.20 | 6 | 1.1804 | 4.3812 |
| `karam_001` | +0.40 | 6 | 1.9623 | 6.5871 |
| `sharaf_001` | -0.40 | 8 | 9.7568 | 9.5619 |
| `sharaf_001` | -0.20 | 8 | 5.3480 | 7.7512 |
| `sharaf_001` | +0.00 | 8 | 0.0000 | 4.1370 |
| `sharaf_001` | +0.20 | 8 | 1.2416 | 4.8251 |
| `sharaf_001` | +0.40 | 8 | 4.4201 | 7.2651 |
| `sabr_001` | -0.40 | 14 | 11.3722 | 10.8160 |
| `sabr_001` | -0.20 | 14 | 1.2348 | 5.3496 |
| `sabr_001` | +0.00 | 14 | 0.0000 | 4.1370 |
| `sabr_001` | +0.20 | 14 | 0.9383 | 4.8109 |
| `sabr_001` | +0.40 | 14 | 3.1474 | 9.0992 |
| `silat_rahim_001` | -0.40 | 14 | 10.8837 | 10.0251 |
| `silat_rahim_001` | -0.20 | 14 | 1.2620 | 5.2574 |
| `silat_rahim_001` | +0.00 | 14 | 0.0000 | 4.1370 |
| `silat_rahim_001` | +0.20 | 14 | 0.8660 | 4.8509 |
| `silat_rahim_001` | +0.40 | 14 | 2.9273 | 8.8808 |
| `jiwar_001` | -0.40 | 9 | 10.2269 | 9.8661 |
| `jiwar_001` | -0.20 | 9 | 3.3502 | 7.6776 |
| `jiwar_001` | +0.00 | 9 | 0.0000 | 4.1370 |
| `jiwar_001` | +0.20 | 9 | 1.1610 | 4.4833 |
| `jiwar_001` | +0.40 | 9 | 4.4574 | 7.7736 |
| `shura_001` | -0.40 | 8 | 9.9971 | 8.7393 |
| `shura_001` | -0.20 | 8 | 4.7889 | 7.4212 |
| `shura_001` | +0.00 | 8 | 0.0000 | 4.1370 |
| `shura_001` | +0.20 | 8 | 1.3299 | 4.7982 |
| `shura_001` | +0.40 | 8 | 3.3885 | 7.4591 |
| `majlis_001` | -0.40 | 3 | 10.5111 | 11.9416 |
| `majlis_001` | -0.20 | 3 | 10.9921 | 10.3139 |
| `majlis_001` | +0.00 | 3 | 0.0000 | 4.1370 |
| `majlis_001` | +0.20 | 3 | 2.6094 | 5.1976 |
| `majlis_001` | +0.40 | 3 | 7.4865 | 6.9080 |
| `fazaa_001` | -0.40 | 8 | 9.8984 | 8.9153 |
| `fazaa_001` | -0.20 | 8 | 5.1127 | 7.6281 |
| `fazaa_001` | +0.00 | 8 | 0.0000 | 4.1370 |
| `fazaa_001` | +0.20 | 8 | 1.2422 | 4.8232 |
| `fazaa_001` | +0.40 | 8 | 4.5400 | 7.3558 |
| `hayaa_001` | -0.40 | 12 | 11.3104 | 9.9393 |
| `hayaa_001` | -0.20 | 12 | 2.2629 | 6.0198 |
| `hayaa_001` | +0.00 | 12 | 0.0000 | 4.1370 |
| `hayaa_001` | +0.20 | 12 | 0.9913 | 4.7074 |
| `hayaa_001` | +0.40 | 12 | 4.1358 | 8.8534 |

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
| `wasta_001` | prompt:neutral | 1.9006 | 0 |
| `wasta_001` | prompt:direct_en | 1.8997 | 8 |
| `wasta_001` | prompt:persona_en | 2.1752 | 18 |
| `wasta_001` | steering@+0.20 | 2.4225 | 0 |
| `muruah_001` | prompt:neutral | 1.9006 | 0 |
| `muruah_001` | prompt:direct_en | 1.7537 | 8 |
| `muruah_001` | prompt:persona_en | 2.2384 | 18 |
| `muruah_001` | steering@+0.20 | 1.7248 | 0 |
| `diyafa_001` | prompt:neutral | 1.9006 | 0 |
| `diyafa_001` | prompt:direct_en | 2.0455 | 9 |
| `diyafa_001` | prompt:persona_en | 2.1245 | 19 |
| `diyafa_001` | steering@+0.20 | 2.4708 | 0 |
| `karam_001` | prompt:neutral | 1.9006 | 0 |
| `karam_001` | prompt:direct_en | 2.0504 | 8 |
| `karam_001` | prompt:persona_en | 1.9186 | 18 |
| `karam_001` | steering@+0.20 | 2.0144 | 0 |
| `sharaf_001` | prompt:neutral | 1.9006 | 0 |
| `sharaf_001` | prompt:direct_en | 2.2255 | 8 |
| `sharaf_001` | prompt:persona_en | 2.1974 | 18 |
| `sharaf_001` | steering@+0.20 | 2.1249 | 0 |
| `sabr_001` | prompt:neutral | 1.9006 | 0 |
| `sabr_001` | prompt:direct_en | 2.1150 | 9 |
| `sabr_001` | prompt:persona_en | 2.1411 | 19 |
| `sabr_001` | steering@+0.20 | 2.1628 | 0 |
| `silat_rahim_001` | prompt:neutral | 1.9006 | 0 |
| `silat_rahim_001` | prompt:direct_en | 2.2116 | 10 |
| `silat_rahim_001` | prompt:persona_en | 3.0209 | 20 |
| `silat_rahim_001` | steering@+0.20 | 2.3366 | 0 |
| `jiwar_001` | prompt:neutral | 1.9006 | 0 |
| `jiwar_001` | prompt:direct_en | 2.1883 | 9 |
| `jiwar_001` | prompt:persona_en | 2.2321 | 19 |
| `jiwar_001` | steering@+0.20 | 1.8032 | 0 |
| `shura_001` | prompt:neutral | 1.9006 | 0 |
| `shura_001` | prompt:direct_en | 1.7885 | 8 |
| `shura_001` | prompt:persona_en | 2.1537 | 18 |
| `shura_001` | steering@+0.20 | 1.7545 | 0 |
| `majlis_001` | prompt:neutral | 1.9006 | 0 |
| `majlis_001` | prompt:direct_en | 1.8438 | 9 |
| `majlis_001` | prompt:persona_en | 2.3030 | 19 |
| `majlis_001` | steering@+0.20 | 2.2581 | 0 |
| `fazaa_001` | prompt:neutral | 1.9006 | 0 |
| `fazaa_001` | prompt:direct_en | 2.1693 | 10 |
| `fazaa_001` | prompt:persona_en | 2.4208 | 20 |
| `fazaa_001` | steering@+0.20 | 1.9091 | 0 |
| `hayaa_001` | prompt:neutral | 1.9006 | 0 |
| `hayaa_001` | prompt:direct_en | 2.0941 | 10 |
| `hayaa_001` | prompt:persona_en | 2.3554 | 20 |
| `hayaa_001` | steering@+0.20 | 2.5934 | 0 |

## 4. Do the Arabic and English exemplars find the same direction?

`aligned` is the cosine between a concept's Arabic-only and
English-only directions. On its own it means nothing: two
directions at the same layer can be similar because the layer
has a dominant axis. `mismatched` is the same measurement
against the *other* concepts' English directions, and
`separation` is the gap. Only the gap carries information.

| Concept | Layer | Aligned | Mismatched | Separation |
|---|---|---|---|---|
| `diyafa_001` | 8 | +0.513 | -0.379 | +0.892 |
| `muruah_001` | 8 | +0.995 | +0.807 | +0.188 |
| `karam_001` | 8 | +0.996 | +0.808 | +0.188 |
| `sharaf_001` | 8 | +0.988 | +0.803 | +0.186 |
| `fazaa_001` | 8 | +0.993 | +0.807 | +0.186 |
| `majlis_001` | 8 | +0.988 | +0.803 | +0.185 |
| `hayaa_001` | 8 | +0.990 | +0.807 | +0.184 |
| `jiwar_001` | 8 | +0.961 | +0.779 | +0.182 |
| `sabr_001` | 8 | +0.989 | +0.808 | +0.182 |
| `silat_rahim_001` | 8 | +0.945 | +0.772 | +0.173 |
| `shura_001` | 8 | +0.965 | +0.797 | +0.168 |
| `wasta_001` | 8 | -0.900 | -0.807 | -0.093 |

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
