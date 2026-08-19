# Pilot results: `gpt2`

- **Model**: `gpt2` on cpu (float32)
- **Commit**: `85544c7600cd`
- **Run (UTC)**: 2026-08-19T21:01:58+00:00
- **Seed**: 42
- **Dataset SHA-256**: `52a52e5705f38566...`

Regenerate with:

```bash
python scripts/run_pilot.py --model gpt2 --output-dir results/pilot_gpt2 --seed 42
```

## 1. Where each concept is linearly readable

Cross-validated logistic-regression accuracy on residual activations,
against the majority-class floor. `+/-` is the standard deviation across
folds; on this many exemplars it is wide, so treat the ranking as a
direction to investigate rather than a result.

| Concept | Best layer | Accuracy | Chance | Lift |
|---|---|---|---|---|
| `wasta_001` | 4 | 0.500 +/- 0.000 | 0.600 | -0.100 |
| `muruah_001` | 5 | 0.700 +/- 0.245 | 0.600 | +0.100 |
| `diyafa_001` | 11 | 0.850 +/- 0.200 | 0.600 | +0.250 |
| `karam_001` | 0 | 0.800 +/- 0.187 | 0.600 | +0.200 |
| `sharaf_001` | 2 | 0.750 +/- 0.224 | 0.600 | +0.150 |
| `sabr_001` | 0 | 0.650 +/- 0.122 | 0.600 | +0.050 |
| `silat_rahim_001` | 1 | 0.800 +/- 0.245 | 0.600 | +0.200 |
| `jiwar_001` | 1 | 0.750 +/- 0.224 | 0.600 | +0.150 |
| `shura_001` | 5 | 0.750 +/- 0.224 | 0.600 | +0.150 |
| `majlis_001` | 1 | 0.700 +/- 0.187 | 0.600 | +0.100 |
| `fazaa_001` | 0 | 0.750 +/- 0.158 | 0.600 | +0.150 |
| `hayaa_001` | 2 | 0.900 +/- 0.122 | 0.600 | +0.300 |

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
| `karam_001` | -0.40 | 0 | 1.9916 | 6.2987 |
| `karam_001` | -0.20 | 0 | 0.0221 | 4.6586 |
| `karam_001` | +0.00 | 0 | 0.0000 | 4.5341 |
| `karam_001` | +0.20 | 0 | 0.0488 | 4.4682 |
| `karam_001` | +0.40 | 0 | 0.1401 | 4.4667 |
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
| `silat_rahim_001` | -0.40 | 1 | 3.2564 | 6.7271 |
| `silat_rahim_001` | -0.20 | 1 | 2.7604 | 6.5511 |
| `silat_rahim_001` | +0.00 | 1 | 0.0000 | 4.5341 |
| `silat_rahim_001` | +0.20 | 1 | 0.2883 | 4.6298 |
| `silat_rahim_001` | +0.40 | 1 | 1.9996 | 6.9771 |
| `jiwar_001` | -0.40 | 1 | 3.9074 | 6.8641 |
| `jiwar_001` | -0.20 | 1 | 1.4036 | 5.9342 |
| `jiwar_001` | +0.00 | 1 | 0.0000 | 4.5341 |
| `jiwar_001` | +0.20 | 1 | 0.1827 | 4.5288 |
| `jiwar_001` | +0.40 | 1 | 1.2588 | 6.5689 |
| `shura_001` | -0.40 | 5 | 0.4639 | 5.2469 |
| `shura_001` | -0.20 | 5 | 0.2020 | 4.6326 |
| `shura_001` | +0.00 | 5 | 0.0000 | 4.5341 |
| `shura_001` | +0.20 | 5 | 0.5330 | 4.9633 |
| `shura_001` | +0.40 | 5 | 4.0592 | 6.8057 |
| `majlis_001` | -0.40 | 1 | 3.5335 | 6.5916 |
| `majlis_001` | -0.20 | 1 | 3.1136 | 6.5671 |
| `majlis_001` | +0.00 | 1 | 0.0000 | 4.5341 |
| `majlis_001` | +0.20 | 1 | 0.3763 | 4.6639 |
| `majlis_001` | +0.40 | 1 | 1.4691 | 7.0850 |
| `fazaa_001` | -0.40 | 0 | 0.3329 | 4.8350 |
| `fazaa_001` | -0.20 | 0 | 0.0315 | 4.5367 |
| `fazaa_001` | +0.00 | 0 | 0.0000 | 4.5341 |
| `fazaa_001` | +0.20 | 0 | 0.0462 | 4.5568 |
| `fazaa_001` | +0.40 | 0 | 0.1304 | 4.6334 |
| `hayaa_001` | -0.40 | 2 | 4.8623 | 7.2055 |
| `hayaa_001` | -0.20 | 2 | 3.9183 | 6.6568 |
| `hayaa_001` | +0.00 | 2 | 0.0000 | 4.5341 |
| `hayaa_001` | +0.20 | 2 | 0.2509 | 4.6856 |
| `hayaa_001` | +0.40 | 2 | 1.2619 | 5.7582 |

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
| `karam_001` | steering@+0.20 | 2.0282 | 0 |
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
| `silat_rahim_001` | steering@+0.20 | 2.0180 | 0 |
| `jiwar_001` | prompt:neutral | 1.9440 | 0 |
| `jiwar_001` | prompt:direct_en | 2.0752 | 9 |
| `jiwar_001` | prompt:persona_en | 1.9236 | 19 |
| `jiwar_001` | steering@+0.20 | 2.0706 | 0 |
| `shura_001` | prompt:neutral | 1.9440 | 0 |
| `shura_001` | prompt:direct_en | 2.1584 | 8 |
| `shura_001` | prompt:persona_en | 2.0393 | 18 |
| `shura_001` | steering@+0.20 | 2.0790 | 0 |
| `majlis_001` | prompt:neutral | 1.9440 | 0 |
| `majlis_001` | prompt:direct_en | 2.1548 | 9 |
| `majlis_001` | prompt:persona_en | 1.8530 | 19 |
| `majlis_001` | steering@+0.20 | 1.9980 | 0 |
| `fazaa_001` | prompt:neutral | 1.9440 | 0 |
| `fazaa_001` | prompt:direct_en | 2.1698 | 10 |
| `fazaa_001` | prompt:persona_en | 2.0424 | 20 |
| `fazaa_001` | steering@+0.20 | 1.9453 | 0 |
| `hayaa_001` | prompt:neutral | 1.9440 | 0 |
| `hayaa_001` | prompt:direct_en | 2.0604 | 10 |
| `hayaa_001` | prompt:persona_en | 1.9737 | 20 |
| `hayaa_001` | steering@+0.20 | 2.1555 | 0 |

## Limitations

- Small exemplar sets mean wide confidence intervals; no claim here
  is statistically established.
- The exemplar and contrast sets are not the same size, so the
  majority-class floor sits above 0.5. Compare each accuracy to the
  `Chance` column in the table above, never to 0.5.
- Most of the concept entries are still awaiting native-speaker
  review (`review_status` in the dataset).
- A model with little Arabic capability can only validate that the
  pipeline measures what it claims to; it cannot support a claim
  about Arab cultural concepts.
