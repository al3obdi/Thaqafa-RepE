# Pilot results: `gpt2`

- **Model**: `gpt2` on cpu (float32)
- **Commit**: `796d792911da`
- **Run (UTC)**: 2026-08-20T12:33:16+00:00
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
| `wasta_001` | 5 | 0.450 +/- 0.323 | 0.500 | -0.050 | 0.642 | 0.600 |
| `muruah_001` | 0 | 0.550 +/- 0.100 | 0.500 | +0.050 | 0.358 | 0.600 |
| `diyafa_001` | 1 | 0.633 +/- 0.360 | 0.500 | +0.133 | 0.154 | 0.600 |
| `karam_001` | 0 | 0.700 +/- 0.292 | 0.500 | +0.200 | 0.070 | 0.600 |
| `sharaf_001` | 1 | 0.800 +/- 0.187 | 0.500 | +0.300 | 0.020 | 0.600 |
| `sabr_001` | 11 | 0.683 +/- 0.244 | 0.500 | +0.183 | 0.075 | 0.600 |
| `silat_rahim_001` | 1 | 0.767 +/- 0.226 | 0.500 | +0.267 | 0.040 | 0.600 |
| `jiwar_001` | 1 | 0.567 +/- 0.355 | 0.500 | +0.067 | 0.328 | 0.600 |
| `shura_001` | 2 | 0.667 +/- 0.230 | 0.500 | +0.167 | 0.154 | 0.600 |
| `majlis_001` | 0 | 0.800 +/- 0.187 | 0.500 | +0.300 | 0.015 | 0.600 |
| `fazaa_001` | 4 | 0.800 +/- 0.187 | 0.500 | +0.300 | 0.030 | 0.600 |
| `hayaa_001` | 9 | 0.750 +/- 0.224 | 0.500 | +0.250 | 0.050 | 0.600 |

## 2. Steering: effect against cost

Strength is a fraction of the layer's mean residual norm, so the
same number means the same relative intervention at every layer.
`effect_kl` is the KL divergence between the steered and unsteered
next-token distributions; `mean_loss` is the model's own
cross-entropy on the prompts, which rises as steering damages
fluency.

| Concept | Strength | Layer | Effect (KL) | Loss |
|---|---|---|---|---|
| `wasta_001` | -0.40 | 5 | 1.4755 | 6.4236 |
| `wasta_001` | -0.20 | 5 | 0.4214 | 5.0320 |
| `wasta_001` | +0.00 | 5 | 0.0000 | 4.5341 |
| `wasta_001` | +0.20 | 5 | 0.4526 | 4.8365 |
| `wasta_001` | +0.40 | 5 | 3.4528 | 6.1037 |
| `muruah_001` | -0.40 | 0 | 0.9904 | 5.7916 |
| `muruah_001` | -0.20 | 0 | 0.0478 | 4.5812 |
| `muruah_001` | +0.00 | 0 | 0.0000 | 4.5341 |
| `muruah_001` | +0.20 | 0 | 0.0251 | 4.5275 |
| `muruah_001` | +0.40 | 0 | 0.1021 | 4.5940 |
| `diyafa_001` | -0.40 | 1 | 0.8353 | 4.9395 |
| `diyafa_001` | -0.20 | 1 | 0.0956 | 4.5270 |
| `diyafa_001` | +0.00 | 1 | 0.0000 | 4.5341 |
| `diyafa_001` | +0.20 | 1 | 0.9224 | 5.5281 |
| `diyafa_001` | +0.40 | 1 | 4.4233 | 6.8224 |
| `karam_001` | -0.40 | 0 | 1.9916 | 6.2987 |
| `karam_001` | -0.20 | 0 | 0.0221 | 4.6586 |
| `karam_001` | +0.00 | 0 | 0.0000 | 4.5341 |
| `karam_001` | +0.20 | 0 | 0.0488 | 4.4682 |
| `karam_001` | +0.40 | 0 | 0.1401 | 4.4667 |
| `sharaf_001` | -0.40 | 1 | 3.7647 | 6.6935 |
| `sharaf_001` | -0.20 | 1 | 2.7527 | 6.4646 |
| `sharaf_001` | +0.00 | 1 | 0.0000 | 4.5341 |
| `sharaf_001` | +0.20 | 1 | 0.2599 | 4.6475 |
| `sharaf_001` | +0.40 | 1 | 1.2119 | 6.7840 |
| `sabr_001` | -0.40 | 11 | 2.2704 | 7.0280 |
| `sabr_001` | -0.20 | 11 | 0.4919 | 5.2450 |
| `sabr_001` | +0.00 | 11 | 0.0000 | 4.5341 |
| `sabr_001` | +0.20 | 11 | 0.1760 | 4.6719 |
| `sabr_001` | +0.40 | 11 | 0.5357 | 5.4458 |
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
| `shura_001` | -0.40 | 2 | 4.6064 | 7.1789 |
| `shura_001` | -0.20 | 2 | 3.4853 | 6.5808 |
| `shura_001` | +0.00 | 2 | 0.0000 | 4.5341 |
| `shura_001` | +0.20 | 2 | 0.2972 | 4.5855 |
| `shura_001` | +0.40 | 2 | 2.2098 | 5.6714 |
| `majlis_001` | -0.40 | 0 | 1.7419 | 5.9857 |
| `majlis_001` | -0.20 | 0 | 0.0158 | 4.6474 |
| `majlis_001` | +0.00 | 0 | 0.0000 | 4.5341 |
| `majlis_001` | +0.20 | 0 | 0.0630 | 4.4815 |
| `majlis_001` | +0.40 | 0 | 0.1463 | 4.5659 |
| `fazaa_001` | -0.40 | 4 | 0.6722 | 5.3627 |
| `fazaa_001` | -0.20 | 4 | 0.3093 | 4.6953 |
| `fazaa_001` | +0.00 | 4 | 0.0000 | 4.5341 |
| `fazaa_001` | +0.20 | 4 | 0.5529 | 4.8353 |
| `fazaa_001` | +0.40 | 4 | 1.3093 | 6.3031 |
| `hayaa_001` | -0.40 | 9 | 0.4632 | 5.1681 |
| `hayaa_001` | -0.20 | 9 | 0.0607 | 4.5952 |
| `hayaa_001` | +0.00 | 9 | 0.0000 | 4.5341 |
| `hayaa_001` | +0.20 | 9 | 0.0756 | 4.5563 |
| `hayaa_001` | +0.40 | 9 | 0.9134 | 5.4151 |

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
| `wasta_001` | steering@+0.20 | 2.1901 | 0 |
| `muruah_001` | prompt:neutral | 1.9440 | 0 |
| `muruah_001` | prompt:direct_en | 2.3420 | 8 |
| `muruah_001` | prompt:persona_en | 2.0933 | 18 |
| `muruah_001` | steering@+0.20 | 2.1263 | 0 |
| `diyafa_001` | prompt:neutral | 1.9440 | 0 |
| `diyafa_001` | prompt:direct_en | 2.2746 | 9 |
| `diyafa_001` | prompt:persona_en | 1.8504 | 19 |
| `diyafa_001` | steering@+0.20 | 2.2401 | 0 |
| `karam_001` | prompt:neutral | 1.9440 | 0 |
| `karam_001` | prompt:direct_en | 2.0604 | 8 |
| `karam_001` | prompt:persona_en | 1.8553 | 18 |
| `karam_001` | steering@+0.20 | 2.0282 | 0 |
| `sharaf_001` | prompt:neutral | 1.9440 | 0 |
| `sharaf_001` | prompt:direct_en | 2.1128 | 8 |
| `sharaf_001` | prompt:persona_en | 1.8429 | 18 |
| `sharaf_001` | steering@+0.20 | 2.4657 | 0 |
| `sabr_001` | prompt:neutral | 1.9440 | 0 |
| `sabr_001` | prompt:direct_en | 2.1128 | 9 |
| `sabr_001` | prompt:persona_en | 1.8724 | 19 |
| `sabr_001` | steering@+0.20 | 2.2149 | 0 |
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
| `shura_001` | steering@+0.20 | 2.0088 | 0 |
| `majlis_001` | prompt:neutral | 1.9440 | 0 |
| `majlis_001` | prompt:direct_en | 2.1548 | 9 |
| `majlis_001` | prompt:persona_en | 1.8530 | 19 |
| `majlis_001` | steering@+0.20 | 1.9450 | 0 |
| `fazaa_001` | prompt:neutral | 1.9440 | 0 |
| `fazaa_001` | prompt:direct_en | 2.1698 | 10 |
| `fazaa_001` | prompt:persona_en | 2.0424 | 20 |
| `fazaa_001` | steering@+0.20 | 2.3245 | 0 |
| `hayaa_001` | prompt:neutral | 1.9440 | 0 |
| `hayaa_001` | prompt:direct_en | 2.0604 | 10 |
| `hayaa_001` | prompt:persona_en | 1.9737 | 20 |
| `hayaa_001` | steering@+0.20 | 2.1793 | 0 |

## 4. Do the Arabic and English exemplars find the same direction?

`aligned` is the cosine between a concept's Arabic-only and
English-only directions. On its own it means nothing: two
directions at the same layer can be similar because the layer
has a dominant axis. `mismatched` is the same measurement
against the *other* concepts' English directions, and
`separation` is the gap. Only the gap carries information.

| Concept | Layer | Aligned | Mismatched | Separation |
|---|---|---|---|---|
| `diyafa_001` | 1 | +0.154 | -0.146 | +0.300 |
| `karam_001` | 1 | +0.651 | +0.459 | +0.192 |
| `majlis_001` | 1 | +0.574 | +0.422 | +0.153 |
| `sharaf_001` | 1 | +0.637 | +0.488 | +0.149 |
| `fazaa_001` | 1 | +0.563 | +0.441 | +0.122 |
| `hayaa_001` | 1 | +0.485 | +0.388 | +0.097 |
| `silat_rahim_001` | 1 | +0.359 | +0.274 | +0.086 |
| `wasta_001` | 1 | -0.372 | -0.440 | +0.068 |
| `sabr_001` | 1 | +0.400 | +0.353 | +0.047 |
| `jiwar_001` | 1 | -0.032 | -0.053 | +0.021 |
| `muruah_001` | 1 | +0.063 | +0.067 | -0.004 |
| `shura_001` | 1 | +0.332 | +0.495 | -0.163 |

## 5. Does steering write what the probe reads?

The direction is injected one block below the layer where the
sweep found the concept most readable, and that layer's probe
is then run on neutral prompts - the same prompts the concept
was contrasted against. `steered` is the share it calls
positive; `random` is the same share under matched-norm random
directions injected at the same layer.

`Probe` is the reading probe's own cross-validated balanced
accuracy. **A lift measured through a probe near 0.5 should be
discarded, not explained**: a probe at chance still has a
decision boundary, and pushing activations across an arbitrary
hyperplane produces a lift that means nothing.

**`lift` is the only column that carries information.** KL
divergence and fluency loss are magnitudes that any large
perturbation produces, and so is a rise in the probe's
positive rate. What a random direction cannot produce is a
rise the *concept's own* probe recognises beyond it.

Rates saturate at high strength: push hard enough and every
prompt reads positive under any direction, which shows up as
the lift shrinking back toward zero. Every strength is
reported rather than one being chosen.

| Concept | Inject | Read | Probe | Strength | Base | Steered | Random | Lift |
|---|---|---|---|---|---|---|---|---|
| `wasta_001` | 4 | 5 | 0.45 | 0.40 | 0.54 | 1.00 | 0.19 | +0.81 |
| `wasta_001` | 4 | 5 | 0.45 | 0.20 | 0.54 | 1.00 | 0.42 | +0.58 |
| `wasta_001` | 4 | 5 | 0.45 | 0.10 | 0.54 | 1.00 | 0.49 | +0.51 |
| `diyafa_001` | 0 | 1 | 0.63 | 0.40 | 0.29 | 1.00 | 0.04 | +0.96 |
| `diyafa_001` | 0 | 1 | 0.63 | 0.20 | 0.29 | 1.00 | 0.12 | +0.88 |
| `diyafa_001` | 0 | 1 | 0.63 | 0.10 | 0.29 | 1.00 | 0.24 | +0.76 |
| `sharaf_001` | 0 | 1 | 0.80 | 0.10 | 0.17 | 0.96 | 0.32 | +0.64 |
| `sharaf_001` | 0 | 1 | 0.80 | 0.20 | 0.17 | 1.00 | 0.36 | +0.64 |
| `sharaf_001` | 0 | 1 | 0.80 | 0.40 | 0.17 | 1.00 | 0.42 | +0.58 |
| `sabr_001` | 10 | 11 | 0.68 | 0.40 | 0.54 | 1.00 | 0.26 | +0.74 |
| `sabr_001` | 10 | 11 | 0.68 | 0.20 | 0.54 | 1.00 | 0.47 | +0.53 |
| `sabr_001` | 10 | 11 | 0.68 | 0.10 | 0.54 | 1.00 | 0.51 | +0.49 |
| `silat_rahim_001` | 0 | 1 | 0.77 | 0.40 | 0.17 | 1.00 | 0.04 | +0.96 |
| `silat_rahim_001` | 0 | 1 | 0.77 | 0.20 | 0.17 | 1.00 | 0.06 | +0.94 |
| `silat_rahim_001` | 0 | 1 | 0.77 | 0.10 | 0.17 | 1.00 | 0.07 | +0.93 |
| `jiwar_001` | 0 | 1 | 0.57 | 0.20 | 0.38 | 1.00 | 0.25 | +0.75 |
| `jiwar_001` | 0 | 1 | 0.57 | 0.10 | 0.38 | 1.00 | 0.31 | +0.69 |
| `jiwar_001` | 0 | 1 | 0.57 | 0.40 | 0.38 | 1.00 | 0.32 | +0.68 |
| `shura_001` | 1 | 2 | 0.67 | 0.10 | 0.33 | 1.00 | 0.32 | +0.68 |
| `shura_001` | 1 | 2 | 0.67 | 0.20 | 0.33 | 1.00 | 0.32 | +0.68 |
| `shura_001` | 1 | 2 | 0.67 | 0.40 | 0.33 | 1.00 | 0.33 | +0.67 |
| `fazaa_001` | 3 | 4 | 0.80 | 0.40 | 0.33 | 1.00 | 0.53 | +0.47 |
| `fazaa_001` | 3 | 4 | 0.80 | 0.10 | 0.33 | 0.96 | 0.54 | +0.42 |
| `fazaa_001` | 3 | 4 | 0.80 | 0.20 | 0.33 | 1.00 | 0.60 | +0.40 |
| `hayaa_001` | 8 | 9 | 0.75 | 0.10 | 0.17 | 1.00 | 0.29 | +0.71 |
| `hayaa_001` | 8 | 9 | 0.75 | 0.20 | 0.17 | 1.00 | 0.42 | +0.58 |
| `hayaa_001` | 8 | 9 | 0.75 | 0.40 | 0.17 | 1.00 | 0.51 | +0.49 |

## Limitations

- Small exemplar sets mean wide confidence intervals; no claim here
  is statistically established. Concretely: two random seeds have
  produced balanced accuracies 0.25 apart for the same concept at
  the same layer. The seed is in `manifest.json`, and a rerun under
  a different one will not reproduce these numbers exactly.
- The reported layer was selected on the same data as the p-value
  beside it, and nothing corrects for having probed every layer of
  every concept. A confirmatory result would need the layer fixed
  in advance, or a correction, or concepts held out.
- A small p-value says the labelling is unlikely to be unrelated to
  the activations. It does not say the probe found the concept
  rather than a word the exemplars happen to share.
- The read-back shows a written direction reaching the probe that
  reads it, across one transformer block. Both sides come from the
  same twelve exemplars, so it is a consistency check on the
  method, not evidence that the direction is the cultural concept
  a person would name, and not evidence that it survives the whole
  stack.
- Most of the concept entries are still awaiting native-speaker
  review (`review_status` in the dataset).
- A model with little Arabic capability can only validate that the
  pipeline measures what it claims to; it cannot support a claim
  about Arab cultural concepts.
