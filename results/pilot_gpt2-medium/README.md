# Pilot results: `gpt2-medium`

- **Model**: `gpt2-medium` on cpu (float32)
- **Commit**: `b41bdefe9443`
- **Run (UTC)**: 2026-08-21T15:05:46+00:00
- **Seed**: 42
- **Dataset SHA-256**: `52a52e5705f38566...`

Regenerate with:

```bash
python scripts/run_pilot.py --model gpt2-medium --output-dir results/pilot_gpt2-medium --seed 42
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
| `wasta_001` | 18 | 0.483 +/- 0.200 | 0.500 | -0.017 | 0.557 | 0.600 |
| `muruah_001` | 0 | 0.817 +/- 0.186 | 0.500 | +0.317 | 0.025 | 0.600 |
| `diyafa_001` | 18 | 0.800 +/- 0.187 | 0.500 | +0.300 | 0.010 | 0.600 |
| `karam_001` | 3 | 0.650 +/- 0.255 | 0.500 | +0.150 | 0.124 | 0.600 |
| `sharaf_001` | 10 | 0.767 +/- 0.162 | 0.500 | +0.267 | 0.040 | 0.600 |
| `sabr_001` | 1 | 0.750 +/- 0.274 | 0.500 | +0.250 | 0.020 | 0.600 |
| `silat_rahim_001` | 2 | 0.700 +/- 0.113 | 0.500 | +0.200 | 0.095 | 0.600 |
| `jiwar_001` | 0 | 0.600 +/- 0.339 | 0.500 | +0.100 | 0.224 | 0.600 |
| `shura_001` | 21 | 0.600 +/- 0.255 | 0.500 | +0.100 | 0.274 | 0.600 |
| `majlis_001` | 2 | 0.767 +/- 0.226 | 0.500 | +0.267 | 0.035 | 0.600 |
| `fazaa_001` | 3 | 0.767 +/- 0.226 | 0.500 | +0.267 | 0.035 | 0.600 |
| `hayaa_001` | 11 | 0.767 +/- 0.162 | 0.500 | +0.267 | 0.020 | 0.600 |

## 2. Steering: effect against cost

Strength is a fraction of the layer's mean residual norm, so the
same number means the same relative intervention at every layer.
`effect_kl` is the KL divergence between the steered and unsteered
next-token distributions; `mean_loss` is the model's own
cross-entropy on the prompts, which rises as steering damages
fluency.

| Concept | Strength | Layer | Effect (KL) | Loss |
|---|---|---|---|---|
| `wasta_001` | -0.40 | 18 | 0.3911 | 5.5373 |
| `wasta_001` | -0.20 | 18 | 0.1123 | 4.6043 |
| `wasta_001` | +0.00 | 18 | 0.0000 | 4.1849 |
| `wasta_001` | +0.20 | 18 | 0.1101 | 4.3587 |
| `wasta_001` | +0.40 | 18 | 0.3740 | 5.1418 |
| `muruah_001` | -0.40 | 0 | 0.1936 | 4.3291 |
| `muruah_001` | -0.20 | 0 | 0.0131 | 4.1571 |
| `muruah_001` | +0.00 | 0 | 0.0000 | 4.1849 |
| `muruah_001` | +0.20 | 0 | 0.0383 | 4.1828 |
| `muruah_001` | +0.40 | 0 | 1.4799 | 5.0560 |
| `diyafa_001` | -0.40 | 18 | 0.3764 | 4.9916 |
| `diyafa_001` | -0.20 | 18 | 0.0974 | 4.4501 |
| `diyafa_001` | +0.00 | 18 | 0.0000 | 4.1849 |
| `diyafa_001` | +0.20 | 18 | 0.0715 | 4.3939 |
| `diyafa_001` | +0.40 | 18 | 0.2485 | 5.0628 |
| `karam_001` | -0.40 | 3 | 0.3199 | 4.2564 |
| `karam_001` | -0.20 | 3 | 0.0225 | 4.1756 |
| `karam_001` | +0.00 | 3 | 0.0000 | 4.1849 |
| `karam_001` | +0.20 | 3 | 0.0370 | 4.1443 |
| `karam_001` | +0.40 | 3 | 0.1123 | 4.1392 |
| `sharaf_001` | -0.40 | 10 | 0.3052 | 4.4209 |
| `sharaf_001` | -0.20 | 10 | 0.0983 | 4.1258 |
| `sharaf_001` | +0.00 | 10 | 0.0000 | 4.1849 |
| `sharaf_001` | +0.20 | 10 | 0.0565 | 4.4248 |
| `sharaf_001` | +0.40 | 10 | 0.6553 | 5.1549 |
| `sabr_001` | -0.40 | 1 | 0.4341 | 4.6756 |
| `sabr_001` | -0.20 | 1 | 0.0258 | 4.2933 |
| `sabr_001` | +0.00 | 1 | 0.0000 | 4.1849 |
| `sabr_001` | +0.20 | 1 | 0.0247 | 4.1438 |
| `sabr_001` | +0.40 | 1 | 0.5838 | 4.5837 |
| `silat_rahim_001` | -0.40 | 2 | 1.7395 | 6.1602 |
| `silat_rahim_001` | -0.20 | 2 | 0.4524 | 4.3764 |
| `silat_rahim_001` | +0.00 | 2 | 0.0000 | 4.1849 |
| `silat_rahim_001` | +0.20 | 2 | 0.0791 | 4.2402 |
| `silat_rahim_001` | +0.40 | 2 | 1.7264 | 5.0427 |
| `jiwar_001` | -0.40 | 0 | 6.8888 | 6.6067 |
| `jiwar_001` | -0.20 | 0 | 0.0353 | 4.2650 |
| `jiwar_001` | +0.00 | 0 | 0.0000 | 4.1849 |
| `jiwar_001` | +0.20 | 0 | 0.0787 | 4.1145 |
| `jiwar_001` | +0.40 | 0 | 2.4219 | 6.3774 |
| `shura_001` | -0.40 | 21 | 0.2303 | 5.6372 |
| `shura_001` | -0.20 | 21 | 0.0656 | 4.7577 |
| `shura_001` | +0.00 | 21 | 0.0000 | 4.1849 |
| `shura_001` | +0.20 | 21 | 0.0656 | 4.0173 |
| `shura_001` | +0.40 | 21 | 0.2318 | 4.2194 |
| `majlis_001` | -0.40 | 2 | 1.5592 | 5.8837 |
| `majlis_001` | -0.20 | 2 | 0.9653 | 4.7757 |
| `majlis_001` | +0.00 | 2 | 0.0000 | 4.1849 |
| `majlis_001` | +0.20 | 2 | 0.0784 | 4.2560 |
| `majlis_001` | +0.40 | 2 | 1.7856 | 5.4811 |
| `fazaa_001` | -0.40 | 3 | 0.3286 | 4.2588 |
| `fazaa_001` | -0.20 | 3 | 0.0262 | 4.1858 |
| `fazaa_001` | +0.00 | 3 | 0.0000 | 4.1849 |
| `fazaa_001` | +0.20 | 3 | 0.0383 | 4.1243 |
| `fazaa_001` | +0.40 | 3 | 0.1348 | 4.1647 |
| `hayaa_001` | -0.40 | 11 | 0.2927 | 4.4022 |
| `hayaa_001` | -0.20 | 11 | 0.0852 | 4.1412 |
| `hayaa_001` | +0.00 | 11 | 0.0000 | 4.1849 |
| `hayaa_001` | +0.20 | 11 | 0.0986 | 4.3953 |
| `hayaa_001` | +0.40 | 11 | 0.6229 | 5.2181 |

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
| `wasta_001` | prompt:neutral | 2.1332 | 0 |
| `wasta_001` | prompt:direct_en | 2.0955 | 8 |
| `wasta_001` | prompt:persona_en | 2.2803 | 18 |
| `wasta_001` | steering@+0.20 | 1.7614 | 0 |
| `muruah_001` | prompt:neutral | 2.1332 | 0 |
| `muruah_001` | prompt:direct_en | 2.3908 | 8 |
| `muruah_001` | prompt:persona_en | 1.9815 | 18 |
| `muruah_001` | steering@+0.20 | 1.9748 | 0 |
| `diyafa_001` | prompt:neutral | 2.1332 | 0 |
| `diyafa_001` | prompt:direct_en | 2.1496 | 9 |
| `diyafa_001` | prompt:persona_en | 1.9421 | 19 |
| `diyafa_001` | steering@+0.20 | 1.9564 | 0 |
| `karam_001` | prompt:neutral | 2.1332 | 0 |
| `karam_001` | prompt:direct_en | 2.1463 | 8 |
| `karam_001` | prompt:persona_en | 1.8587 | 18 |
| `karam_001` | steering@+0.20 | 2.0378 | 0 |
| `sharaf_001` | prompt:neutral | 2.1332 | 0 |
| `sharaf_001` | prompt:direct_en | 2.3126 | 8 |
| `sharaf_001` | prompt:persona_en | 2.1860 | 18 |
| `sharaf_001` | steering@+0.20 | 1.8952 | 0 |
| `sabr_001` | prompt:neutral | 2.1332 | 0 |
| `sabr_001` | prompt:direct_en | 2.2633 | 9 |
| `sabr_001` | prompt:persona_en | 2.0626 | 19 |
| `sabr_001` | steering@+0.20 | 1.9193 | 0 |
| `silat_rahim_001` | prompt:neutral | 2.1332 | 0 |
| `silat_rahim_001` | prompt:direct_en | 2.2290 | 10 |
| `silat_rahim_001` | prompt:persona_en | 2.2464 | 20 |
| `silat_rahim_001` | steering@+0.20 | 1.9714 | 0 |
| `jiwar_001` | prompt:neutral | 2.1332 | 0 |
| `jiwar_001` | prompt:direct_en | 2.1370 | 9 |
| `jiwar_001` | prompt:persona_en | 2.0472 | 19 |
| `jiwar_001` | steering@+0.20 | 2.1240 | 0 |
| `shura_001` | prompt:neutral | 2.1332 | 0 |
| `shura_001` | prompt:direct_en | 1.9413 | 8 |
| `shura_001` | prompt:persona_en | 2.0417 | 18 |
| `shura_001` | steering@+0.20 | 2.1865 | 0 |
| `majlis_001` | prompt:neutral | 2.1332 | 0 |
| `majlis_001` | prompt:direct_en | 2.0836 | 9 |
| `majlis_001` | prompt:persona_en | 2.0778 | 19 |
| `majlis_001` | steering@+0.20 | 1.8092 | 0 |
| `fazaa_001` | prompt:neutral | 2.1332 | 0 |
| `fazaa_001` | prompt:direct_en | 2.1573 | 10 |
| `fazaa_001` | prompt:persona_en | 2.0792 | 20 |
| `fazaa_001` | steering@+0.20 | 2.1834 | 0 |
| `hayaa_001` | prompt:neutral | 2.1332 | 0 |
| `hayaa_001` | prompt:direct_en | 2.1628 | 10 |
| `hayaa_001` | prompt:persona_en | 2.1753 | 20 |
| `hayaa_001` | steering@+0.20 | 2.1901 | 0 |

## 4. Do the Arabic and English exemplars find the same direction?

`aligned` is the cosine between a concept's Arabic-only and
English-only directions. On its own it means nothing: two
directions at the same layer can be similar because the layer
has a dominant axis. `mismatched` is the same measurement
against the *other* concepts' English directions, and
`separation` is the gap. Only the gap carries information.

| Concept | Layer | Aligned | Mismatched | Separation |
|---|---|---|---|---|
| `diyafa_001` | 3 | +0.286 | -0.276 | +0.562 |
| `karam_001` | 3 | +0.934 | +0.750 | +0.184 |
| `majlis_001` | 3 | +0.897 | +0.721 | +0.177 |
| `sharaf_001` | 3 | +0.918 | +0.748 | +0.170 |
| `fazaa_001` | 3 | +0.892 | +0.729 | +0.163 |
| `silat_rahim_001` | 3 | +0.827 | +0.669 | +0.158 |
| `hayaa_001` | 3 | +0.840 | +0.691 | +0.149 |
| `sabr_001` | 3 | +0.827 | +0.687 | +0.140 |
| `shura_001` | 3 | +0.792 | +0.741 | +0.051 |
| `muruah_001` | 3 | +0.186 | +0.157 | +0.029 |
| `jiwar_001` | 3 | -0.485 | -0.412 | -0.073 |
| `wasta_001` | 3 | -0.816 | -0.717 | -0.099 |

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
| `wasta_001` | 17 | 18 | 0.48 | 0.05 | 0.58 | 1.00 | 0.62 | +0.38 |
| `wasta_001` | 17 | 18 | 0.48 | 0.10 | 0.58 | 1.00 | 0.69 | +0.31 |
| `wasta_001` | 17 | 18 | 0.48 | 0.02 | 0.58 | 0.88 | 0.62 | +0.25 |
| `wasta_001` | 17 | 18 | 0.48 | 0.20 | 0.58 | 1.00 | 0.76 | +0.24 |
| `diyafa_001` | 17 | 18 | 0.80 | 0.20 | 0.42 | 1.00 | 0.28 | +0.72 |
| `diyafa_001` | 17 | 18 | 0.80 | 0.10 | 0.42 | 1.00 | 0.36 | +0.64 |
| `diyafa_001` | 17 | 18 | 0.80 | 0.05 | 0.42 | 1.00 | 0.42 | +0.58 |
| `diyafa_001` | 17 | 18 | 0.80 | 0.02 | 0.42 | 0.92 | 0.42 | +0.50 |
| `karam_001` | 2 | 3 | 0.65 | 0.20 | 0.04 | 1.00 | 0.11 | +0.89 |
| `karam_001` | 2 | 3 | 0.65 | 0.10 | 0.04 | 1.00 | 0.12 | +0.88 |
| `karam_001` | 2 | 3 | 0.65 | 0.05 | 0.04 | 0.71 | 0.07 | +0.64 |
| `karam_001` | 2 | 3 | 0.65 | 0.02 | 0.04 | 0.42 | 0.04 | +0.38 |
| `sharaf_001` | 9 | 10 | 0.77 | 0.10 | 0.25 | 1.00 | 0.46 | +0.54 |
| `sharaf_001` | 9 | 10 | 0.77 | 0.05 | 0.25 | 0.75 | 0.36 | +0.39 |
| `sharaf_001` | 9 | 10 | 0.77 | 0.20 | 0.25 | 1.00 | 0.61 | +0.39 |
| `sharaf_001` | 9 | 10 | 0.77 | 0.02 | 0.25 | 0.50 | 0.29 | +0.21 |
| `sabr_001` | 0 | 1 | 0.75 | 0.05 | 0.50 | 1.00 | 0.51 | +0.49 |
| `sabr_001` | 0 | 1 | 0.75 | 0.10 | 0.50 | 1.00 | 0.54 | +0.46 |
| `sabr_001` | 0 | 1 | 0.75 | 0.20 | 0.50 | 1.00 | 0.58 | +0.42 |
| `sabr_001` | 0 | 1 | 0.75 | 0.02 | 0.50 | 0.92 | 0.51 | +0.40 |
| `silat_rahim_001` | 1 | 2 | 0.70 | 0.05 | 0.21 | 1.00 | 0.21 | +0.79 |
| `silat_rahim_001` | 1 | 2 | 0.70 | 0.10 | 0.21 | 1.00 | 0.21 | +0.79 |
| `silat_rahim_001` | 1 | 2 | 0.70 | 0.20 | 0.21 | 1.00 | 0.22 | +0.78 |
| `silat_rahim_001` | 1 | 2 | 0.70 | 0.02 | 0.21 | 0.79 | 0.21 | +0.58 |
| `shura_001` | 20 | 21 | 0.60 | 0.02 | 0.46 | 0.96 | 0.50 | +0.46 |
| `shura_001` | 20 | 21 | 0.60 | 0.05 | 0.46 | 1.00 | 0.56 | +0.44 |
| `shura_001` | 20 | 21 | 0.60 | 0.10 | 0.46 | 1.00 | 0.62 | +0.38 |
| `shura_001` | 20 | 21 | 0.60 | 0.20 | 0.46 | 1.00 | 0.71 | +0.29 |
| `majlis_001` | 1 | 2 | 0.77 | 0.20 | 0.38 | 1.00 | 0.32 | +0.68 |
| `majlis_001` | 1 | 2 | 0.77 | 0.10 | 0.38 | 1.00 | 0.36 | +0.64 |
| `majlis_001` | 1 | 2 | 0.77 | 0.05 | 0.38 | 0.96 | 0.36 | +0.60 |
| `majlis_001` | 1 | 2 | 0.77 | 0.02 | 0.38 | 0.71 | 0.38 | +0.33 |
| `fazaa_001` | 2 | 3 | 0.77 | 0.20 | 0.42 | 1.00 | 0.32 | +0.68 |
| `fazaa_001` | 2 | 3 | 0.77 | 0.10 | 0.42 | 1.00 | 0.36 | +0.64 |
| `fazaa_001` | 2 | 3 | 0.77 | 0.05 | 0.42 | 1.00 | 0.38 | +0.62 |
| `fazaa_001` | 2 | 3 | 0.77 | 0.02 | 0.42 | 0.96 | 0.42 | +0.54 |
| `hayaa_001` | 10 | 11 | 0.77 | 0.10 | 0.29 | 1.00 | 0.24 | +0.76 |
| `hayaa_001` | 10 | 11 | 0.77 | 0.20 | 0.29 | 1.00 | 0.25 | +0.75 |
| `hayaa_001` | 10 | 11 | 0.77 | 0.05 | 0.29 | 0.96 | 0.24 | +0.72 |
| `hayaa_001` | 10 | 11 | 0.77 | 0.02 | 0.29 | 0.58 | 0.28 | +0.31 |

## 6. Does subtracting the concept remove it?

The mirror of the section above, and the claim representation
engineering is most often reached for and least often checked.
The direction is *subtracted* at the same layer, and the probe
is run on the concept's own exemplars - held out fold by fold,
because a probe trained on an exemplar recognises it whatever
is injected.

`base` is how often the probes recognise exemplars they never
saw; it caps how far suppression could possibly push. `Probe`
is those probes' held-out balanced accuracy, which has to be
read first: a probe answering "positive" to everything would
reach a baseline of 1.00 too, and anything that unsettled it
would look like removal.

**`drop` is the column that carries information.** Subtracting
a large enough vector damages the representation whatever its
direction, and a probe stops recognising damaged activations;
only the part a random direction of the same norm fails to
reproduce is evidence about the concept.

A steered rate of 0.00 says the probe's decision was flipped on
every held-out exemplar. That is not the same claim as the
concept having been removed from the model: a linear probe
flips once the shift along its normal exceeds its margin, and
the shift here is a fixed fraction of the residual norm. The
gap to the random arm shows the flip is specific to this
direction, not that nothing else changed.

| Concept | Inject | Read | Probe | Strength | Base | Steered | Random | Drop |
|---|---|---|---|---|---|---|---|---|
| `wasta_001` | 17 | 18 | 0.46 | -0.05 | 0.67 | 0.00 | 0.58 | +0.58 |
| `wasta_001` | 17 | 18 | 0.46 | -0.10 | 0.67 | 0.00 | 0.58 | +0.58 |
| `wasta_001` | 17 | 18 | 0.46 | -0.02 | 0.67 | 0.17 | 0.61 | +0.44 |
| `wasta_001` | 17 | 18 | 0.46 | -0.20 | 0.67 | 0.00 | 0.42 | +0.42 |
| `diyafa_001` | 17 | 18 | 0.75 | -0.10 | 1.00 | 0.00 | 0.97 | +0.97 |
| `diyafa_001` | 17 | 18 | 0.75 | -0.20 | 1.00 | 0.00 | 0.92 | +0.92 |
| `diyafa_001` | 17 | 18 | 0.75 | -0.05 | 1.00 | 0.08 | 0.97 | +0.89 |
| `diyafa_001` | 17 | 18 | 0.75 | -0.02 | 1.00 | 0.25 | 0.97 | +0.72 |
| `karam_001` | 2 | 3 | 0.67 | -0.10 | 0.83 | 0.00 | 0.72 | +0.72 |
| `karam_001` | 2 | 3 | 0.67 | -0.20 | 0.83 | 0.00 | 0.67 | +0.67 |
| `karam_001` | 2 | 3 | 0.67 | -0.05 | 0.83 | 0.25 | 0.83 | +0.58 |
| `karam_001` | 2 | 3 | 0.67 | -0.02 | 0.83 | 0.58 | 0.83 | +0.25 |
| `sharaf_001` | 9 | 10 | 0.77 | -0.10 | 0.92 | 0.00 | 0.69 | +0.69 |
| `sharaf_001` | 9 | 10 | 0.77 | -0.05 | 0.92 | 0.17 | 0.78 | +0.61 |
| `sharaf_001` | 9 | 10 | 0.77 | -0.20 | 0.92 | 0.00 | 0.58 | +0.58 |
| `sharaf_001` | 9 | 10 | 0.77 | -0.02 | 0.92 | 0.75 | 0.89 | +0.14 |
| `sabr_001` | 0 | 1 | 0.71 | -0.05 | 0.92 | 0.00 | 0.89 | +0.89 |
| `sabr_001` | 0 | 1 | 0.71 | -0.10 | 0.92 | 0.00 | 0.89 | +0.89 |
| `sabr_001` | 0 | 1 | 0.71 | -0.02 | 0.92 | 0.17 | 0.94 | +0.78 |
| `sabr_001` | 0 | 1 | 0.71 | -0.20 | 0.92 | 0.00 | 0.67 | +0.67 |
| `silat_rahim_001` | 1 | 2 | 0.62 | -0.20 | 0.75 | 0.00 | 0.78 | +0.78 |
| `silat_rahim_001` | 1 | 2 | 0.62 | -0.05 | 0.75 | 0.00 | 0.75 | +0.75 |
| `silat_rahim_001` | 1 | 2 | 0.62 | -0.10 | 0.75 | 0.00 | 0.75 | +0.75 |
| `silat_rahim_001` | 1 | 2 | 0.62 | -0.02 | 0.75 | 0.42 | 0.75 | +0.33 |
| `shura_001` | 20 | 21 | 0.58 | -0.05 | 0.92 | 0.00 | 0.69 | +0.69 |
| `shura_001` | 20 | 21 | 0.58 | -0.10 | 0.92 | 0.00 | 0.67 | +0.67 |
| `shura_001` | 20 | 21 | 0.58 | -0.02 | 0.92 | 0.25 | 0.83 | +0.58 |
| `shura_001` | 20 | 21 | 0.58 | -0.20 | 0.92 | 0.00 | 0.53 | +0.53 |
| `majlis_001` | 1 | 2 | 0.71 | -0.20 | 0.92 | 0.00 | 0.92 | +0.92 |
| `majlis_001` | 1 | 2 | 0.71 | -0.10 | 0.92 | 0.00 | 0.89 | +0.89 |
| `majlis_001` | 1 | 2 | 0.71 | -0.05 | 0.92 | 0.17 | 0.92 | +0.75 |
| `majlis_001` | 1 | 2 | 0.71 | -0.02 | 0.92 | 0.58 | 0.92 | +0.33 |
| `fazaa_001` | 2 | 3 | 0.71 | -0.05 | 0.92 | 0.00 | 0.94 | +0.94 |
| `fazaa_001` | 2 | 3 | 0.71 | -0.10 | 0.92 | 0.00 | 0.92 | +0.92 |
| `fazaa_001` | 2 | 3 | 0.71 | -0.20 | 0.92 | 0.00 | 0.89 | +0.89 |
| `fazaa_001` | 2 | 3 | 0.71 | -0.02 | 0.92 | 0.42 | 0.94 | +0.53 |
| `hayaa_001` | 10 | 11 | 0.71 | -0.10 | 0.92 | 0.00 | 0.89 | +0.89 |
| `hayaa_001` | 10 | 11 | 0.71 | -0.20 | 0.92 | 0.00 | 0.89 | +0.89 |
| `hayaa_001` | 10 | 11 | 0.71 | -0.05 | 0.92 | 0.25 | 0.94 | +0.69 |
| `hayaa_001` | 10 | 11 | 0.71 | -0.02 | 0.92 | 0.83 | 0.94 | +0.11 |

## 7. Does one language's direction steer the other's reader?

Section 4 asks the geometric question and answers it with a
cosine. This asks the behavioural one, which can disagree: a
modest cosine in a high-dimensional space still leaves a large
shared component, and a probe reads a projection, not an angle.

The probe and the prompts both come from `read`, so nothing in
the measurement is bilingual except the injected direction. A
rise cannot be the reader recognising the other script - it
never sees any.

`own` is the ceiling: what the reader's own language direction
achieved over the random floor. `other` is the transfer.
`ratio` is the second as a fraction of the first, so 1.00 means
the other language's direction moved this reader exactly as far
as its own did. Rates saturate, and at a saturated point both
arms sit at 1.00 and the ratio reads 1.00 for free - which is
why every strength is here.

| Concept | Read | Probe | Strength | Own lift | Transfer lift | Ratio |
|---|---|---|---|---|---|---|
| `wasta_001` | en | 0.69 | 0.02 | +0.00 | +0.00 | 0.00 |
| `wasta_001` | en | 0.69 | 0.05 | +0.00 | +0.00 | 0.00 |
| `wasta_001` | en | 0.69 | 0.10 | +0.00 | +0.00 | 0.00 |
| `wasta_001` | en | 0.69 | 0.20 | +0.00 | -0.08 | 0.00 |
| `wasta_001` | ar | 0.62 | 0.02 | +0.53 | +0.03 | 0.05 |
| `wasta_001` | ar | 0.62 | 0.10 | +0.75 | -0.00 | -0.00 |
| `wasta_001` | ar | 0.62 | 0.05 | +0.72 | -0.03 | -0.04 |
| `wasta_001` | ar | 0.62 | 0.20 | +0.75 | -0.17 | -0.22 |
| `diyafa_001` | en | 0.88 | 0.10 | +0.44 | +0.03 | 0.06 |
| `diyafa_001` | en | 0.88 | 0.20 | +0.44 | +0.03 | 0.06 |
| `diyafa_001` | en | 0.88 | 0.02 | +0.08 | +0.00 | 0.00 |
| `diyafa_001` | en | 0.88 | 0.05 | +0.42 | +0.00 | 0.00 |
| `diyafa_001` | ar | 0.75 | 0.20 | +0.64 | +0.39 | 0.61 |
| `diyafa_001` | ar | 0.75 | 0.10 | +0.69 | +0.11 | 0.16 |
| `diyafa_001` | ar | 0.75 | 0.02 | +0.67 | +0.00 | 0.00 |
| `diyafa_001` | ar | 0.75 | 0.05 | +0.67 | +0.00 | 0.00 |
| `karam_001` | en | 0.56 | 0.02 | +0.42 | +0.00 | 0.00 |
| `karam_001` | en | 0.56 | 0.05 | +0.53 | -0.06 | -0.11 |
| `karam_001` | en | 0.56 | 0.10 | +0.42 | -0.08 | -0.20 |
| `karam_001` | en | 0.56 | 0.20 | +0.22 | -0.11 | -0.50 |
| `karam_001` | ar | 0.75 | 0.05 | +0.89 | +0.06 | 0.06 |
| `karam_001` | ar | 0.75 | 0.20 | +0.89 | +0.06 | 0.06 |
| `karam_001` | ar | 0.75 | 0.10 | +0.94 | +0.03 | 0.03 |
| `karam_001` | ar | 0.75 | 0.02 | +0.42 | +0.00 | 0.00 |
| `sharaf_001` | en | 0.62 | 0.02 | +0.39 | +0.06 | 0.14 |
| `sharaf_001` | en | 0.62 | 0.10 | +0.67 | +0.08 | 0.12 |
| `sharaf_001` | en | 0.62 | 0.05 | +0.44 | +0.03 | 0.06 |
| `sharaf_001` | en | 0.62 | 0.20 | +0.56 | -0.03 | -0.05 |
| `sharaf_001` | ar | 0.62 | 0.20 | +0.61 | +0.53 | 0.86 |
| `sharaf_001` | ar | 0.62 | 0.10 | +0.67 | +0.42 | 0.62 |
| `sharaf_001` | ar | 0.62 | 0.02 | +0.47 | +0.14 | 0.29 |
| `sharaf_001` | ar | 0.62 | 0.05 | +0.64 | +0.14 | 0.22 |
| `sabr_001` | en | 0.75 | 0.02 | +0.75 | +0.00 | 0.00 |
| `sabr_001` | en | 0.75 | 0.05 | +0.89 | -0.03 | -0.03 |
| `sabr_001` | en | 0.75 | 0.10 | +0.81 | -0.11 | -0.14 |
| `sabr_001` | en | 0.75 | 0.20 | +0.64 | -0.28 | -0.43 |
| `sabr_001` | ar | 0.75 | 0.10 | +0.03 | +0.03 | 1.00 |
| `sabr_001` | ar | 0.75 | 0.20 | +0.06 | +0.06 | 1.00 |
| `sabr_001` | ar | 0.75 | 0.02 | +0.08 | +0.00 | 0.00 |
| `sabr_001` | ar | 0.75 | 0.05 | +0.06 | -0.03 | -0.50 |
| `silat_rahim_001` | en | 0.69 | 0.02 | +0.00 | +0.00 | 0.00 |
| `silat_rahim_001` | en | 0.69 | 0.05 | +1.00 | +0.00 | 0.00 |
| `silat_rahim_001` | en | 0.69 | 0.10 | +1.00 | +0.00 | 0.00 |
| `silat_rahim_001` | en | 0.69 | 0.20 | +1.00 | +0.00 | 0.00 |
| `silat_rahim_001` | ar | 0.62 | 0.20 | +0.78 | +0.44 | 0.57 |
| `silat_rahim_001` | ar | 0.62 | 0.10 | +0.86 | +0.11 | 0.13 |
| `silat_rahim_001` | ar | 0.62 | 0.05 | +0.89 | +0.06 | 0.06 |
| `silat_rahim_001` | ar | 0.62 | 0.02 | +0.67 | +0.00 | 0.00 |
| `shura_001` | en | 0.44 | 0.20 | +0.31 | +0.06 | 0.18 |
| `shura_001` | en | 0.44 | 0.10 | +0.36 | +0.03 | 0.08 |
| `shura_001` | en | 0.44 | 0.02 | +0.42 | +0.00 | 0.00 |
| `shura_001` | en | 0.44 | 0.05 | +0.42 | +0.00 | 0.00 |
| `shura_001` | ar | 0.56 | 0.02 | +0.61 | -0.06 | -0.09 |
| `shura_001` | ar | 0.56 | 0.05 | +0.53 | -0.14 | -0.26 |
| `shura_001` | ar | 0.56 | 0.10 | +0.47 | -0.28 | -0.59 |
| `shura_001` | ar | 0.56 | 0.20 | +0.36 | -0.39 | -1.08 |
| `majlis_001` | en | 0.62 | 0.20 | +0.47 | +0.31 | 0.65 |
| `majlis_001` | en | 0.62 | 0.10 | +0.56 | +0.14 | 0.25 |
| `majlis_001` | en | 0.62 | 0.02 | +0.42 | +0.00 | 0.00 |
| `majlis_001` | en | 0.62 | 0.05 | +0.58 | +0.00 | 0.00 |
| `majlis_001` | ar | 0.88 | 0.10 | +0.56 | +0.31 | 0.55 |
| `majlis_001` | ar | 0.88 | 0.20 | +0.44 | +0.19 | 0.44 |
| `majlis_001` | ar | 0.88 | 0.05 | +0.61 | +0.11 | 0.18 |
| `majlis_001` | ar | 0.88 | 0.02 | +0.42 | +0.00 | 0.00 |
| `fazaa_001` | en | 0.75 | 0.02 | +0.25 | +0.00 | 0.00 |
| `fazaa_001` | en | 0.75 | 0.05 | +0.33 | +0.00 | 0.00 |
| `fazaa_001` | en | 0.75 | 0.10 | +0.28 | -0.06 | -0.20 |
| `fazaa_001` | en | 0.75 | 0.20 | +0.22 | -0.28 | -1.25 |
| `fazaa_001` | ar | 0.62 | 0.20 | +0.64 | +0.06 | 0.09 |
| `fazaa_001` | ar | 0.62 | 0.05 | +0.61 | +0.03 | 0.05 |
| `fazaa_001` | ar | 0.62 | 0.02 | +0.50 | +0.00 | 0.00 |
| `fazaa_001` | ar | 0.62 | 0.10 | +0.58 | +0.00 | 0.00 |
| `hayaa_001` | en | 0.88 | 0.02 | +0.17 | +0.00 | 0.00 |
| `hayaa_001` | en | 0.88 | 0.05 | +0.50 | +0.00 | 0.00 |
| `hayaa_001` | en | 0.88 | 0.10 | +0.44 | -0.06 | -0.12 |
| `hayaa_001` | en | 0.88 | 0.20 | +0.36 | -0.06 | -0.15 |
| `hayaa_001` | ar | 0.88 | 0.05 | +0.89 | +0.06 | 0.06 |
| `hayaa_001` | ar | 0.88 | 0.10 | +0.89 | +0.06 | 0.06 |
| `hayaa_001` | ar | 0.88 | 0.20 | +0.89 | +0.06 | 0.06 |
| `hayaa_001` | ar | 0.88 | 0.02 | +0.67 | +0.00 | 0.00 |

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
- The transfer check reads a projection, not a meaning. It shows
  one language's direction moving the other language's probe; that
  probe was trained on twelve exemplars of the same concept, so a
  high ratio says the two directions share what that probe reads,
  not that the model holds one cultural concept across languages.
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
