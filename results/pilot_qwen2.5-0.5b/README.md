# Pilot results: `Qwen/Qwen2.5-0.5B`

- **Model**: `Qwen/Qwen2.5-0.5B` on cpu (float32)
- **Commit**: `b41bdefe9443`
- **Run (UTC)**: 2026-08-21T16:15:46+00:00
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
| `wasta_001` | 16 | 0.683 +/- 0.153 | 0.500 | +0.183 | 0.154 | 0.600 |
| `muruah_001` | 10 | 1.000 +/- 0.000 | 0.500 | +0.500 | <0.005 | 0.600 |
| `diyafa_001` | 4 | 0.867 +/- 0.194 | 0.500 | +0.367 | <0.005 | 0.600 |
| `karam_001` | 10 | 0.800 +/- 0.292 | 0.500 | +0.300 | 0.030 | 0.600 |
| `sharaf_001` | 20 | 1.000 +/- 0.000 | 0.500 | +0.500 | <0.005 | 0.600 |
| `sabr_001` | 8 | 1.000 +/- 0.000 | 0.500 | +0.500 | <0.005 | 0.600 |
| `silat_rahim_001` | 6 | 0.783 +/- 0.194 | 0.500 | +0.283 | 0.030 | 0.600 |
| `jiwar_001` | 8 | 0.750 +/- 0.224 | 0.500 | +0.250 | 0.040 | 0.600 |
| `shura_001` | 10 | 0.900 +/- 0.200 | 0.500 | +0.400 | 0.010 | 0.600 |
| `majlis_001` | 22 | 0.867 +/- 0.113 | 0.500 | +0.367 | 0.010 | 0.600 |
| `fazaa_001` | 8 | 1.000 +/- 0.000 | 0.500 | +0.500 | <0.005 | 0.600 |
| `hayaa_001` | 4 | 0.950 +/- 0.100 | 0.500 | +0.450 | <0.005 | 0.600 |

## 2. Steering: effect against cost

Strength is a fraction of the layer's mean residual norm, so the
same number means the same relative intervention at every layer.
`effect_kl` is the KL divergence between the steered and unsteered
next-token distributions; `mean_loss` is the model's own
cross-entropy on the prompts, which rises as steering damages
fluency.

| Concept | Strength | Layer | Effect (KL) | Loss |
|---|---|---|---|---|
| `wasta_001` | -0.40 | 16 | 3.9638 | 9.6078 |
| `wasta_001` | -0.20 | 16 | 1.1778 | 6.0806 |
| `wasta_001` | +0.00 | 16 | 0.0000 | 4.1370 |
| `wasta_001` | +0.20 | 16 | 1.1978 | 4.8928 |
| `wasta_001` | +0.40 | 16 | 11.8060 | 13.0333 |
| `muruah_001` | -0.40 | 10 | 11.5967 | 9.7946 |
| `muruah_001` | -0.20 | 10 | 3.0628 | 6.4388 |
| `muruah_001` | +0.00 | 10 | 0.0000 | 4.1370 |
| `muruah_001` | +0.20 | 10 | 1.0681 | 4.6755 |
| `muruah_001` | +0.40 | 10 | 4.4538 | 8.4596 |
| `diyafa_001` | -0.40 | 4 | 8.9983 | 8.9199 |
| `diyafa_001` | -0.20 | 4 | 1.5873 | 5.2662 |
| `diyafa_001` | +0.00 | 4 | 0.0000 | 4.1370 |
| `diyafa_001` | +0.20 | 4 | 9.5973 | 10.0084 |
| `diyafa_001` | +0.40 | 4 | 9.6982 | 11.6843 |
| `karam_001` | -0.40 | 10 | 11.6898 | 9.7517 |
| `karam_001` | -0.20 | 10 | 3.1685 | 6.5366 |
| `karam_001` | +0.00 | 10 | 0.0000 | 4.1370 |
| `karam_001` | +0.20 | 10 | 1.1020 | 4.5506 |
| `karam_001` | +0.40 | 10 | 4.3192 | 7.9807 |
| `sharaf_001` | -0.40 | 20 | 13.1425 | 13.9851 |
| `sharaf_001` | -0.20 | 20 | 6.3272 | 7.9832 |
| `sharaf_001` | +0.00 | 20 | 0.0000 | 4.1370 |
| `sharaf_001` | +0.20 | 20 | 2.7206 | 6.3729 |
| `sharaf_001` | +0.40 | 20 | 8.7004 | 10.6164 |
| `sabr_001` | -0.40 | 8 | 9.5669 | 9.5895 |
| `sabr_001` | -0.20 | 8 | 5.0466 | 7.5329 |
| `sabr_001` | +0.00 | 8 | 0.0000 | 4.1370 |
| `sabr_001` | +0.20 | 8 | 1.2718 | 4.7841 |
| `sabr_001` | +0.40 | 8 | 4.7915 | 7.3909 |
| `silat_rahim_001` | -0.40 | 6 | 10.3240 | 9.5102 |
| `silat_rahim_001` | -0.20 | 6 | 6.5281 | 7.8665 |
| `silat_rahim_001` | +0.00 | 6 | 0.0000 | 4.1370 |
| `silat_rahim_001` | +0.20 | 6 | 1.2841 | 4.4258 |
| `silat_rahim_001` | +0.40 | 6 | 1.9577 | 6.6653 |
| `jiwar_001` | -0.40 | 8 | 10.2698 | 9.6031 |
| `jiwar_001` | -0.20 | 8 | 4.9426 | 8.7546 |
| `jiwar_001` | +0.00 | 8 | 0.0000 | 4.1370 |
| `jiwar_001` | +0.20 | 8 | 1.3146 | 4.7724 |
| `jiwar_001` | +0.40 | 8 | 4.5775 | 7.9013 |
| `shura_001` | -0.40 | 10 | 11.3339 | 9.2997 |
| `shura_001` | -0.20 | 10 | 3.5430 | 6.8535 |
| `shura_001` | +0.00 | 10 | 0.0000 | 4.1370 |
| `shura_001` | +0.20 | 10 | 1.1552 | 4.6416 |
| `shura_001` | +0.40 | 10 | 5.3714 | 8.9048 |
| `majlis_001` | -0.40 | 22 | 0.3574 | 4.9203 |
| `majlis_001` | -0.20 | 22 | 0.0796 | 4.4325 |
| `majlis_001` | +0.00 | 22 | 0.0000 | 4.1370 |
| `majlis_001` | +0.20 | 22 | 0.0880 | 4.0320 |
| `majlis_001` | +0.40 | 22 | 0.4265 | 4.1247 |
| `fazaa_001` | -0.40 | 8 | 9.8984 | 8.9153 |
| `fazaa_001` | -0.20 | 8 | 5.1127 | 7.6281 |
| `fazaa_001` | +0.00 | 8 | 0.0000 | 4.1370 |
| `fazaa_001` | +0.20 | 8 | 1.2422 | 4.8232 |
| `fazaa_001` | +0.40 | 8 | 4.5400 | 7.3558 |
| `hayaa_001` | -0.40 | 4 | 10.1551 | 11.8453 |
| `hayaa_001` | -0.20 | 4 | 10.4660 | 10.6372 |
| `hayaa_001` | +0.00 | 4 | 0.0000 | 4.1370 |
| `hayaa_001` | +0.20 | 4 | 1.8134 | 4.8936 |
| `hayaa_001` | +0.40 | 4 | 7.3060 | 6.7860 |

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
| `wasta_001` | steering@+0.20 | 2.4699 | 0 |
| `muruah_001` | prompt:neutral | 1.9006 | 0 |
| `muruah_001` | prompt:direct_en | 1.7537 | 8 |
| `muruah_001` | prompt:persona_en | 2.2384 | 18 |
| `muruah_001` | steering@+0.20 | 2.1009 | 0 |
| `diyafa_001` | prompt:neutral | 1.9006 | 0 |
| `diyafa_001` | prompt:direct_en | 2.0455 | 9 |
| `diyafa_001` | prompt:persona_en | 2.1245 | 19 |
| `diyafa_001` | steering@+0.20 | 5.5729 | 0 |
| `karam_001` | prompt:neutral | 1.9006 | 0 |
| `karam_001` | prompt:direct_en | 2.0504 | 8 |
| `karam_001` | prompt:persona_en | 1.9186 | 18 |
| `karam_001` | steering@+0.20 | 1.8243 | 0 |
| `sharaf_001` | prompt:neutral | 1.9006 | 0 |
| `sharaf_001` | prompt:direct_en | 2.2255 | 8 |
| `sharaf_001` | prompt:persona_en | 2.1974 | 18 |
| `sharaf_001` | steering@+0.20 | 1.3329 | 0 |
| `sabr_001` | prompt:neutral | 1.9006 | 0 |
| `sabr_001` | prompt:direct_en | 2.1150 | 9 |
| `sabr_001` | prompt:persona_en | 2.1411 | 19 |
| `sabr_001` | steering@+0.20 | 2.0990 | 0 |
| `silat_rahim_001` | prompt:neutral | 1.9006 | 0 |
| `silat_rahim_001` | prompt:direct_en | 2.2116 | 10 |
| `silat_rahim_001` | prompt:persona_en | 3.0209 | 20 |
| `silat_rahim_001` | steering@+0.20 | 2.3125 | 0 |
| `jiwar_001` | prompt:neutral | 1.9006 | 0 |
| `jiwar_001` | prompt:direct_en | 2.1883 | 9 |
| `jiwar_001` | prompt:persona_en | 2.2321 | 19 |
| `jiwar_001` | steering@+0.20 | 1.9384 | 0 |
| `shura_001` | prompt:neutral | 1.9006 | 0 |
| `shura_001` | prompt:direct_en | 1.7885 | 8 |
| `shura_001` | prompt:persona_en | 2.1537 | 18 |
| `shura_001` | steering@+0.20 | 1.5831 | 0 |
| `majlis_001` | prompt:neutral | 1.9006 | 0 |
| `majlis_001` | prompt:direct_en | 1.8438 | 9 |
| `majlis_001` | prompt:persona_en | 2.3030 | 19 |
| `majlis_001` | steering@+0.20 | 1.7562 | 0 |
| `fazaa_001` | prompt:neutral | 1.9006 | 0 |
| `fazaa_001` | prompt:direct_en | 2.1693 | 10 |
| `fazaa_001` | prompt:persona_en | 2.4208 | 20 |
| `fazaa_001` | steering@+0.20 | 1.9091 | 0 |
| `hayaa_001` | prompt:neutral | 1.9006 | 0 |
| `hayaa_001` | prompt:direct_en | 2.0941 | 10 |
| `hayaa_001` | prompt:persona_en | 2.3554 | 20 |
| `hayaa_001` | steering@+0.20 | 2.3058 | 0 |

## 4. Do the Arabic and English exemplars find the same direction?

`aligned` is the cosine between a concept's Arabic-only and
English-only directions. On its own it means nothing: two
directions at the same layer can be similar because the layer
has a dominant axis. `mismatched` is the same measurement
against the *other* concepts' English directions, and
`separation` is the gap. Only the gap carries information.

| Concept | Layer | Aligned | Mismatched | Separation |
|---|---|---|---|---|
| `diyafa_001` | 10 | +0.508 | -0.361 | +0.869 |
| `muruah_001` | 10 | +0.994 | +0.805 | +0.189 |
| `karam_001` | 10 | +0.995 | +0.807 | +0.189 |
| `sharaf_001` | 10 | +0.987 | +0.800 | +0.187 |
| `fazaa_001` | 10 | +0.991 | +0.805 | +0.186 |
| `majlis_001` | 10 | +0.987 | +0.801 | +0.186 |
| `hayaa_001` | 10 | +0.989 | +0.805 | +0.184 |
| `sabr_001` | 10 | +0.988 | +0.807 | +0.182 |
| `jiwar_001` | 10 | +0.956 | +0.775 | +0.180 |
| `silat_rahim_001` | 10 | +0.944 | +0.771 | +0.173 |
| `shura_001` | 10 | +0.965 | +0.795 | +0.170 |
| `wasta_001` | 10 | -0.888 | -0.804 | -0.083 |

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
| `wasta_001` | 15 | 16 | 0.68 | 0.20 | 0.67 | 1.00 | 0.50 | +0.50 |
| `wasta_001` | 15 | 16 | 0.68 | 0.10 | 0.67 | 1.00 | 0.57 | +0.43 |
| `wasta_001` | 15 | 16 | 0.68 | 0.02 | 0.67 | 1.00 | 0.65 | +0.35 |
| `wasta_001` | 15 | 16 | 0.68 | 0.05 | 0.67 | 1.00 | 0.65 | +0.35 |
| `muruah_001` | 9 | 10 | 1.00 | 0.20 | 0.25 | 0.92 | 0.32 | +0.60 |
| `muruah_001` | 9 | 10 | 1.00 | 0.10 | 0.25 | 0.62 | 0.25 | +0.38 |
| `muruah_001` | 9 | 10 | 1.00 | 0.05 | 0.25 | 0.38 | 0.21 | +0.17 |
| `muruah_001` | 9 | 10 | 1.00 | 0.02 | 0.25 | 0.25 | 0.22 | +0.03 |
| `diyafa_001` | 3 | 4 | 0.87 | 0.20 | 0.62 | 1.00 | 0.33 | +0.67 |
| `diyafa_001` | 3 | 4 | 0.87 | 0.10 | 0.62 | 1.00 | 0.38 | +0.62 |
| `diyafa_001` | 3 | 4 | 0.87 | 0.05 | 0.62 | 1.00 | 0.56 | +0.44 |
| `diyafa_001` | 3 | 4 | 0.87 | 0.02 | 0.62 | 1.00 | 0.60 | +0.40 |
| `karam_001` | 9 | 10 | 0.80 | 0.20 | 0.17 | 0.75 | 0.36 | +0.39 |
| `karam_001` | 9 | 10 | 0.80 | 0.10 | 0.17 | 0.50 | 0.32 | +0.18 |
| `karam_001` | 9 | 10 | 0.80 | 0.05 | 0.17 | 0.38 | 0.22 | +0.15 |
| `karam_001` | 9 | 10 | 0.80 | 0.02 | 0.17 | 0.25 | 0.19 | +0.06 |
| `sharaf_001` | 19 | 20 | 1.00 | 0.10 | 0.50 | 1.00 | 0.47 | +0.53 |
| `sharaf_001` | 19 | 20 | 1.00 | 0.20 | 0.50 | 1.00 | 0.50 | +0.50 |
| `sharaf_001` | 19 | 20 | 1.00 | 0.05 | 0.50 | 0.88 | 0.49 | +0.39 |
| `sharaf_001` | 19 | 20 | 1.00 | 0.02 | 0.50 | 0.67 | 0.49 | +0.18 |
| `sabr_001` | 7 | 8 | 1.00 | 0.20 | 0.38 | 1.00 | 0.40 | +0.60 |
| `sabr_001` | 7 | 8 | 1.00 | 0.10 | 0.38 | 0.88 | 0.36 | +0.51 |
| `sabr_001` | 7 | 8 | 1.00 | 0.05 | 0.38 | 0.58 | 0.35 | +0.24 |
| `sabr_001` | 7 | 8 | 1.00 | 0.02 | 0.38 | 0.50 | 0.31 | +0.19 |
| `silat_rahim_001` | 5 | 6 | 0.78 | 0.20 | 0.25 | 1.00 | 0.17 | +0.83 |
| `silat_rahim_001` | 5 | 6 | 0.78 | 0.10 | 0.25 | 0.83 | 0.22 | +0.61 |
| `silat_rahim_001` | 5 | 6 | 0.78 | 0.05 | 0.25 | 0.62 | 0.24 | +0.39 |
| `silat_rahim_001` | 5 | 6 | 0.78 | 0.02 | 0.25 | 0.33 | 0.21 | +0.12 |
| `jiwar_001` | 7 | 8 | 0.75 | 0.20 | 0.33 | 1.00 | 0.61 | +0.39 |
| `jiwar_001` | 7 | 8 | 0.75 | 0.10 | 0.33 | 0.88 | 0.54 | +0.33 |
| `jiwar_001` | 7 | 8 | 0.75 | 0.05 | 0.33 | 0.62 | 0.46 | +0.17 |
| `jiwar_001` | 7 | 8 | 0.75 | 0.02 | 0.33 | 0.46 | 0.39 | +0.07 |
| `shura_001` | 9 | 10 | 0.90 | 0.20 | 0.62 | 1.00 | 0.32 | +0.68 |
| `shura_001` | 9 | 10 | 0.90 | 0.10 | 0.62 | 1.00 | 0.47 | +0.53 |
| `shura_001` | 9 | 10 | 0.90 | 0.05 | 0.62 | 0.92 | 0.47 | +0.44 |
| `shura_001` | 9 | 10 | 0.90 | 0.02 | 0.62 | 0.79 | 0.54 | +0.25 |
| `majlis_001` | 21 | 22 | 0.87 | 0.05 | 0.46 | 1.00 | 0.47 | +0.53 |
| `majlis_001` | 21 | 22 | 0.87 | 0.20 | 0.46 | 1.00 | 0.47 | +0.53 |
| `majlis_001` | 21 | 22 | 0.87 | 0.10 | 0.46 | 1.00 | 0.50 | +0.50 |
| `majlis_001` | 21 | 22 | 0.87 | 0.02 | 0.46 | 0.71 | 0.46 | +0.25 |
| `fazaa_001` | 7 | 8 | 1.00 | 0.20 | 0.21 | 0.96 | 0.78 | +0.18 |
| `fazaa_001` | 7 | 8 | 1.00 | 0.10 | 0.21 | 0.54 | 0.49 | +0.06 |
| `fazaa_001` | 7 | 8 | 1.00 | 0.02 | 0.21 | 0.25 | 0.25 | +0.00 |
| `fazaa_001` | 7 | 8 | 1.00 | 0.05 | 0.21 | 0.25 | 0.35 | -0.10 |
| `hayaa_001` | 3 | 4 | 0.95 | 0.20 | 0.29 | 0.92 | 0.08 | +0.83 |
| `hayaa_001` | 3 | 4 | 0.95 | 0.10 | 0.29 | 0.58 | 0.15 | +0.43 |
| `hayaa_001` | 3 | 4 | 0.95 | 0.05 | 0.29 | 0.42 | 0.21 | +0.21 |
| `hayaa_001` | 3 | 4 | 0.95 | 0.02 | 0.29 | 0.42 | 0.22 | +0.19 |

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
| `wasta_001` | 15 | 16 | 0.60 | -0.05 | 0.83 | 0.00 | 0.81 | +0.81 |
| `wasta_001` | 15 | 16 | 0.60 | -0.10 | 0.83 | 0.00 | 0.67 | +0.67 |
| `wasta_001` | 15 | 16 | 0.60 | -0.02 | 0.83 | 0.17 | 0.81 | +0.64 |
| `wasta_001` | 15 | 16 | 0.60 | -0.20 | 0.83 | 0.00 | 0.58 | +0.58 |
| `muruah_001` | 9 | 10 | 1.00 | -0.20 | 1.00 | 0.00 | 0.83 | +0.83 |
| `muruah_001` | 9 | 10 | 1.00 | -0.10 | 1.00 | 0.58 | 0.94 | +0.36 |
| `muruah_001` | 9 | 10 | 1.00 | -0.05 | 1.00 | 0.75 | 0.94 | +0.19 |
| `muruah_001` | 9 | 10 | 1.00 | -0.02 | 1.00 | 1.00 | 1.00 | +0.00 |
| `diyafa_001` | 3 | 4 | 0.83 | -0.05 | 0.92 | 0.00 | 0.97 | +0.97 |
| `diyafa_001` | 3 | 4 | 0.83 | -0.20 | 0.92 | 0.00 | 0.97 | +0.97 |
| `diyafa_001` | 3 | 4 | 0.83 | -0.10 | 0.92 | 0.00 | 0.92 | +0.92 |
| `diyafa_001` | 3 | 4 | 0.83 | -0.02 | 0.92 | 0.33 | 0.92 | +0.58 |
| `karam_001` | 9 | 10 | 0.77 | -0.20 | 0.92 | 0.25 | 0.72 | +0.47 |
| `karam_001` | 9 | 10 | 0.77 | -0.10 | 0.92 | 0.67 | 0.83 | +0.17 |
| `karam_001` | 9 | 10 | 0.77 | -0.05 | 0.92 | 0.75 | 0.81 | +0.06 |
| `karam_001` | 9 | 10 | 0.77 | -0.02 | 0.92 | 0.83 | 0.83 | +0.00 |
| `sharaf_001` | 19 | 20 | 1.00 | -0.20 | 1.00 | 0.00 | 0.86 | +0.86 |
| `sharaf_001` | 19 | 20 | 1.00 | -0.10 | 1.00 | 0.17 | 0.97 | +0.81 |
| `sharaf_001` | 19 | 20 | 1.00 | -0.05 | 1.00 | 0.67 | 0.97 | +0.31 |
| `sharaf_001` | 19 | 20 | 1.00 | -0.02 | 1.00 | 1.00 | 1.00 | +0.00 |
| `sabr_001` | 7 | 8 | 1.00 | -0.20 | 1.00 | 0.00 | 0.75 | +0.75 |
| `sabr_001` | 7 | 8 | 1.00 | -0.10 | 1.00 | 0.33 | 0.83 | +0.50 |
| `sabr_001` | 7 | 8 | 1.00 | -0.02 | 1.00 | 0.92 | 0.97 | +0.06 |
| `sabr_001` | 7 | 8 | 1.00 | -0.05 | 1.00 | 0.83 | 0.89 | +0.06 |
| `silat_rahim_001` | 5 | 6 | 0.73 | -0.20 | 0.83 | 0.00 | 0.83 | +0.83 |
| `silat_rahim_001` | 5 | 6 | 0.73 | -0.10 | 0.83 | 0.33 | 0.75 | +0.42 |
| `silat_rahim_001` | 5 | 6 | 0.73 | -0.05 | 0.83 | 0.67 | 0.75 | +0.08 |
| `silat_rahim_001` | 5 | 6 | 0.73 | -0.02 | 0.83 | 0.83 | 0.81 | -0.03 |
| `jiwar_001` | 7 | 8 | 0.77 | -0.10 | 0.92 | 0.00 | 0.69 | +0.69 |
| `jiwar_001` | 7 | 8 | 0.77 | -0.20 | 0.92 | 0.00 | 0.50 | +0.50 |
| `jiwar_001` | 7 | 8 | 0.77 | -0.05 | 0.92 | 0.50 | 0.78 | +0.28 |
| `jiwar_001` | 7 | 8 | 0.77 | -0.02 | 0.92 | 0.83 | 0.86 | +0.03 |
| `shura_001` | 9 | 10 | 0.88 | -0.10 | 1.00 | 0.00 | 0.86 | +0.86 |
| `shura_001` | 9 | 10 | 0.88 | -0.20 | 1.00 | 0.00 | 0.69 | +0.69 |
| `shura_001` | 9 | 10 | 0.88 | -0.05 | 1.00 | 0.42 | 0.94 | +0.53 |
| `shura_001` | 9 | 10 | 0.88 | -0.02 | 1.00 | 0.67 | 1.00 | +0.33 |
| `majlis_001` | 21 | 22 | 0.83 | -0.10 | 0.92 | 0.00 | 0.94 | +0.94 |
| `majlis_001` | 21 | 22 | 0.83 | -0.20 | 0.92 | 0.00 | 0.94 | +0.94 |
| `majlis_001` | 21 | 22 | 0.83 | -0.05 | 0.92 | 0.58 | 0.94 | +0.36 |
| `majlis_001` | 21 | 22 | 0.83 | -0.02 | 0.92 | 0.83 | 0.94 | +0.11 |
| `fazaa_001` | 7 | 8 | 1.00 | -0.20 | 1.00 | 0.00 | 0.17 | +0.17 |
| `fazaa_001` | 7 | 8 | 1.00 | -0.02 | 1.00 | 1.00 | 1.00 | +0.00 |
| `fazaa_001` | 7 | 8 | 1.00 | -0.05 | 1.00 | 1.00 | 0.97 | -0.03 |
| `fazaa_001` | 7 | 8 | 1.00 | -0.10 | 1.00 | 0.67 | 0.50 | -0.17 |
| `hayaa_001` | 3 | 4 | 0.94 | -0.20 | 1.00 | 0.08 | 0.97 | +0.89 |
| `hayaa_001` | 3 | 4 | 0.94 | -0.10 | 1.00 | 0.67 | 1.00 | +0.33 |
| `hayaa_001` | 3 | 4 | 0.94 | -0.05 | 1.00 | 0.92 | 1.00 | +0.08 |
| `hayaa_001` | 3 | 4 | 0.94 | -0.02 | 1.00 | 1.00 | 1.00 | +0.00 |

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
| `wasta_001` | en | 0.38 | 0.02 | +0.08 | +0.08 | 1.00 |
| `wasta_001` | en | 0.38 | 0.05 | +0.11 | +0.11 | 1.00 |
| `wasta_001` | en | 0.38 | 0.10 | +0.25 | +0.25 | 1.00 |
| `wasta_001` | en | 0.38 | 0.20 | +0.42 | +0.42 | 1.00 |
| `wasta_001` | ar | 0.31 | 0.02 | +0.19 | +0.44 | 2.29 |
| `wasta_001` | ar | 0.31 | 0.05 | +0.58 | +0.67 | 1.14 |
| `wasta_001` | ar | 0.31 | 0.10 | +0.69 | +0.69 | 1.00 |
| `wasta_001` | ar | 0.31 | 0.20 | +0.64 | +0.64 | 1.00 |
| `muruah_001` | en | 1.00 | 0.05 | +0.14 | +0.06 | 0.40 |
| `muruah_001` | en | 1.00 | 0.10 | +0.11 | +0.03 | 0.25 |
| `muruah_001` | en | 1.00 | 0.20 | +0.58 | +0.00 | 0.00 |
| `muruah_001` | en | 1.00 | 0.02 | +0.11 | -0.06 | -0.50 |
| `muruah_001` | ar | 0.88 | 0.20 | +0.64 | +0.56 | 0.87 |
| `muruah_001` | ar | 0.88 | 0.10 | +0.53 | +0.44 | 0.84 |
| `muruah_001` | ar | 0.88 | 0.05 | +0.39 | +0.31 | 0.79 |
| `muruah_001` | ar | 0.88 | 0.02 | +0.17 | +0.08 | 0.50 |
| `diyafa_001` | en | 0.69 | 0.02 | +0.31 | +0.31 | 1.00 |
| `diyafa_001` | en | 0.69 | 0.05 | +0.50 | +0.50 | 1.00 |
| `diyafa_001` | en | 0.69 | 0.10 | +0.58 | +0.58 | 1.00 |
| `diyafa_001` | en | 0.69 | 0.20 | +0.67 | +0.67 | 1.00 |
| `diyafa_001` | ar | 0.75 | 0.05 | +0.44 | +0.44 | 1.00 |
| `diyafa_001` | ar | 0.75 | 0.10 | +0.39 | +0.39 | 1.00 |
| `diyafa_001` | ar | 0.75 | 0.20 | +0.58 | +0.58 | 1.00 |
| `diyafa_001` | ar | 0.75 | 0.02 | +0.50 | +0.42 | 0.83 |
| `karam_001` | en | 0.69 | 0.20 | +0.64 | +0.56 | 0.87 |
| `karam_001` | en | 0.69 | 0.10 | +0.31 | +0.22 | 0.73 |
| `karam_001` | en | 0.69 | 0.05 | +0.17 | +0.08 | 0.50 |
| `karam_001` | en | 0.69 | 0.02 | +0.00 | +0.00 | 0.00 |
| `karam_001` | ar | 0.56 | 0.10 | +0.58 | +0.42 | 0.71 |
| `karam_001` | ar | 0.56 | 0.20 | +0.44 | +0.28 | 0.62 |
| `karam_001` | ar | 0.56 | 0.05 | +0.56 | +0.14 | 0.25 |
| `karam_001` | ar | 0.56 | 0.02 | +0.33 | +0.00 | 0.00 |
| `sharaf_001` | en | 0.62 | 0.20 | +0.92 | +0.92 | 1.00 |
| `sharaf_001` | en | 0.62 | 0.10 | +0.83 | +0.75 | 0.90 |
| `sharaf_001` | en | 0.62 | 0.02 | +0.25 | +0.17 | 0.67 |
| `sharaf_001` | en | 0.62 | 0.05 | +0.72 | +0.47 | 0.65 |
| `sharaf_001` | ar | 0.94 | 0.20 | +0.36 | +0.36 | 1.00 |
| `sharaf_001` | ar | 0.94 | 0.10 | +0.42 | +0.25 | 0.60 |
| `sharaf_001` | ar | 0.94 | 0.05 | +0.44 | +0.19 | 0.44 |
| `sharaf_001` | ar | 0.94 | 0.02 | +0.25 | +0.00 | 0.00 |
| `sabr_001` | en | 0.88 | 0.20 | +0.33 | +0.33 | 1.00 |
| `sabr_001` | en | 0.88 | 0.10 | +0.47 | +0.22 | 0.47 |
| `sabr_001` | en | 0.88 | 0.05 | +0.56 | +0.14 | 0.25 |
| `sabr_001` | en | 0.88 | 0.02 | +0.25 | +0.00 | 0.00 |
| `sabr_001` | ar | 0.75 | 0.02 | +0.11 | +0.11 | 1.00 |
| `sabr_001` | ar | 0.75 | 0.05 | +0.28 | +0.28 | 1.00 |
| `sabr_001` | ar | 0.75 | 0.20 | +0.36 | +0.36 | 1.00 |
| `sabr_001` | ar | 0.75 | 0.10 | +0.33 | +0.17 | 0.50 |
| `silat_rahim_001` | en | 0.69 | 0.02 | +0.06 | +0.06 | 1.00 |
| `silat_rahim_001` | en | 0.69 | 0.05 | +0.25 | +0.25 | 1.00 |
| `silat_rahim_001` | en | 0.69 | 0.10 | +0.56 | +0.56 | 1.00 |
| `silat_rahim_001` | en | 0.69 | 0.20 | +0.89 | +0.89 | 1.00 |
| `silat_rahim_001` | ar | 0.62 | 0.02 | +0.44 | +0.03 | 0.06 |
| `silat_rahim_001` | ar | 0.62 | 0.05 | +0.61 | +0.03 | 0.05 |
| `silat_rahim_001` | ar | 0.62 | 0.10 | +0.47 | -0.11 | -0.24 |
| `silat_rahim_001` | ar | 0.62 | 0.20 | +0.36 | -0.31 | -0.85 |
| `jiwar_001` | en | 0.81 | 0.05 | +0.28 | +0.28 | 1.00 |
| `jiwar_001` | en | 0.81 | 0.10 | +0.42 | +0.42 | 1.00 |
| `jiwar_001` | en | 0.81 | 0.20 | +0.42 | +0.42 | 1.00 |
| `jiwar_001` | en | 0.81 | 0.02 | +0.14 | +0.06 | 0.40 |
| `jiwar_001` | ar | 0.50 | 0.20 | +0.58 | +0.58 | 1.00 |
| `jiwar_001` | ar | 0.50 | 0.10 | +0.50 | +0.33 | 0.67 |
| `jiwar_001` | ar | 0.50 | 0.05 | +0.47 | +0.31 | 0.65 |
| `jiwar_001` | ar | 0.50 | 0.02 | +0.42 | +0.08 | 0.20 |
| `shura_001` | en | 0.56 | 0.05 | +0.31 | +0.31 | 1.00 |
| `shura_001` | en | 0.56 | 0.10 | +0.53 | +0.53 | 1.00 |
| `shura_001` | en | 0.56 | 0.20 | +0.67 | +0.67 | 1.00 |
| `shura_001` | en | 0.56 | 0.02 | +0.31 | +0.06 | 0.18 |
| `shura_001` | ar | 0.69 | 0.02 | +0.06 | +0.06 | 1.00 |
| `shura_001` | ar | 0.69 | 0.10 | +0.44 | +0.44 | 1.00 |
| `shura_001` | ar | 0.69 | 0.20 | +0.50 | +0.50 | 1.00 |
| `shura_001` | ar | 0.69 | 0.05 | +0.25 | +0.17 | 0.67 |
| `majlis_001` | en | 0.50 | 0.20 | +0.67 | +0.67 | 1.00 |
| `majlis_001` | en | 0.50 | 0.10 | +0.75 | +0.67 | 0.89 |
| `majlis_001` | en | 0.50 | 0.05 | +0.72 | +0.47 | 0.65 |
| `majlis_001` | en | 0.50 | 0.02 | +0.50 | +0.17 | 0.33 |
| `majlis_001` | ar | 0.50 | 0.10 | +0.42 | +0.42 | 1.00 |
| `majlis_001` | ar | 0.50 | 0.20 | +0.42 | +0.42 | 1.00 |
| `majlis_001` | ar | 0.50 | 0.05 | +0.42 | +0.25 | 0.60 |
| `majlis_001` | ar | 0.50 | 0.02 | +0.17 | +0.08 | 0.50 |
| `fazaa_001` | en | 1.00 | 0.10 | +0.31 | +0.31 | 1.00 |
| `fazaa_001` | en | 1.00 | 0.20 | +0.28 | +0.19 | 0.70 |
| `fazaa_001` | en | 1.00 | 0.05 | +0.14 | -0.03 | -0.20 |
| `fazaa_001` | en | 1.00 | 0.02 | +0.06 | -0.03 | -0.50 |
| `fazaa_001` | ar | 1.00 | 0.20 | +0.47 | +0.39 | 0.82 |
| `fazaa_001` | ar | 1.00 | 0.10 | +0.25 | +0.08 | 0.33 |
| `fazaa_001` | ar | 1.00 | 0.02 | -0.03 | -0.03 | 0.00 |
| `fazaa_001` | ar | 1.00 | 0.05 | +0.08 | -0.08 | -1.00 |
| `hayaa_001` | en | 1.00 | 0.05 | +0.14 | +0.14 | 1.00 |
| `hayaa_001` | en | 1.00 | 0.20 | +0.89 | +0.56 | 0.62 |
| `hayaa_001` | en | 1.00 | 0.10 | +0.39 | +0.22 | 0.57 |
| `hayaa_001` | en | 1.00 | 0.02 | +0.17 | +0.00 | 0.00 |
| `hayaa_001` | ar | 0.69 | 0.02 | +0.19 | +0.19 | 1.00 |
| `hayaa_001` | ar | 0.69 | 0.20 | +0.47 | +0.22 | 0.47 |
| `hayaa_001` | ar | 0.69 | 0.05 | +0.14 | +0.06 | 0.40 |
| `hayaa_001` | ar | 0.69 | 0.10 | +0.22 | +0.06 | 0.25 |

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
