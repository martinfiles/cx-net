# Review request: CX-Net

Thank you for taking a look. This is a request for a critical read of an **exploratory,
mostly negative methodological note**. I am not asking you to validate a biological claim (the
note makes none); I am asking where the reasoning, the modelling or the framing is wrong,
naive or missing prior work.

**Estimated time:** 60–90 minutes for a focused read; 15 minutes if you only take the summary,
the five figures and the questions below.

## What it is, in five lines

A recurrent rate network is constrained only by the real synaptic topology of the core
head-direction circuit of *Drosophila* (152 neurons, 9,160 edges; EPG, EPGt, PEN_a, PEN_b, PEG,
Delta7; male CNS `male-cns:v1.0`), with weight magnitudes fixed to (normalised) synapse counts
and the sign of each edge as the only free per-edge parameter. The original question was whether
gradient descent on a heading-integration task recovers the annotated neurotransmitter of each
neuron. The answer was "no evidence, and the design could not have shown it". What remains are
five methodological findings, and a set of retractions of my own earlier claims.

## What to read, in this order

1. `summary-en.md` (5 min): the whole argument.
2. The six figures in `figures/en/` (5 min).
3. `preprint-discussion.md` (in Spanish; sections 2 and 4–8 carry the substance, ≈35 min). If
   Spanish is a problem, tell me and I will translate the sections you need.
4. The questions below.
5. **Only afterwards**, `expert-review.md`: my own adversarial read of the note. It lists the
   weaknesses I already know, so you do not spend time on them, but it may anchor you; read it
   after forming your own view.

## Questions I would most like answered

**Circuit and sign assumptions**
1. I assume Dale's law, acetylcholine → excitatory, glutamate → inhibitory (via GluClα for
   Delta7), and I treat `predictedNt` as annotation (a classifier prediction, not a
   measurement). Are these assumptions safe for EPG, EPGt, PEG, PEN_a, PEN_b and Delta7? Which
   references would you cite, and are there known exceptions (muscarinic receptors, excitatory
   glutamate, co-transmission)?
2. In this dataset the annotated neurotransmitter is an exact function of cell type. Is my
   conclusion that H1 reduces to one already-known bit (*Delta7 inhibits*) fair, or is there a
   sign-related question here I am overlooking (e.g. within-type, edge-level, or region-specific)?

**Model**
3. Is a `tanh` rate model with count-normalised weights, one time constant and a synthetic
   velocity input into PEN a reasonable baseline, or is it so far from the standard models
   that the negative results say nothing? Which hand-tuned connectome ring-attractor models
   should I compare against (I suspect Kakaria & de Bivort 2017, Turner-Evans et al. 2020 and
   Pisokas et al. 2020; please correct me), and what parameter regime do they need to get a
   bump that rotates?
4. Signed rates are read as deviations from a zero baseline. Is that acceptable, and how would
   you implement an explicit baseline? My one sigmoid attempt (baseline 0.5, same
   hyperparameters) was inconclusive.
5. The networks that "integrate" do so compressively (gain 0.54–0.76 against an ideal 1, with
   saturation; bump localisation 0.34–0.62). Is that plausible for this circuit, or a sign that
   something is off?

**Geometry**
6. I measured each neuron's angular position in the ellipsoid body from synapse coordinates (a
   PCA plane fit, `src/cx_net/extract_eb_angles.py`). The two hemispheres' EPG glomeruli
   traverse the ring in opposite directions, ≈22° apart, and EPGt sits at the phase of glomerulus
   1. Does that match what you expect? What weaknesses do you see in the method (torus,
   uneven sampling), and should it be checked against hemibrain and the published tables?

**Design and statistics**
7. I enumerated all 2⁶ = 64 per-type sign patterns (one run each, seed 0). 26 "integrate"; the
   real assignment ranks 22nd (exact rank p = 22/64). Is one run per pattern acceptable for an
   exploratory note? How many seeds would you want per pattern before believing a rank?
8. Hyperparameters were chosen in regions where the real assignment worked, which if anything
   favours it. Do you agree that this makes the null more, not less, credible?

**Framing**
9. Is this publishable as a short methods/negative note, and where (preprint, blog, workshop)?
   What is the minimum you would want added before I make it public?
10. Anything wrong, misleading or missing that I have not asked about.

## Main claims and where the evidence is

| claim | evidence |
|---|---|
| A constant decoder (0.18) beats all trained models (0.58–0.72) on the original task | `results/runs_summary.csv` (`signreg05_seed*`); report §4 |
| The anchored task's reference "remember the initial phase" scores 0.466 | report §4, notebook entries (4), (6) |
| EPG glomeruli traverse the ring in opposite directions per hemisphere | `results/epg_glomerulus_angles.csv`; report §5; figure 2 |
| Real signs do not integrate in any explored regime (588 evaluations) | `results/regime_search*.csv`; report §6 |
| With per-type gains, real signs integrate (0.106–0.124, 3 seeds); 12 shuffled assignments do not (0.444–0.598) | `results/dose_analysis.json` |
| 26 / 64 per-type patterns integrate; the real one ranks 22nd | `results/types_analysis.json`; figure 3 |
| The real-assignment solution runs Delta7 below baseline 89% of the time | `results/rates_by_type.json`; figure 5 |
| Integrating networks respond antisymmetrically and roughly proportionally, compressively | `results/bump_check.json` |
| The integrating networks move a broad graded bump (FWHM ≈ 120–170°, no saturation); the non-integrating one is a frozen saturated pattern; phase memory does not discriminate | `results/bump_profile.json`; figure 6 |
| A generic sign learner does not find the solution (4 pilots, 0.458–0.561) | `results/runs_summary.csv` (`joint_sr*`) |

## Reproducing

Without any data (seconds): `python -m src.cx_net.make_figures` regenerates the figures from
`results/`, and `results/runs_summary.csv` has one row per training run (133+ runs).
With data: a personal neuPrint token, `python -m src.cx_net.extract_graph`, then
`python -m src.cx_net.extract_eb_angles`; each 600-epoch run takes ≈40–50 min on CPU, and the
64-pattern enumeration ≈2.5 h on 16 cores. Please train with a recent PyTorch (an old version
returned `nan` for `atan2(0,0)`; fixed in `task.decode_heading`).

## Scope

I am **not** asking for a statistical audit of every run, a reproduction of the training, or an
opinion on the fly's actual neurotransmitter chemistry. A frank list of what is wrong or naive is
the most useful thing you can send. Comments in any format are fine.

## Data and licence

Code MIT. The male CNS connectome is CC-BY 4.0 (Berg et al., 2026, *Cell*; bioRxiv
10.1101/2025.10.09.680999) and is not redistributed. Neurotransmitter predictions: Eckstein et
al. (2024).
