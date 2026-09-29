# Review request: CX-Net

Thank you for taking a look. This is a request for a critical read of an **exploratory,
mostly negative methodological note**. I am not asking you to validate a biological claim (the
note makes none); I am asking where the reasoning, the modelling or the framing is wrong,
naive or missing prior work.

**Estimated time:** 60–90 minutes for a focused read; 15 minutes if you only take the summary,
the five figures and the questions below.

**Since your last read (2026-09-28):** fixed the 8 numeric/text errors and the reproducibility
gap you flagged; verified and added the citations you found; confirmed the Delta7×EPG×PEN_b
loop pattern from your enumeration data; and did four follow-up checks: (1) split the PEN phase
offset by ROI (PB-only input / EB-only output) — confirms the offset was diluted ≈9× by
region-mixing, as you suspected; (2) rebuilt the graph from PB+EB synapses only and retrained —
surprising, opposite-of-expected result (see the last two rows of the claims table); (3)
formalised the sign-flip symmetry against this model's actual constraints — it turns out **not**
to hold, so the 64 patterns are not gauge-equivalent; (4) redesigned the task per your
overshoot/undershoot critique (correlated velocity, multi-lap trials, process noise) — closes
the loophole in principle, but nothing has converged on it in three attempts (more epochs, then
a curriculum). All four are new and none has had a second pair of eyes — if you only have time
for one thing, that's probably the most useful place to look.

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
5. The networks that "integrate" respond to velocity with a sigmoidal gain, not a constant one:
   they overshoot the expected displacement at low speeds and saturate at high speeds
   (per-step gain ≈1.65→0.46 for the real assignment as av goes 0.01→0.08; bump localisation
   0.34–0.62). Is that plausible for this circuit, or a sign that
   something is off?

**Geometry**
6. I measured each neuron's angular position in the ellipsoid body from synapse coordinates (a
   PCA plane fit, `src/cx_net/extract_eb_angles.py`). The two hemispheres' EPG glomeruli
   traverse the ring in opposite directions, and the 16 hemisphere-glomerulus positions
   interleave around the ring at a mean step of ≈22.5° (not a constant 22° gap between
   homologous L/R glomeruli — that only holds for two of the eight glomerulus indices), and
   EPGt sits at the phase of glomerulus 1. Does that match what you expect? What weaknesses do
   you see in the method (torus, uneven sampling), and should it be checked against hemibrain
   and the published tables?

**Design and statistics**
7. **Update (answers your own question):** I enumerated all 2⁶ = 64 per-type sign patterns with
   3 seeds each (was 1). 28 "integrate" by majority vote (was 26); the real assignment now
   ranks 14th by mean held-out loss (p = 14/64 = 0.22, was 22nd/0.34) and integrates in all 3
   of its own seeds. 4 of the other 63 patterns flip verdict between seeds. Is 3 seeds enough,
   or would you still want more before trusting this rank?
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
| 28 / 64 per-type patterns integrate (3 seeds, majority vote); the real one ranks 14th of 64 | `results/types_analysis_3seeds.json`; figure 3 |
| The real-assignment solution runs Delta7 below baseline 89% of the time | `results/rates_by_type.json`; figure 5 |
| Integrating networks respond antisymmetrically, but with a sigmoidal (not constant) gain — they overshoot at low speed and saturate at high speed | `results/bump_check.json` |
| The integrating networks move a broad graded bump (FWHM ≈ 120–170°, no saturation); the non-integrating one is a frozen saturated pattern; phase memory does not discriminate | `results/bump_profile.json`; figure 6 |
| With free positive per-edge gains (signs fixed) the real assignment integrates only if the edge-gain learning rate is small (3/3 seeds), where the gains barely move (5–95th percentile 0.8–1.3) and 0/6 shuffled patterns integrate; with the default rate it fails (optimisation) | `results/edge_gain_analysis.json`, `results/edge_gain_lr005_analysis.json` |
| A generic sign learner does not find the solution (4 pilots, 0.458–0.561) | `results/runs_summary.csv` (`joint_sr*`) |
| Splitting PEN's EPG neighbours by ROI (PB-only input, EB-only output) gives a much larger, consistent phase offset (≈50–54°, close to one glomerulus) than the region-mixed graph (≈6°) | `results/pen_offset_by_region.json`; report §5 |
| Rebuilding the whole graph from PB+EB synapses only and repeating the check: the untrained search still can't integrate with real signs alone (388+216=604 evaluations; this untrained search says nothing about the trained model), and with per-type gains trained on it, the real assignment does *worse* than on the full graph (none of 3 seeds integrate) | `results/regime_search_pbeb*.csv`; report §6, §11 |
| The naive sign-flip symmetry (`r → Dr`) does **not** hold for this model's actual parametrization (positive per-type-pair gains, complete type-connectivity graph forces `D` to be globally constant) — the 64 patterns are not gauge-equivalent | `src/cx_net/sign_symmetry_analysis.py`; report §6 |
| A redesigned task (correlated OU velocity, trials spanning several full laps, process noise) closes the "gain-0.54 integrator passes" gap in principle (that integrator now scores 0.85, as bad as not integrating) — the first three attempts (600 epochs, 1400 epochs, a curriculum) failed because of velocity (filtered p90 0.19 vs 0.047 in the old task); with max_av=0.04, T=500 and an increment loss the 3 real seeds converge and 4 shuffles do not (2026-09-29) | `results/task_v2_reference.json`; notebook 2026-09-28 (6), 2026-09-29, 2026-09-29 (2) |

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
