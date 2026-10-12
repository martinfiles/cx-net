# CX-Net: summary in English

*Exploratory methodological note. The full report (in Spanish) is
[`preprint-discussion.md`](preprint-discussion.md); the day-by-day record, including mistakes and
their corrections, is [`lab-notebook.md`](lab-notebook.md); an internal critical review is
[`expert-review.md`](expert-review.md).*

## Question

If a recurrent network is constrained only by the real synaptic **topology** of the core
head-direction circuit of *Drosophila* (152 neurons, 9,160 edges: EPG, EPGt, PEN_a, PEN_b, PEG,
Delta7; male CNS connectome `male-cns:v1.0`), with every weight magnitude fixed to the synapse
count and the **sign** as the only free per-edge parameter, does gradient descent on a
heading-integration task recover the annotated neurotransmitter of each neuron (H1)?

## Answer

**We found no evidence that it does, and the design could not have found it.** The value of the
work is five methodological findings, each of which invalidated an earlier version of the
experiment. Throughout, "integrates" means held-out loss < 0.466 (the loss of remembering the
initial phase without integrating) and a slope > 0.5 between decoded and true heading change.

1. **The original task was trivially solvable.** A constant decoder scored 0.18 held-out loss;
   the trained models scored 0.58–0.72. The "chance" baseline (≈0.85) used at first was a random
   phase, not a trivial solution ([figure 1](figures/en/fig1.png)).
2. **The task had no phase anchor** (it asked for absolute heading with no cue), so even a
   perfect integrator scored ≈1.0. Adding a random initial phase with a brief cue fixed this.
3. **The angular map of the ring was wrong.** Measured from synapse coordinates in neuPrint
   (261,546 synapses), EPG glomeruli of the two hemispheres traverse the ring in *opposite*
   directions, and the 16 hemisphere-glomerulus positions interleave around the ring at a mean
   step of ≈22.5° ([figure 2](figures/en/fig2.png)); homologous L/R glomeruli are *not* a
   constant 22° apart (the gap grows with glomerulus index). This matches the known anatomy;
   none of the earlier assumed mappings reflected it.
4. **With the original model, even the real signs do not integrate** in any dynamical regime
   explored (588 evaluations). Adding trainable *per-type* gains (≈43 shared parameters, never
   per-edge) gives solutions that integrate, but **the real neurotransmitter assignment is not
   special**: with 3 seeds per pattern, the number that integrate depends on the aggregation rule
   (20/64 in all 3 seeds, 28/64 by majority, 36/64 in at least one), and the real one ranks
   14th–28th of 64 depending on that rule (p = 0.22–0.44; a single-seed pass had put it 22nd of
   26 integrating). 16 of the other 63 patterns are not unanimous across seeds, confirming a
   single run is noisy; of those, 4 flip their majority verdict relative to seed 0. The real
   assignment itself integrates in all 3 seeds (0.106–0.124; [figure 3](figures/en/fig3.png)). The
   solutions rely on signed rates, i.e. deviations from a zero baseline, so an "inhibitory" type
   below baseline ends up exciting its targets ([figure 5](figures/en/fig5.png)), which makes the
   sign of a weight poorly identifiable. The networks that integrate do move a graded, broad bump
   (FWHM ≈ 120–170° at 30° resolution, no saturated units), but the speed response is **sigmoidal,
   not a constant gain**: it overshoots the expected displacement at low speeds and saturates at
   high speeds (per-step gain ≈1.65→0.46 over av = 0.01→0.08 for the real assignment); the one
   that does not integrate is a frozen saturated pattern ([figure 6](figures/en/fig6.png)).
   Holding an arbitrary phase at zero velocity does not discriminate between them.
5. **The annotated neurotransmitter is an exact function of cell type** (Delta7 = glutamate,
   the other five types = acetylcholine). "Recovering the neurotransmitter" therefore reduces
   to recovering one already-known bit (*Delta7 inhibits*), and per-neuron power calculations
   do not apply.

A generic sign learner, trained from a neutral initialisation, does not find the integrating
solution either (four pilots, held-out 0.458–0.561). On the corrected task (preregistered,
32 runs, none excluded) a learner with one sign per cell type integrates in 6-7 of 10 seeds but
recovers the real assignment in **0 of 20** runs (chance reference 1/37), landing on 9 different
patterns per arm; a per-edge learner agrees with the real signs at chance level (0.51); and
started *at* the real assignment all 6 runs keep integrating but drift away from it (Delta7
stays inhibitory). So this task does not make the learner prefer the real chemistry, consistent
with 37 of 64 patterns integrating; it does not show that no learner could find it.

## What is and is not claimed

- Claimed: the five pitfalls above, with numbers traceable to [`results/`](../results); a
  checklist to run before interpreting sign agreement in connectome-constrained models.
- **Not** claimed: anything about the fly's actual neurotransmitter chemistry. Runs are
  exploratory (3 seeds per pattern in the enumeration, 1 seed in most other analyses), and
  several hyperparameters were chosen with the help of the real assignment (which, if anything,
  favours it).
- Retracted along the way: the "chance" baseline, a per-neuron power control, a first
  real-versus-shuffled comparison, an "interleaved" angular map, and a "functional sufficiency"
  reading. The table is in section 9 of the report.

## Open items

Comparing the bump profile with hand-tuned connectome ring-attractor models; validating the
phase map on hemibrain; a search for a dynamical regime that integrates with non-negative rates
(a first sigmoid attempt was inconclusive); explaining *why* the trained per-type-gain solution
fails on the PB+EB-only graph below, now that "insufficient recurrent_gain" has been ruled out;
review by someone from the field. (The 64-pattern enumeration under the redesigned task
has now been run; see the update of 2026-10-12.)

**Sign-flip symmetry, formalised and checked against this model's actual constraints
(2026-09-28).** The external review proposed that with an odd activation and trainable biases,
the dynamics are invariant under `r → Dr` (`D` diagonal, ±1 entries) — a sign-graph "switching"
symmetry predicting that only loop-sign products, not individual type signs, should be
identifiable. That holds for a model with *free-signed* per-edge or per-type gains. This
project's model constrains its per-type-pair gains to be *positive* (`exp(log_pair_gain)`).
Checking that constraint (`src/cx_net/sign_symmetry_analysis.py`): a nontrivial `D` is only
compatible with the model if, for every pair of types with a synapse between them, `D` agrees on
both — and the type-connectivity graph is complete (all 36 ordered type pairs, including
self-loops, have nonzero weight, both on the full graph and the PB+EB-only graph). That forces
`D` to be globally constant: **only the trivial symmetry exists; the 64 sign patterns are not
gauge-equivalent, and their very different held-out losses (0.05–0.77) are genuine dynamical
differences**, not an artifact of an unbroken symmetry. This corrects an earlier, more informal
statement of the symmetry in the Spanish report. The Delta7×EPG×PEN_b regularity found in the
enumeration (section 6) remains an unexplained empirical pattern, not a consequence of gauge
freedom.

**EMA-based checkpoint selection, implemented (2026-09-28).** Picking the checkpoint from the
single lowest-loss batch of 16 trials (`best_state`) is itself a source of selection noise, as
the lab notebook already flagged. `train.py` now supports `checkpoint_selection="ema"`, picking
the checkpoint at the minimum of the same exponential moving average already used for the LR
scheduler — much less noisy than one batch. Default behaviour is unchanged. Re-running the full
64-pattern enumeration with it is still pending.

**Task redesign (2026-09-28): closes the theoretical gap, but needed more training budget than
expected.** The external review flagged that velocity was step-independent white noise (median
trial heading drift ≈60°, never a full lap), and the dynamics had no process noise — together
letting a gain-0.54 "integrator" pass the "integrates" bar with loss 0.138. `task.py` gained
`velocity_process="ou"` (a time-correlated Ornstein-Uhlenbeck process) and `model.py` gained
`state_noise_std` (additive process noise in the recurrent dynamics); both default to the old
behaviour. With `T=300`, `tau_v=30` (median 1.68 laps per trial, 63% exceed one full lap), the
loophole closes cleanly: a gain-0.54 integrator now scores 0.85 (vs. 0.98 for memorising the
initial phase and 1.12 for a constant decoder) — as bad as not integrating at all — and gain
must be within ±5% of 1.0 for loss < 0.2 (`results/task_v2_reference.json`). **But nothing has
trained on it in three attempts.** 3 real seeds + 4 shuffles, same dynamical hyperparameters
throughout: 600 epochs converges on nothing (held-out 0.83–1.06, slope ≈0); 1400 epochs (more
than double) doesn't either (held-out 0.82–1.04, slope ≈0); a curriculum (1400 epochs, the first
800 ramping from T=200/`tau_v`=5 — almost as easy as the old task — up to the T=300/`tau_v`=30
target) doesn't either (held-out 0.93–1.07, slope between −0.04 and +0.02). Not even the real
assignment separates from shuffled controls in any of the three, so neither more epochs nor a
difficulty ramp alone is the problem. An ideal cumsum integrator with gain near 1.0 scores loss ≈0 (that is not a network of this model class, so it does not show a solution exists in the search space), but gradient descent with this dynamical-hyperparameter recipe (inherited from
the old task, never re-tuned) doesn't find it under any of the three training schedules tried.
**(Superseded 2026-09-29, see the update at the end: the failures were about velocity.) At that point the task redesign was an open limitation, not a result**: it closes the theoretical gap that
motivated it, but couldn't be used to redo any analysis in this report. Not yet tried: a regime
search (untrained, like the one used for the PB+EB graph) specific to this task, or a different
learning rate or initialisation.

**PEN phase offset, split by region, and a graph rebuilt from PB+EB synapses only
(2026-09-28).** The original phase-map validation (`lab-notebook.md`, 2026-09-19) weighted each
PEN neuron's EPG neighbours without separating compartments, and got a diluted ≈6° L/R offset.
Neuprint's per-synapse ROI breakdown shows why: EPG→PEN synapses are almost all in the PB,
PEN→EPG synapses almost all in the EB. Splitting the input phase (presynaptic EPG neighbours, PB
only) from the output phase (postsynaptic EPG neighbours, EB only) per PEN neuron gives a much
larger, consistent offset — PEN_a −54.1°(L) / +54.3°(R) (n=10+10), PEN_b −51.4°(L) / +50.4°(R)
(n=11+11) — close to one glomerulus (≈45°), with the same perfect L/R antisymmetry
(`src/cx_net/pen_offset_by_region.py`, `results/pen_offset_by_region.json`). Rebuilding the
whole connectivity graph from PB+EB synapses only (dropping the 13.4% of synaptic weight in NO,
FB, LAL(L/R) and CentralBrain-unspecified) and repeating both the untrained regime search and
the per-type-gain training gave a **surprising, opposite-of-hypothesised result**: the untrained
search still cannot integrate with real signs alone (same as the full graph), and with per-type
gains trained on this smaller graph — same hyperparameters as before, not re-tuned — the real
assignment does *worse*, not better: held-out 0.28–0.34 (vs. 0.11–0.12 before), slope 0.36–0.42
(vs. 0.75–0.84 before), and **none of the 3 seeds meet the "integrates" bar** on this graph.
Dropping the synapses outside PB/EB apparently removes dynamics the trained solution relies on,
not just noise diluting the phase offset — the opposite of what motivated the check. A follow-up,
re-tuned untrained regime search (`recurrent_gain` up to 24, 1.5× the range tried on the full
graph; 216 more evaluations, `results/regime_search_pbeb_r3.csv`) closes the obvious objection:
it is not a matter of insufficient recurrent gain — the same negative holds across the whole
range (best loss 0.498, slope between −0.14 and +0.01 in the top 20 configurations). What NO/FB/
LAL(L/R) contribute to the trained solution isn't just "more recurrence"; it's something more
specific that the per-type gains can't compensate for on the restricted graph — mechanism still
unexplained.

## Data, license and citation

Code: MIT. The connectome is licensed CC-BY 4.0 and is not redistributed here. Please cite Berg
et al. (2026), *Sexual dimorphism in the complete Drosophila male central nervous system
connectome*, Cell (bioRxiv 10.1101/2025.10.09.680999), and Eckstein et al. (2024) for the
neurotransmitter predictions. Figures: [`figures/en/`](figures/en/). Interactive explorer of the
64 patterns: [`explorer.html`](explorer.html) (open locally in a browser; EN/ES toggle).

## Update 2026-09-29 (third external review)

- **The region missing in PB+EB is the NO.** The NO holds only PEN-PEN synapses (about 11% of total weight). Zeroing it in the three trained real-assignment models breaks integration (held-out 0.43, slope 0.10, 0/3); zeroing FB, LAL or CentralBrain changes nothing. Retraining confirms it: full graph without NO 0/3 integrate (held-out 0.30-0.36), PB+EB+NO 3/3 (0.10-0.13). `results/region_analysis.json`, `data/interim/launch_regions.sh`.
- **PEN offset without region filtering** is about 6 deg (mean |offset| 6.3 PEN_a, 6.2 PEN_b): the dilution is real. Correction: only 12% of EPG-to-PEN synapses are in the PB (1,439 of about 12.3k), so the mixed-graph offset is dominated by EB synapses.
- **PB+EB specificity exists** (real held-out 0.31, slope 0.39, vs shuffled 0.43-0.60 and <=0.07); it just falls short of the arbitrary integrates threshold. The untrained-search count is 388+216=604 evaluations, and that search says nothing about the trained model (the "caveat closed" wording is withdrawn).
- **OU task, corrected.** The earlier OU failures were about velocity (filtered p90 0.19 vs 0.047 in the old task; no trained network moved the bump faster than about 0.037-0.055 rad/step). With `max_av=0.04`, T=500, `tau_v=50` (p90 0.055) and an increment-based loss, the 3 real seeds converge (increment loss 0.12, slope 0.73-0.75) and the 4 per-neuron shuffles do not (0.35-0.46, slope 0.10-0.20). Real networks trained on the old task already transfer (slope 0.73-0.78). The 64-pattern enumeration on this task was redone afterwards: see the update of 2026-10-12.
- **Symmetry.** "No exact per-type symmetry" does not imply "the differences are real": some type pairs are near zero (PEN to Delta7: 14-15 synapses) and PEG/EPGt outputs are weak, so approximate symmetries exist; changing only their sign moves held-out by a median 0.055 (global sd 0.17); 16/63 patterns are not unanimous across seeds. `results/sign_symmetry_pairs.json`.
- **Delta7 x EPG rule (PEN_b excitatory):** at least one of EPG or Delta7 inhibits - 22/24 integrate vs 0/8 when neither does. It is not "negative loop feedback": when both inhibit, the loop is positive and 6/8 still integrate.

## Update 2026-10-12 (64 patterns on the corrected OU task)

- **Same conclusion on the corrected task.** All 64 sign patterns by cell type x 3 seeds (189 runs, none excluded; `src/cx_net/analyze_types_v3.py`, `results/types_analysis_v3.json`). Criterion fixed beforehand: "integrates" = held-out increment loss < 0.538 (remembering the initial phase) and slope > 0.5, majority vote over 3 seeds. **37/64 patterns integrate by majority (32/64 in all 3 seeds); the real assignment (increment loss 0.121, slope 0.74, integrates in 3/3) ranks 30th of 64 (p = 0.47).** 29 patterns beat it, with slopes around 0.95 against 0.74, so the real network is a more compressive integrator.
- **By type** (of 32 patterns, number that integrate with the type excitatory / inhibitory): Delta7 7/30, EPG 20/17, EPGt 19/18, PEG 18/19, PEN_a 15/22, PEN_b 18/19. Only Delta7 separates (inhibitory is nearly sufficient, not necessary); the PEN_b contrast seen on the old task (20/32 vs 6/32) disappears. Agreement with the old-task verdict: 23 integrate in both, 5 only on the old task, 14 only on the new one, 22 on neither.
- **Robustness (post hoc).** The real assignment ranks 37th, 36th and 37th in single seeds (30th by the 3-seed mean), so the rank is a descriptive, not a p-value. The count depends on the slope threshold: 37 patterns with slope > 0.5 or > 0.7, 32 with > 0.8 and 20 with > 0.9; at > 0.8 the real assignment (slope 0.74) no longer integrates in any seed, so "the real one is not special" is robust but "the real one integrates" is not. Given Delta7 inhibitory (32 patterns), 30 integrate and the real one ranks 28th of 32; with Delta7 excitatory 7 of 32 integrate and the best reaches 0.058. The task therefore selects the one known bit (Delta7 inhibits) and nothing more specific.
- **Caveats.** Three seeds (seed 0 alone gave 41/64 and rank 37, so the figures move); the sign-learner result (0/20) depends on hyperparameters chosen with the real assignment and on an 'integrates' threshold whose binarised/continuous reading changes the formal label (not the 0/20); the loss threshold is loose and the slope does the discriminating; per-type descriptives are untested; hyperparameters were chosen with the real assignment.
