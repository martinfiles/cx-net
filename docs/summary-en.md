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
   directions, offset by ≈22° ([figure 2](figures/en/fig2.png)). This matches the known anatomy;
   none of the earlier assumed mappings reflected it.
4. **With the original model, even the real signs do not integrate** in any dynamical regime
   explored (588 evaluations). Adding trainable *per-type* gains (≈43 shared parameters, never
   per-edge) gives solutions that integrate, but **the real neurotransmitter assignment is not
   special**: 26 of the 64 per-type sign patterns integrate and the real one ranks 22nd
   (exact rank p = 22/64 = 0.34; [figure 3](figures/en/fig3.png)). The solutions rely on
   signed rates, i.e. deviations from a zero baseline, so an "inhibitory" type below baseline
   ends up exciting its targets ([figure 5](figures/en/fig5.png)), which makes the sign of a
   weight poorly identifiable.
5. **The annotated neurotransmitter is an exact function of cell type** (Delta7 = glutamate,
   the other five types = acetylcholine). "Recovering the neurotransmitter" therefore reduces
   to recovering one already-known bit (*Delta7 inhibits*), and per-neuron power calculations
   do not apply.

A generic sign learner, trained from a neutral initialisation, does not find the integrating
solution either (four pilots, held-out 0.458–0.561).

## What is and is not claimed

- Claimed: the five pitfalls above, with numbers traceable to [`results/`](../results); a
  checklist to run before interpreting sign agreement in connectome-constrained models.
- **Not** claimed: anything about the fly's actual neurotransmitter chemistry. Runs are
  exploratory (one run per configuration in the enumeration), and several hyperparameters were
  chosen with the help of the real assignment (which, if anything, favours it).
- Retracted along the way: the "chance" baseline, a per-neuron power control, a first
  real-versus-shuffled comparison, an "interleaved" angular map, and a "functional sufficiency"
  reading. The table is in section 9 of the report.

## Open items

Replicating each pattern with several seeds; characterising the bump and comparing it with
hand-tuned connectome ring-attractor models; validating the phase map on hemibrain; a search for a
dynamical regime that integrates with non-negative rates (a first sigmoid attempt was
inconclusive); review by someone from the field.

## Data, license and citation

Code: MIT. The connectome is licensed CC-BY 4.0 and is not redistributed here. Please cite Berg
et al. (2026), *Sexual dimorphism in the complete Drosophila male central nervous system
connectome*, Cell (bioRxiv 10.1101/2025.10.09.680999), and Eckstein et al. (2024) for the
neurotransmitter predictions. Figures: [`figures/en/`](figures/en/). Interactive explorer of the
64 patterns: see the project page.
