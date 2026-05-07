---
name: Paper hypotheses — reformulated framings (2026-05-07)
description: Updated H1–H3 framings agreed in design discussion before coding. These override looser framings implied by earlier README drafts.
type: project
---

The paper's headline definition: $\mathcal{R}(\pi_\theta) = \mathbb{E}_x[U(x, y^\circ_\theta) - \mathbb{E}_{y\sim\pi_\theta}U(x,y)]$. Empirical estimator $\hat{\mathcal{R}}_K = U^\circ_K - \bar{U}_K$.

**H1 — existence, reformulated:** $\hat{\mathcal{R}}_K$ grows substantially with $K$ before saturating, indicating the reachable region of $\pi_\theta$ contains high-utility outputs that sampling does not concentrate on.
- $K$ is a first-class axis throughout the paper.
- Headline figure is the saturation curve $\hat{\mathcal{R}}_K$ vs $K$, not a bar chart at a single $K$.
- $K=1$ is the trivial baseline ($\hat{\mathcal{R}}_1 \equiv 0$ by construction).
- $K=64$ is reported as a reference point with explicit acknowledgement that the magnitude depends on $K$.

**H2 — independence from value alignment, reformulated as claim (b):** alignment does not *eliminate* the gap, even when $\bar{U}_K$ improves.
- Along SFT → DPO → RLVR: $\bar{U}_K$ should rise monotonically (alignment working).
- $U^\circ_K$ may stagnate or decline in late stages (distributional sharpening narrows the reachable region).
- $\hat{\mathcal{R}}_K$ may shrink, but if it shrinks because $U^\circ_K$ shrinks (rather than $\bar{U}_K$ catching up), that supports H2 in a particularly strong form.

**H3 — inference mechanisms:** oracle Best-of-$N$ at $N=K$ is by construction equal to $U^\circ_K$; it is the reachable upper bound, not a candidate inference procedure.
- Oracle BoN drawn as horizontal reference line on H3 plots, labelled "reachable upper bound".
- Deployable procedures evaluated against it: direct sampling at $\tau\in\{0, 0.7, 1.0\}$, CoT, self-consistency over 8 CoT samples, MCTS with budget-matched search.
- Metric of interest is the *residual gap* $U^\circ_K - u_I$ for each procedure $I$.

**H4 — context length:** $\hat{\mathcal{R}}_K(L)$ as a function of max reasoning length $L \in \{0, 64, 128, 256, 512, 1024, 2048\}$ on MATH and GSM8K via two-stage budget forcing. Relationship may be non-monotonic.

**Why:** These framings were agreed in the first design discussion (2026-05-07), correcting and tightening the framings implicit in the original README. The user committed to updating the paper draft and README to reflect them.

**How to apply:** When implementing experiments, plotting, or writing scripts that touch H1–H4 outputs, follow these framings — not the original single-$K$ / oracle-BoN-as-procedure framings that earlier drafts implied. If a script or figure spec contradicts these, flag it.
