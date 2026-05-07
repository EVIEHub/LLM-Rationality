---
name: Rational-gap terminology rule (never pass@K)
description: In all code, docstrings, logs, and discussion for this project, use U_circ_K / U_bar_K / R_hat_K — never pass@K or pass@1.
type: feedback
---

Rule: Use the paper's symbols throughout — $U^\circ_K$, $\bar{U}_K$, $\hat{\mathcal{R}}_K$. In code: `U_circ_K`, `U_bar_K`, `R_hat_K` (or the long forms `reachable_utility`, `mean_utility`, `rational_gap`). Never `pass_at_k` / `pass_at_1` / `pass@K` / `pass@1` as variable names, log fields, function names, or in user-facing discussion.

**Why:** `pass@K` is a benchmark evaluation metric. $U^\circ_K$ and $\bar{U}_K$ are estimators of the two terms in the paper's *general* definition of rational gap, which is well-defined for any utility function. They coincide *numerically* with pass@K and pass@1 only in the binary-utility case. Using "pass@K" obscures the conceptual generality and the paper-specific framing the user has built. The user explicitly corrected this on 2026-05-07 in our first design discussion.

**How to apply:**
- Variable / function / log-field names: `U_circ_K`, `U_bar_K`, `R_hat_K`, or the long forms `reachable_utility`, `mean_utility`, `rational_gap`.
- Docstrings: primary description references $U^\circ_K$ as defined in the paper. May mention "coincides with pass@K under binary utility" once per module for reader convenience, no more.
- The `metrics` module takes a callable verifier returning `float`, not `int`, so the implementation is general over $U$ even though all current verifiers are binary. Individual verifier modules may still return `int` in {0, 1}; the metrics layer treats them as floats.
- In discussion with the user (chat, plans, summaries): use the paper symbols. Do not slip into pass@K shorthand even when explaining intuition.
- AGENT.md §4.2 contains a "Forbidden alternatives" bullet listing `pass_at_k`, `pass@k`, `pass1`, `pass_rate`, `accuracy`, `success_rate` as banned identifiers. That list is enforced by code review, not by tooling.
