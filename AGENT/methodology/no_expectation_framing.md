---
name: U_circ_K is a sample average, not an expectation
description: When computing R_hat_K at varying K, transcribe the paper formula directly via truncation. No binomial closed forms, no "estimator of an expectation" framing.
type: feedback
---

Rule: $U^\circ_K$, $\bar{U}_K$, $\hat{\mathcal{R}}_K$ at varying $K$ from an $(M, n_\text{max})$ utility matrix are computed by **direct truncation**:

```
U_circ_at_K(utilities, K)  =  utilities[:, :K].max(axis=1).mean()
U_bar_at_K(utilities, K)   =  utilities[:, :K].mean()
R_hat_at_K(utilities, K)   =  U_circ_at_K(utilities, K) - U_bar_at_K(utilities, K)
```

Never:
- Use the Codex/HumanEval pass@K closed form `1 - C(n-c, K) / C(n, K)`.
- Describe these quantities as "estimators of an expectation" or "unbiased estimators of …" in code, docstrings, or prose.
- Use phrases like "in expectation", "asymptotic", "expected pass rate", "unbiased", "marginalised over subsets".

**Why:** The paper defines $U^\circ_K = \frac{1}{M}\sum_i \max_{k\in[K]} U(x_i, y_{i,k})$ — a direct sample average over the $K$ samples we have, not an expectation over a stochastic-process draw. Closed-form binomial estimators are mathematically equivalent under "random uniform K' subset of $n_\text{max}$" sampling and reduce per-prompt sampling variance, but they import a framing the paper does not adopt. With $M \approx 1000+$ prompts, the per-prompt-variance reduction is negligible compared to between-prompt variance (which the methodology memo's bootstrap CI captures). The user explicitly corrected an earlier suggestion to use the binomial form on 2026-05-07: *"We do not need the binomial-coefficient closed form, and we should not introduce 'expectation' framing… Keep the code as a one-line direct transcription of the paper formula."*

**How to apply:**
- Saturation curves take the first $K'$ columns of the $(M, n_\text{max})$ matrix.
- With vLLM's `n = n_max` + fixed `seed`, the column ordering is deterministic, so the curve is reproducible.
- Docstrings write the paper formula in LaTeX and stop there. Do not annotate it as "estimator of" anything.
- Prose ("the rational gap at sampling budget $K$") is fine; expectation language is not.
- The implementation lives at `src/metrics/rational_gap.py`: `U_circ_at_K`, `U_bar_at_K`, `R_hat_at_K`.
