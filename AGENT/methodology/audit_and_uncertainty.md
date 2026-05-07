---
name: Methodology commitments — verifier audit and uncertainty quantification
description: Protocols agreed for stratified verifier audit and bootstrap-CI uncertainty reporting; deviating from these silently is a paper-correctness bug.
type: project
---

**Verifier audit set per dataset (GSM8K, MATH, HumanEval):**
- 100 prompts hand-verified, written to `tests/manual_audit_<dataset>.md`.
- 50 sampled uniformly at random from a real model run.
- 50 *load-bearing* prompts: cases where exactly 1 of $K$ samples passes the automated verifier (these most influence the per-prompt $\max_k U(x_i, y_{i,k})$ term and so most influence $U^\circ_K$).

**Why:** Verifier false negatives (correct answer marked wrong) are the dominant measurement risk. Uniform sampling under-represents the prompts where the verifier matters most; load-bearing prompts are where a single misclassification flips the per-prompt outcome.

**How to apply:** When writing the audit spec for any dataset, build the audit set with this stratification. The 50 random + 50 load-bearing split is the contract; do not collapse to "100 random".

---

**Bootstrap CI protocol for $\hat{\mathcal{R}}_K$:**
- For each (model, dataset, seed): compute per-prompt values $\hat{\mathcal{R}}_K(x_i) = \max_k U(x_i, y_{i,k}) - \tfrac{1}{K}\sum_k U(x_i, y_{i,k})$.
- Bootstrap resample (B = 1000) over the prompt set; compute the mean of the resample; report mean and 95% CI.
- Across the 3 seeds: report mean ± std of the bootstrapped means.
- Table format: `R_hat_K = 0.156 [0.142, 0.171] (95% bootstrap CI over prompts), averaged over 3 seeds (std 0.004)`.

**Why:** Between-prompt variance is the load-bearing uncertainty for "the gap is real" claims (we generalise across prompts). Between-seed variance only captures sampling noise at fixed prompts and is secondary. Reporting only seed std understates uncertainty.

**How to apply:** Any table or figure that quotes $\hat{\mathcal{R}}_K$ uses this format. Bootstrap is over prompts (not over samples within a prompt, not over seeds). The metrics module exposes a function returning the per-prompt array so callers can bootstrap themselves.
