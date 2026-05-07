# methodology/ — load-bearing rules carried forward across sessions

This directory contains the project's methodological commitments — rules that don't fit naturally into [AGENT.md](../AGENT.md) (which is a coding-agent contract) but that any agent working on the codebase must respect.

The previous session's per-project memory was machine-local and is not portable across servers. These files are the durable replacement: they ship in the repo and are read by every new session.

## Read order (≈ 10 minutes)

1. [user.md](user.md) — who the user is and what kind of collaboration they expect.
2. [terminology.md](terminology.md) — strict rules on `U_circ_K` / `U_bar_K` / `R_hat_K`. Banned identifiers.
3. [hypotheses.md](hypotheses.md) — the paper's H1–H4 reformulations. Overrides any looser framing in older README drafts.
4. [audit_and_uncertainty.md](audit_and_uncertainty.md) — stratified verifier audit + bootstrap-over-prompts CI protocol.
5. [no_expectation_framing.md](no_expectation_framing.md) — saturation curves are direct truncation, not closed-form expectation estimators.
6. [no_skipif_for_invariants.md](no_skipif_for_invariants.md) — paper-load-bearing tests run live, never skipped.
7. [build_plan.md](build_plan.md) — historical record of why the codebase has its current shape.

If any rule here conflicts with [AGENT.md](../AGENT.md), the rule wins for methodology and AGENT.md wins for code-style enforcement; in practice they have not conflicted.
