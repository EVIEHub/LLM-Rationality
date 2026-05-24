# Appendix sections still needing extra experiments

The following sections are NOT auto-filled because they depend on data
that isn't in the current result JSONs. Each item lists exactly what
needs to be produced.

## C.2 — L (verifier-call) sensitivity (UltraFeedback / preference cell)
Needs the per-(prompt, candidate) **L=5 raw verdicts** to re-aggregate at
L'∈{1,3,5,7,9}. The current result JSONs only carry the aggregated utility
matrix. To produce this:

    # Re-run preference cells with the audit-log option turned on, e.g.
    python -m scripts.run_h1 --model tulu3-8b-rlvr --dataset ultrafeedback \
        --seed 0 --K 32 --num-prompts 1000 --max-tokens 512  # writes
        # logs/verifier/ultrafeedback_log.jsonl with raw_verdicts per row

Then a small post-processor over that JSONL produces the C.2 table.

## D.1 — MATH verifier failure-mode rates
Needs the **failure-cause** for each `U=0` verification (one of
{no-boxed, empty-boxed, parse-timeout, verify-timeout, internal-exception,
incorrect}). The current verifier returns only the float utility. To
produce this:

  1. Instrument `src.verification.math.verify` to optionally return a
     reason string alongside `0.0` (no behavioural change at K=64).
  2. Re-run the verification pass over the cached samples for the three
     headline (model, MATH) cells; aggregate the reasons.
  3. The re-verifier here is identical to the one we used to fix the 72B
     MATH cell (see `_local_backups/rerun_72b_math.py`); just add reason
     capture.

## D.2 — Position bias / inter-rater agreement (self-judge cells)
Needs the per-(prompt, candidate, l) raw verdicts and the
`a_is_candidate` flag, currently dropped after aggregation. Same fix as
C.2: re-run with audit logging, then post-process.

## D.3 — GSM8K extractor pattern-firing rates
Needs to record which of the four GSM8K extractor patterns matched each
generation. Smallest patch: add an optional `with_reason=True` return mode
to `src.verification.gsm8k.verify`, then re-extract on the cached samples
for the three headline (model, GSM8K) cells.

## G — Failure-case traces
Qualitative. The pipeline outputs `G_candidates.tex` listing the
candidate-prompt counts per cell; selecting which 4-5 traces to show in
the paper is editorial. To populate the traces, pull the sample cache
for the relevant cell and pick the lowest-k correct sample + a
representative incorrect sample.
