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

## D.2 — Position bias / inter-rater agreement (self-judge cells)
Needs the per-(prompt, candidate, l) raw verdicts and the
`a_is_candidate` flag, currently dropped after aggregation. Same fix as
C.2: re-run with audit logging, then post-process.

## G — Failure-case traces
Qualitative. The pipeline outputs `G_candidates.tex` listing the
candidate-prompt counts per cell; selecting which 4-5 traces to show in
the paper is editorial. To populate the traces, pull the sample cache
for the relevant cell and pick the lowest-k correct sample + a
representative incorrect sample.
