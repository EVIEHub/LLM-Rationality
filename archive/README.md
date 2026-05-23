# archive/

Superseded files kept for safety (recoverable here AND in git history).
Nothing here is run by the pipeline.

## legacy_scope_panels_2026-05-22/
The old scope-split orchestration: `development_exp/`, `preference_exp/`,
`h5_exp/`, each with per-hypothesis `run_h*_panel.sh`. These were merged into
ONE script per hypothesis at `scripts/run_h{1,2,3,4,5}_panel.sh` (each now
spans all scopes: development + deployment + preference). Kept in case a
detail (param, comment, edge case) needs to be recovered.
