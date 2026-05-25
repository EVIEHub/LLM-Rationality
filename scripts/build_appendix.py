"""Generate the auto-fillable LaTeX fragments for ``drafts/appendix.tex``.

Run from the repo root:

    python -m scripts.build_appendix --results-dir ~/rational_gap_outputs/results

Outputs land in ``drafts/appendix_inputs/`` — one ``.tex`` file per section
(e.g.\\ ``C1_saturation.tex``). The master ``drafts/appendix.tex`` ``\\input``-s
each. Sections that need extra experiments / instrumented re-runs are recorded
in ``drafts/appendix_inputs/PENDING.md``.

Pass ``--inline`` to also emit ``drafts/appendix_inline.tex`` — a single,
self-contained appendix with every ``\\input{appendix_inputs/...}`` line
substituted by the corresponding fragment's contents. Convenient for
Overleaf / single-file submissions: upload one file, no directory tree
needed; re-run the script anytime to regenerate the inline copy from the
latest numbers. The inlined fragments are wrapped in
``% AUTO-INLINED FROM ...`` / ``% END AUTO-INLINE`` markers so diffs stay
readable.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.plotting.appendix import (
    build_B4_gpu_hours, build_C1_saturation, build_C2_L_sensitivity,
    build_C3_M_convergence, build_C4_epsilon, build_D1_math_failures,
    build_D2_position_bias, build_D3_gsm8k_patterns,
    build_E2_per_difficulty, build_G_candidates,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


_PENDING = r"""# Appendix sections still needing extra experiments

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

## C.2 / D.2 — Self-judge audit-log re-aggregation
Both sections now auto-fill IFF the self-judge audit log at
`outputs/logs/verifier/ultrafeedback_log.jsonl` exists with the
`a_is_candidate` flag baked in (Phase-2 patch to
`scripts/run_h1.py` and `src/verification/self_judge.py`,
2026-05-24). If the log is missing or pre-Phase-2, the builders emit
a one-line `% NOTE` stub explaining what's needed.

## G — Failure-case traces
Qualitative. The pipeline outputs `G_candidates.tex` listing the
candidate-prompt counts per cell; selecting which 4-5 traces to show in
the paper is editorial. To populate the traces, pull the sample cache
for the relevant cell and pick the lowest-k correct sample + a
representative incorrect sample.
"""


_INPUT_RE = re.compile(r"\\input\{appendix_inputs/([^}]+)\}")


def _build_inline(appendix_path: Path, fragments_dir: Path,
                  out_path: Path) -> None:
    """Substitute every ``\\input{appendix_inputs/X}`` in ``appendix_path``
    with the contents of ``fragments_dir/X`` and write to ``out_path``.

    Fragments not on disk are left as the original ``\\input{}`` line
    plus a ``% WARNING: fragment missing`` marker, so the inline file
    still compiles if the user later replaces the marker by hand.
    """
    lines: list[str] = []
    fragments_used: list[str] = []
    fragments_missing: list[str] = []
    src = appendix_path.read_text().splitlines()
    for ln in src:
        m = _INPUT_RE.search(ln)
        if not m:
            lines.append(ln)
            continue
        frag_name = m.group(1)
        frag_path = fragments_dir / frag_name
        if not frag_path.exists():
            lines.append(f"% WARNING: fragment missing: {frag_path}")
            lines.append(ln)
            fragments_missing.append(frag_name)
            continue
        lines.append(f"% ===== AUTO-INLINED FROM appendix_inputs/{frag_name} =====")
        lines.append(frag_path.read_text().rstrip())
        lines.append(f"% ===== END AUTO-INLINE ({frag_name}) =====")
        fragments_used.append(frag_name)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  inlined {len(fragments_used)} fragment(s) into {out_path}")
    for name in fragments_used:
        print(f"    + {name}")
    for name in fragments_missing:
        print(f"    ! {name}  (MISSING — left as \\input)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir",
                   default=str(_REPO_ROOT / "drafts" / "appendix_inputs"))
    p.add_argument(
        "--inline", action="store_true",
        help="Also emit drafts/appendix_inline.tex — a single self-"
             "contained file with every \\input{appendix_inputs/...} "
             "substituted in-place. Convenient for Overleaf / "
             "single-file submissions.",
    )
    p.add_argument(
        "--appendix-tex",
        default=str(_REPO_ROOT / "drafts" / "appendix.tex"),
        help="Path to the appendix template used by --inline (default: "
             "drafts/appendix.tex).",
    )
    p.add_argument(
        "--inline-out",
        default=str(_REPO_ROOT / "drafts" / "appendix_inline.tex"),
        help="Output path for --inline (default: "
             "drafts/appendix_inline.tex).",
    )
    args = p.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir).expanduser()
    else:
        from src.pipeline.paths import load_paths
        results_dir = load_paths().results_dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sections = [
        ("B4_gpu_hours.tex",    build_B4_gpu_hours(results_dir, args.seed)),
        ("C1_saturation.tex",   build_C1_saturation(results_dir, args.seed)),
        ("C2_L_sensitivity.tex", build_C2_L_sensitivity(results_dir, args.seed)),
        ("C3_M_convergence.tex", build_C3_M_convergence(results_dir, args.seed)),
        ("C4_epsilon.tex",      build_C4_epsilon(results_dir, args.seed)),
        ("D1_math_failures.tex", build_D1_math_failures(results_dir, args.seed)),
        ("D2_position_bias.tex", build_D2_position_bias(results_dir, args.seed)),
        ("D3_gsm8k_patterns.tex", build_D3_gsm8k_patterns(results_dir, args.seed)),
        ("E2_per_difficulty.tex", build_E2_per_difficulty(results_dir, args.seed)),
        ("G_candidates.tex",    build_G_candidates(results_dir, args.seed)),
    ]
    for name, body in sections:
        (out_dir / name).write_text(body)
        print(f"  wrote {out_dir / name}")
    (out_dir / "PENDING.md").write_text(_PENDING)
    print(f"  wrote {out_dir / 'PENDING.md'}")

    if args.inline:
        _build_inline(
            appendix_path=Path(args.appendix_tex),
            fragments_dir=out_dir,
            out_path=Path(args.inline_out),
        )


if __name__ == "__main__":
    main()
