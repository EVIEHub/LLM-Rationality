# Drafts

Sandbox area for exploratory experiments that are **not** part of the
formal H1-H4 reproducibility kit. Anything here can be deleted without
affecting paper results. Treat as scratchpad.

## Layout

```
drafts/
├── README.md                    # this file
├── notebooks/                   # Jupyter notebooks for exploration
│   └── helpsteer2_h1_toy.ipynb  # toy-scale H1 with preference-RM utility
└── outputs/                     # pickles / plots / artifacts from notebooks
                                 # (gitignored — re-rendered on demand)
```

## Naming convention

Notebooks: `{topic}_{hypothesis}_{scale}.ipynb` (e.g. `helpsteer2_h1_toy.ipynb`).
Tag obviously-exploratory work with `toy` or `draft` in the filename
so it's clear at a glance vs. the formal panels in
`scripts/development_exp/` and `scripts/h5_exp/` (the H5 deployment battery).

## Promoting drafts to the formal kit

When a draft hardens into a paper experiment:

1. Convert notebook → CLI cell runner under `scripts/run_*.py`
2. Add a panel script `scripts/{development,deployment}_exp/run_*_panel.sh`
3. Add tests under `tests/test_*.py`
4. Move the original notebook to `drafts/notebooks/_archived/` (preserves
   the exploration history; ungitignore as needed)

## Caveats

- Notebooks **may use ad-hoc paths and skip the methodology rules**
  (paper-symbol-only naming, audit logging, etc.) that bind the formal
  pipeline. Anything that gets promoted needs to be brought up to
  spec.
- Outputs are gitignored by `outputs/` rule already covering this dir.
