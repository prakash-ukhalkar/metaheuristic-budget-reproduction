# Budget-Controlled Reproduction Study of Nature-Inspired Metaheuristics

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-pre--submission-yellow)

A reproduction study comparing six metaphor-based metaheuristics against five established
baselines on constrained engineering design problems, under matched evaluation budgets, matched
tuning effort, and identical constraint handling. Target venue: **Engineering Research Express**
(IOP Publishing).

## Contents

- [Overview](#overview)
- [Key results](#key-results)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Reproducing the results](#reproducing-the-results)
- [Notebooks](#notebooks)
- [Data availability](#data-availability)
- [Status and known limitations](#status-and-known-limitations)
- [Citation](#citation)
- [License](#license)

## Overview

Metaphor-based metaheuristics (e.g. algorithms inspired by animal behaviour, natural phenomena,
or social processes) are frequently reported to outperform established optimisation baselines.
This study re-evaluates that claim under conditions designed to remove the most common sources of
an unfair comparison:

- **Matched evaluation budget** — every algorithm is metered through the same `Evaluator` object
  (15,000 function evaluations, 51 seeded runs per algorithm-problem pair).
- **Matched tuning effort** — hyperparameters are tuned by an identical successive-halving racing
  protocol per algorithm, not hand-picked per method.
- **Paired seeding** — run `r` on problem `j` uses the same seed for every algorithm, enabling
  paired statistical tests.
- **Multiple constraint-handling schemes** — Deb feasibility rules, static penalty, and
  epsilon-constrained, to check that conclusions are not an artefact of one scheme.
- **Independent fidelity check** — the six metaphor-based implementations are cross-validated
  against an independent third-party library (`mealpy`).

Six metaphor-based algorithms (GWO, WOA, SCA, SSA, HHO, AOA) are compared against five baselines
(DE, PSO, L-SHADE, CMA-ES, random search) on nine constrained engineering design problems.

## Key results

Full statistics are in [results/analysis.json](results/analysis.json) and
[results/headline_v2.json](results/headline_v2.json); figures are in [figures/](figures/).

- **Friedman test across the 8 admitted problems**: χ² = 50.97, p = 1.8 × 10⁻⁷ — highly
  significant differences among the 11 algorithms.
- **Mean rank, all problems (lower is better)**: DE (2.13) < L-SHADE (2.38) < CMA-ES (3.06) <
  GWO (4.5) < WOA (5.88) < PSO (6.5) < SCA (7.25) < SSA (7.69) < AOA (8.25) < RS (8.63) <
  HHO (9.75). The three top-ranked algorithms are established baselines, not metaphor-based
  methods.
- **Paired Wilcoxon (Holm-corrected), metaphor-based vs. baselines**: 150 losses, 55 ties, only
  35 wins out of 240 comparisons for the metaphor-based group (median A₁₂ = 0.92).
- **Ranking is stable** across three constraint-handling schemes and three evaluation budgets
  (5,000 / 15,000 / 50,000).
- **Implementation fidelity**: no statistically significant difference from an independent
  (`mealpy`) implementation on 19 of 48 algorithm-problem pairs; overall median relative
  difference in reported objective values is 0.06.

## Repository layout

```
src/                    Python source (see below)
notebooks/              Annotated Jupyter notebook version of every script in src/
results/                Raw run data (JSON), summary tables (CSV), aggregated analysis (JSON)
figures/                Manuscript figures (fig1-fig8), PNG + PDF
```

The manuscript source is submitted directly to the journal and is not included in this
repository.

| Script | Purpose |
|---|---|
| `src/problems.py` | 9 constrained design problems, vectorised, with repair operators |
| `src/verify_problems.py` | Checks every problem against its published optimum |
| `src/algorithms.py` | 11 algorithms + shared `Evaluator` (budget metering, 3 constraint-handling schemes, anytime traces) |
| `src/runner.py` | Experiment driver: `tune` \| `main` \| `tuned` \| `cons` |
| `src/fidelity.py` | Cross-validation against the independent `mealpy` library |
| `src/analysis.py` | Friedman/Nemenyi, paired Wilcoxon + Holm, A₁₂, early figure set |
| `src/extra_analysis.py` | Headline statistics for the revised manuscript |
| `src/make_figures.py` | Final, manuscript-numbered figures (1-8) |
| `src/figures_extra.py` | Standalone regeneration of the design schematic, tuning and boxplot figures |
| `src/verify_references.py` | Resolves every manuscript reference against the Crossref API |

## Installation

```bash
pip install -r requirements.txt
```

Requires `numpy`, `scipy`, `pandas`, `matplotlib`, `cma`, `scikit-posthocs`, and `mealpy` (for the
fidelity check only).

## Reproducing the results

Two admission checks run before any experiment: the published optimum must reproduce, and no
algorithm may be shown to beat the published value in a short search (both implemented in
`src/verify_problems.py`).

```bash
cd src
python3 verify_problems.py
python3 runner.py tune && python3 runner.py main \
  && python3 runner.py tuned && python3 runner.py cons
python3 fidelity.py
python3 extra_analysis.py && python3 make_figures.py
```

Set `RESUME=1` to resume a partially completed sweep; results are written after every problem.

Seeds are fixed: run `r` on problem `j` uses seed `10000*j + r` for every algorithm, so all
cross-algorithm comparisons are paired.

**Configuration used for the reported results**: evaluation budget 15,000; 51 runs per
algorithm-problem pair; hyperparameters tuned by successive-halving racing (32 configurations
raced on 3 training problems).

## Notebooks

An annotated Jupyter notebook is provided for every script in `src/`, under
[notebooks/](notebooks/), with section-by-section explanations and a findings summary per
notebook based on the results in this repository. See [notebooks/README.md](notebooks/README.md)
for the full index and suggested run order. Notebooks reproduce the same code as the
corresponding `.py` file; no logic was changed in the conversion.

## Data availability

All raw run data (`results/*.json`), summary tables (`results/*.csv`) and figures (`figures/`)
generated for the reported results are included in this repository. The repository will be
archived on Zenodo prior to submission and the DOI added here and in the manuscript's data
availability statement.

## Status and known limitations

**Completed**:
- 8-problem suite (1 excluded, see below), dual admission checks per problem
- 51 runs per algorithm-problem pair
- Successive-halving racing tuner, identical protocol per algorithm
- Implementation-fidelity validation against `mealpy` across 48 algorithm-problem pairs
- Budget sensitivity at 5,000 / 15,000 / 50,000 evaluations
- Constraint-handling sensitivity across 3 schemes
- Computational overhead measured; statistical power limitation of Friedman-Nemenyi stated in text

**Excluded**: the multiple-disc clutch brake problem. Feasible solutions ~33% better than its
published optimum are found readily, indicating the formulation as commonly circulated is
under-constrained (manuscript Section 4.6).

## Citation

See [CITATION.cff](CITATION.cff). Please cite the associated article once published; until then,
cite this repository directly.

## License

Released under the [MIT License](LICENSE).
