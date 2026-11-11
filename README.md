# Nature-inspired metaheuristics under matched budgets (v2)

Benchmarking study of six recent nature-inspired optimisers (GWO, WOA, SCA, SSA, HHO, AOA) against five
reference algorithms (DE, PSO, L-SHADE, CMA-ES, random search) on 19 constrained problems, with every
algorithm run through one shared evaluator so that the evaluation budget, constraint handling and random
seeds are identical. Third-party (mealpy) versions of the same algorithms and six further algorithms
(MFO, FFA, BA, EO, SMA, MPA) are run through the same evaluator to separate algorithm effects from
implementation effects.

## Contents

| Path | Purpose |
|---|---|
| `src/problems.py`, `src/cec_problems.py` | 9 mechanical-design problems and 10 CEC 2006 problems (G01-G04, G06-G10, G12) |
| `src/algorithms.py` | shared `Evaluator` (budget metering, repair, constraint schemes, anytime trace) and all 23 algorithms |
| `src/runner.py` | parallel, resumable experiment driver |
| `src/analysis.py` | statistics: Friedman/Nemenyi, paired Wilcoxon with Holm, A12, bootstrap, rank stability |
| `src/make_figures.py` | Figures 1-9 (PDF and PNG) |
| `src/audit_discbrake.py` | exhaustive enumeration of the clutch-brake design space (7.1 million designs) |
| `src/verify_problems.py` | checks that every problem reproduces its published optimum |
| `src/timing.py` | per-run computational overhead of each algorithm |
| `tests/test_core.py` | automated checks of the experimental machinery |
| `results/` | raw per-job results (JSON), tuned settings, analysis output |
| `figures/` | final figures |

## Environment

Python 3.13 on Windows 11 (8 cores). Exact versions are pinned in `requirements.txt`.

```
pip install numpy==2.1.2 scipy==1.14.1 pandas==2.2.3 matplotlib==3.9.2 cma==4.5.0 pytest
pip install --no-deps mealpy==3.0.3
```

`mealpy` declares `numpy<=1.26` but runs correctly on numpy 2.1.2 for the algorithms used here, so it is
installed without its dependency pins. Without mealpy the 12 third-party algorithms are unavailable
and `runner.py main` will fail on them.

## Reproducing the results

All commands are run from `src/`. Every job writes one file under `results/<phase>/` and is skipped if the
file exists, so any interrupted run can simply be repeated. Set `WORKERS=n` to change the process count
(default 7).

```
python verify_problems.py          # 19/19 problems reproduce their published optimum
python runner.py main              # default settings, 23 algorithms x 19 problems (Deb rules)
python runner.py tune 32           # racing tuner, 32 initial configurations per algorithm
python runner.py tune 128          # same with 128 (tests whether the tuning budget was enough)
python runner.py tuned 32          # core algorithms with tuned settings
python runner.py tuned 128
python runner.py cons              # static-penalty and epsilon-constrained schemes
python runner.py budget            # 5,000 / 50,000 / 150,000 evaluations
python runner.py status            # progress of every phase
python analysis.py                 # results/analysis.json and results/tables/*.csv
python make_figures.py             # figures/Figure_*.{pdf,png}
python timing.py                   # run on an otherwise idle machine
python audit_discbrake.py          # clutch-brake audit
python -m pytest ../tests -q       # automated checks
```

`bash run_rest.sh` chains every phase after `main`. The file `results/ALL_PHASES_COMPLETE` appears when
all phases are done. A second runner started while one is active waits for the first (file lock).

## Protocol in brief

* **Budget.** 15,000 function evaluations per run. The shared evaluator returns infinity once the budget is
  spent, so every algorithm, including third-party ones whose own counters overshoot, receives exactly
  15,000 evaluations (checked by `tests/test_core.py`).
* **Seeds.** Run `r` on problem index `j` uses seed `10000*j + r` for every algorithm, so comparisons are
  paired by seed. Tuning seeds start at 777000 and never overlap evaluation seeds.
* **Runs.** 51 per cell for in-house algorithms and baselines; 31 for the third-party algorithms
  (about 40 times slower per run); 31 for constraint-scheme sensitivity; 15 for budget sensitivity.
* **Tuning.** Successive-halving racing on four training problems (PressureVessel, CantileverBeam, G04, G08);
  the other 15 problems are held out.
* **Normalised gap.** `(f - f*) / (median_RS - f*)` where `RS` is random search and `f*` the published
  optimum. 0 means the published optimum was reached, 1 means random search's median, and a run with no
  feasible solution scores 10. **Where random search is feasible in fewer than half of its runs
  (G01, G03, G07, G10), the scale falls back to `|f*|`**, a relative error. The rule is fixed in advance
  and independent of every compared algorithm.
* **Statistics.** Friedman test on per-problem medians with the Nemenyi critical difference, paired
  Wilcoxon tests with Holm correction, Vargha-Delaney A12, bootstrap intervals on ranks, Kendall tau for
  rank stability across schemes and budgets.

## Known limitations

* Third-party algorithms use 31 runs per cell, against 51 for the others.
* GearTrain's random-search scale is about 2e-9, so its normalised gap is dominated by clipping at 10;
  rank-based results are unaffected, but averages of gaps on that problem should be read with care.
* The tuner is a simple racing method and is weaker than irace; the 32 versus 128 configuration comparison
  is reported to show how sensitive the conclusions are to it.
* Results were produced on one machine; timings are indicative only.

## Clutch-brake correction

An earlier version used mass `pi (ro^2 - ri^2) t Z rho`. The published formulation uses `(Z + 1)`.
`audit_discbrake.py` enumerates all 7.1 million discrete designs: with `(Z + 1)` the exact optimum is
0.2352425 at (ri, ro, t, F, Z) = (70, 90, 1, 964, 2), matching the published value, so the earlier
"improvement" below the published value was a transcription error.

## Licence and citation

MIT licence (`LICENSE`; the copyright holder line still needs the author's name). Cite the archived release
once it has a DOI.
