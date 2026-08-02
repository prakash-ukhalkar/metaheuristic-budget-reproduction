# Notebooks

Jupyter notebook versions of every script in `src/`, generated for interactive
exploration. Each notebook mirrors its source script cell-by-cell with added
markdown explanations; no logic was changed relative to the original `.py`
files. Run notebooks from this folder (or add `src/` to `sys.path`) so that
project-local imports (`problems`, `algorithms`, `analysis`, ...) resolve.

| Notebook | Source script | Purpose |
|---|---|---|
| [01_problems.ipynb](01_problems.ipynb) | `src/problems.py` | Constrained engineering design problem suite |
| [02_algorithms.ipynb](02_algorithms.ipynb) | `src/algorithms.py` | Algorithm implementations and shared evaluator |
| [03_verify_problems.ipynb](03_verify_problems.ipynb) | `src/verify_problems.py` | Admission check: published optimum reproduction |
| [04_runner.ipynb](04_runner.ipynb) | `src/runner.py` | Experiment driver (tune / main / tuned / cons) |
| [05_fidelity.ipynb](05_fidelity.ipynb) | `src/fidelity.py` | Implementation-fidelity validation against mealpy |
| [06_analysis.ipynb](06_analysis.ipynb) | `src/analysis.py` | Statistical analysis and manuscript figures |
| [07_extra_analysis.ipynb](07_extra_analysis.ipynb) | `src/extra_analysis.py` | Headline statistics for the revised manuscript |
| [08_make_figures.ipynb](08_make_figures.ipynb) | `src/make_figures.py` | Final manuscript figure generation (figs 1-8) |
| [09_figures_extra.ipynb](09_figures_extra.ipynb) | `src/figures_extra.py` | Supplementary figure regeneration pass |
| [10_verify_references.ipynb](10_verify_references.ipynb) | `src/verify_references.py` | Reference DOI verification via Crossref |

Suggested run order: 01 -> 02 -> 03 -> 04 -> 05 -> 06/07 -> 08/09 -> 10.
