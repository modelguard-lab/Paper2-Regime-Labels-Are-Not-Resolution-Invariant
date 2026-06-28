# Paper 2 -- companion notebooks

One notebook per CLI command. Each notebook walks through the experiment
end-to-end as a sequence of code cells calling the actual `src/`
functions: load raw 5m OHLC, build features, fit GMM regimes per
frequency, align labels to the 5m time axis, run the experiment-specific
computation, and finally compare the in-process result to the cached
full-scale CSV under `outputs/`. For long-running steps (1,000-iter
bootstrap, 1,000-rep simulation, full 4-asset GMM sweep) the in-process
call uses a reduced parameter set that keeps notebook runtime under
~60 s; the cached CSV preserves the paper number.

## Run order and mapping to paper

| Notebook | Paper artifact | Re-run command |
| --- | --- | --- |
| [00_data_quality.ipynb](00_data_quality.ipynb)               | Raw 5m OHLC sanity check (no paper artifact)                                | `python run.py pipeline --download` (then `--validate`) |
| [01_main_pipeline.ipynb](01_main_pipeline.ipynb)             | Tables 1-2, Fig. 1, Tables A.1-A.14 (cross_freq, ari_matrix, decomposition, perm, robustness, gmm_diag, expanding, block_perm*, window_sweep, cl_roll, OOS 2022) | `python run.py pipeline` |
| [02_extended_majority_vote.ipynb](02_extended_majority_vote.ipynb) | Table A.15 (majority-vote upward aggregation)                          | `python run.py extended_majority_vote` |
| [03_extended_bootstrap.ipynb](03_extended_bootstrap.ipynb)   | Table A.16 (1,000-window bootstrap)                                         | `python run.py extended_bootstrap` |
| [04_extended_hypothesis_tests.ipynb](04_extended_hypothesis_tests.ipynb) | Kruskal-Wallis / Mann-Whitney prose (Appendix S.3)                | `python run.py extended_hypothesis_tests` |
| [05_extended_simulation.ipynb](05_extended_simulation.ipynb) | Markov-Gaussian RSS calibration (Supplementary)                             | `python run.py extended_simulation` |
| [06_extended_calm_subsample.ipynb](06_extended_calm_subsample.ipynb) | Table A.17 (calm-day subsample, Q7)                                  | `python run.py extended_calm_subsample` |
| [07_extended_var_uplift.ipynb](07_extended_var_uplift.ipynb) | Table A.18 (resolution-conditional VaR uplift, Q9)                          | `python run.py extended_var_uplift` |
| [08_stress_vs_calm.ipynb](08_stress_vs_calm.ipynb)           | Table A.19 (paired stress-vs-calm test)                                     | `python run.py stress_vs_calm` |
| [11_cross_asset_figure.ipynb](11_cross_asset_figure.ipynb)   | Cross-asset rolling-ARI resonance figure (Supplementary)                    | `python run.py cross_asset` |

**Note:** experiments exp\_11 through exp\_18 (GARCH-MS calibration, K$\times$window sweep, K=3 baseline, GLD block sweep, disagree-day config, asymmetric persistence, EM-restart placebo, per-asset full ML baseline) do not have a dedicated companion notebook. They can be re-run individually via `python run.py extended_<command>` or collectively via `python run.py extended`; outputs land in `outputs/` alongside the pipeline CSVs.

## Re-executing the notebooks

The `.ipynb` files are hand-maintained; edit them directly in JupyterLab,
VSCode, or via `nbformat`. To re-execute every notebook in place and
embed the resulting tables / figures as cell outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

## Why these notebooks exist

The reproduction script ([`reproduce_paper.sh`](../reproduce_paper.sh))
regenerates every output CSV and figure as a single batch. The notebooks
complement it by walking through the experiment one step at a time on a
focal asset, exposing the intermediate state (feature matrix, regime
label distribution, ARI matrix, permutation null) that the headline
numbers depend on. A reviewer can audit any specific number from the
paper without reading the orchestration code.
