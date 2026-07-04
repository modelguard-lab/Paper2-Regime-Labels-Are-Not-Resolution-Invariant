# Paper 2: Regime Labels Are Not Resolution-Invariant

Multi-frequency (5m / 15m / 1h / 1d) regime-detection experiment on the same event window.

**Structure** aligned with **Regime Labels Are Not Representation-Invariant**:

- `run.py`: main entry (ensures 5m data via IB if missing; then runs the multi-frequency pipeline). Accepts optional CLI overrides.
- `config.yaml`: `raw_dir`, `outputs_dir`, `assets`, `episode`, and `ib` (host, port, dates, future_expiry).
- `data/`: raw 5m CSVs (`<symbol>_5m.csv`).
- `outputs/`: generated results tables and plots.
- `src/`: code: `data/data_ib.py` (IB download), `workflows/pipeline.py` (analysis), `commands/cli_main.py` (CLI dispatcher).
- `paper/`: LaTeX manuscript sources (`main.tex`, `refs.bib`); build with `paper/build_paper.bat`.

## Run

```bash
pip install -r requirements.txt
python run.py
```

- **Default:** ensure 5m data (download via IB if any asset is missing), then run the pipeline and write outputs under `outputs/`.
- **`--download`:** only download 5m data via IB and exit (for future multi-purpose: pipeline, report, etc.).

### One-button reproduction

```bash
python run.py all           # both episodes (2026 + 2022 OOS) + extended + standalone
```

`run.py all` runs the 2026 pipeline, the 2022 OOS replica (when `data_2022/` is present), then every extended/standalone experiment in sequence; the **Experiments** section below documents what each sub-experiment does, the method, the headline result, and the output files. Each is also individually CLI-registered so reviewers can re-run a single one (e.g. `python run.py extended_bootstrap`).

### `notebooks/` vs CLI

The `notebooks/` directory is the **exploratory view** only. Each notebook reads from `outputs/*.csv` already produced by the CLI and renders figures / sanity checks for that single experiment. They do not regenerate the underlying CSVs and are not part of the reproduction pipeline. For paper-grade replication always run `python run.py all` (or a specific `python run.py <command>`); use the notebooks only to inspect intermediate artefacts after a CLI run.

### Experiments

The pipeline is **layered**. Tier 1 produces the headline numbers; Tier 2 supplies DGP baselines and parameter-sensitivity sweeps that contextualise them; Tier 3 is post-processing paper artefacts (referee-response material lives here too); Tier 4 is dev helpers not in the reproduce chain. The `extended` orchestrator (`python run.py extended`) runs every Tier 2/3 sub-experiment in sequence; each is also independently CLI-registered so reviewers can re-run a single one. The 2022 OOS episode shares the same commands; outputs land in `outputs_2022/` when `data_2022/` is present.

**Headline result the layers support:** empirical mean off-diagonal cross-frequency ARI is 0.11–0.39 across the current 2026 panel (0.13–0.15 in 2022 OOS), with a structural break at the 15m→1h boundary; the calibrated K=2 Markov-switching baseline is 0.113 (IQR 0.099–0.125). All five 2026 assets exceed the intraday-only calibrated upper bound (0.195 IQR [0.177, 0.213]) by +0.03 (USD/JPY) to +0.24 (CL).

#### Tier 1: Main pipeline

| Command | Purpose | Method | Headline result | Outputs |
| --- | --- | --- | --- | --- |
| `pipeline` | Produce the cross-frequency ARI/AMI/VI tables, GMM diagnostics, regime timelines, expanding-window estimates, block-permutation p-values, calendar-window robustness, HMM parallel, time-of-day adjustment, and rolling-7d trace for one episode. | Independent two-component GMMs on log(rolling RV) at 5m / 15m / 1h / 1d; align via forward-fill; ARI on every off-diagonal pair; default block size 50 5m bars for the permutation test. | 4×4 ARI matrices per asset; mean off-diag 0.11–0.39 (2026, five assets), 0.13–0.15 (2022 OOS); 15m→1h drop is the largest single-step fall in four of five 2026 assets; block-perm p < 0.002 at every block size in [25, 250]. **Paper:** Tables 1–2, Fig. 1, Tables A.1–A.5, A.8–A.10, A.13, A.14. | `outputs/<asset>_*.csv` (≈25 files/asset), `<asset>_timeline.png`, `pipeline_summary.csv`, `run.log` |

For the 2022 OOS replica run: `python run.py pipeline --episode 2022_ukraine --raw-dir data_2022 --outputs-dir outputs_2022`.

#### Tier 2: DGP baselines and robustness sweeps

| Command | Purpose | Method | Headline result | Outputs |
| --- | --- | --- | --- | --- |
| `extended_simulation` | Establish the K=2 calibrated reference ARI under a known regime-switching DGP. | Two-state Markov-Gaussian DGP at calibrated and three sensitivity persistence settings (P=0.005 / 0.003 / 0.001, durations ≈3.3 / 5.5 / 17 h); 200 reps × 100k 1-min bars; aggregate to 5m / 15m / 1h / 1d; fit GMMs; compute null (permuted) and alternative ARI. | Calibrated baseline all-4 ARI: **0.113 (IQR [0.099, 0.125])**; intraday-only baseline 0.195 (IQR [0.177, 0.213]); empirical 0.11–0.39 range spans 0.11–0.39 vs calibrated 0.113 mean. **Paper:** §app:rss_sim. | `outputs/simulation_rss.csv`, `simulation_detection_rate.csv` |
| `extended_garch_ms` | Rule out i.i.d. Gaussian assumption as the load-bearing baseline driver. | Replace within-regime innovations with GARCH(1,1) (α=0.05, β=0.90); same Markov chain, same 200×100k spec. | Baseline tracks Gaussian within 0.02 (all-4) / 0.04 (intraday); volatility clustering is **not** load-bearing. **Paper:** §app:garchms_sim. | `outputs/simulation_rss_garchms.csv` |
| `extended_k3_baseline` | Test whether K=3 collapses the empirical-vs-baseline gap. | Refit K=3 GMM on the same 1-min paths, binarise via argmax-of-means; 500 reps; same persistence sweep. | K=3 baseline 0.25–0.33; empirical K=3 panel mean 0.21 still undershoots by **0.04–0.12** (vs K=2 gap 0.14–0.25); CL 2026 **exceeds** the K=3 upper bound (third state absorbs supply-shock bursts). **Paper:** Suppl Table `tab:k3_baseline`, §app:k3_baseline. | `outputs/simulation_rss_k3.csv` |
| `extended_asym_baseline` | Address simulated R2 M2: symmetric P=P fixes stationary crisis share at 50%, but empirical 1h crisis shares span 16.3%–70.0%. | Per-asset DGP with τ=P+P fixed and stationary share π matched to empirical 1h share; 200 reps × 20k bars. | Per-asset baseline 0.111–0.134 (within or just above the symmetric ML-fit IQR). Per-asset 4-frequency gap: SPY +0.07, QQQ +0.04, USD/JPY -0.03, CL +0.28, GLD +0.04. Cross-frequency ARI is governed by 1/τ, not by π; relaxing symmetry **does not absorb** the empirical-vs-calibrated shortfall. **Paper:** Suppl Table `tab:asym_baseline`, §app:asym_baseline. | `outputs/simulation_rss_asym_persistence_1h_anchor.csv`, `outputs/simulation_rss_asym_persistence_1d_anchor.csv` |
| `extended_kxw_sweep` | Bound the headline ARI under parameter perturbations of K and rolling-volatility window length. | Wraps `pipeline.run_robustness` with K ∈ {2,3} × window-scale ∈ {0.5×, 1×, 2×} for the five-asset 2026 panel and three-asset 2022 OOS panel. | 15m→1h structural break preserved at every cell; ARI envelope and worst-case bounds reported. **Paper:** Suppl Tables A.6 `tab:robustness`, A.7 `tab:robustness_extremes`, A.11 `tab:window_sweep`. | `outputs/robustness_summary.csv`, `robustness_ranges.csv`, `robustness_report.md` (and `outputs_2022/` mirrors) |
| `extended_block_sweep_gld` | Fill the GLD row of the block-size sweep table (other four rows ship with the main pipeline at default block size 50). | Block-permutation p-value on GLD aligned-regime labels at block sizes {25, 50, 100, 250} 5m bars. | Closes the 4×4 sweep grid that the pipeline starts. **Paper:** Suppl Table A.13 `tab:block_sweep` (GLD row). | `outputs/gld_block_sweep.csv` |
| `extended_block_sweep_assets` | Fill the SPY/QQQ/CL/USD-JPY rows of the block-size sweep table at all four block sizes. | Block-permutation p-value on each asset's aligned-regime labels at block sizes {25, 50, 100, 250} 5m bars; seed=42, n_perm=500. | Completes the 5-asset × 4-block-size grid alongside `extended_block_sweep_gld`. **Paper:** Suppl Table A.13 `tab:block_sweep` (SPY/QQQ/CL/USD-JPY rows). | `outputs/block_sweep_assets.csv` |

#### Tier 3: Post-processing paper artefacts

Consume main-pipeline outputs (or Tier-2 simulation outputs). Round 1 referee responses live here next to their peers; grouping by function avoids splitting peers across "round" buckets.

| Command | Purpose | Method | Headline result | Outputs |
| --- | --- | --- | --- | --- |
| `extended_majority_vote` | Symmetric counterpart to the forward-fill baseline (does the empirical ARI level depend on aggregation direction?). | Fine→coarse majority-vote upward aggregation; ARI against the coarser-grid GMM labels. | Combined-episode range **0.09–0.42**, comparable to forward-fill (0.11–0.39); aggregation direction does not change the empirical level. **Paper:** Suppl Table A.15 `tab:majority_vote`. | `outputs/<asset>_majority_vote_cross_freq_ari.csv`, `majority_vote_summary.csv` |
| `extended_bootstrap` | Quantify sampling variability of the mean off-diag ARI; test whether a specific calm/stress 5-day window is unusual. | 1,000 random 5-day windows drawn uniformly from the calendar; GMM fitted once on full sample. | Across 16 calm/stress vs bootstrap comparisons (8 asset×episode), p ∈ **[0.08, 0.99]**, median p = 0.59; only SPY 2026 stress drops below 0.10. Calm and stress windows are not statistically unusual draws. **Paper:** Suppl Table A.16 `tab:bootstrap_5d`. | `outputs/bootstrap_five_day_windows.csv` |
| `extended_hypothesis_tests` | Test whether the cross-resolution ARI level differs by frequency-pair category. | KW + Mann–Whitney U on pooled off-diag ARI grouped into adjacent intraday / non-adjacent intraday / intraday-daily. | Significant difference across categories driven by the 15m→1h break; consistent with the structural-pair claim. **Paper:** Suppl §S.3 prose. | `outputs/hypothesis_tests.csv` |
| `extended_calm_subsample` | Referee R1 Q7: do results hold under "normal" market conditions? | Subset to days with daily RV ≤ in-sample median **and** outside the peak-stress window; recompute ARI with full-sample GMM boundaries fixed. | Calm-only ARI: **0.05–0.11 (2026, five assets)**, **0.05–0.12 (2022 OOS, 3 assets)**; cross-resolution shape (5m–15m high, 15m→1h drop, 1h–1d low) preserved. The 15m→1h break is **not** a stress artefact. **Paper:** Suppl Table A.17 `tab:calm_day_subsample`, §3.4. | `outputs/calm_day_subsample_ari.csv` |
| `extended_var_uplift` | Referee R1 Q9: does the dissonance signal carry economic value? | Per (asset, regime) cell 99% empirical-VaR from 5m log-returns under the 1h and 1d classifiers; conservative-resolution rule posts max(\|VaR_1h\|, \|VaR_1d\|). | 2026 always-conservative uplift over 1d baseline: **4.2% (QQQ) – 21.0% (CL)**; 2022 OOS: **10.0%–12.7%**. Disagreement-day-only ratio gives per-asset risk-budget tightening factors (e.g. SPY 1.33× → tighten by 0.75). **Paper:** Suppl Table A.18 `tab:var_uplift`, §4. | `outputs/var_uplift_by_resolution.csv` |
| `extended_disagree_config` | Caption-level precision footnote for the VaR-uplift table. | Per (asset, episode), count disagreement days falling into each of the two configurations: 1h-crisis/1d-calm vs 1h-calm/1d-crisis. | Quantifies whether the "single-configuration concentration" framing is exact or approximate. **Paper:** Suppl Table `tab:var_uplift` notes. | `outputs/disagree_config_breakdown.csv` |
| `stress_vs_calm` | Referee R1 Q7 follow-up: paired test of stress-window vs calm-window ARI. | Consume bootstrap CSVs from both episodes; pool 7 asset×episode pairs; paired t / Wilcoxon / sign tests on the difference. | Mean diff +0.014 (σ 0.060); paired t **p = 0.54**; Wilcoxon p = 0.58; sign p = 1.00; 4/7 pairs stress > calm. **Cannot reject equality.** **Paper:** Suppl Table A.19 `tab:stress_vs_calm`. | `outputs/stress_vs_calm_test.csv`, `stress_vs_calm_test_summary.csv`, `stress_vs_calm_test.txt` |
| `cross_asset` | Cross-asset rolling-ARI resonance figure for the supplement. | Per-day mean off-diag ARI across the current 2026 panel; rolling 7-day trace. | Shows synchronised dissonance episodes across the 2026 panel. **Paper:** Supplementary figure. | `outputs/cross_asset_resonance.png` |

#### Tier 4: Dev helpers (not in reproduce chain)

| Command | Purpose | Outputs |
| --- | --- | --- |
| `summarize` | Console pivot of mean off-diag ARI by (asset, model, window) for quick eyeballing during development. | stdout only |

**Event window:** The default outbreak window is **~5–10 trading days** to avoid large 5m data volume. Default in `config.yaml` is ~2 weeks (e.g. 2026-01-06 to 2026-01-20); T0 can be fixed from volatility/news.

Override dates/symbols/port (with or without `--download`):

```bash
python run.py --download --start 2026-01-06 --end 2026-01-20 --symbols SPY USDJPY CL
python run.py --download --port 7497 --raw-dir data
```

**IB prerequisite:** Gateway (or TWS) running; API connected, Historical Data Farm ON. Default: `127.0.0.1:4002` (Gateway paper). TWS paper = `7497`.

Output: `raw_dir/<symbol>_5m.csv` with **Date** (America/New_York), **Open**, **High**, **Low**, **Close**, **Volume**.

## Config: `config.yaml`

- `raw_dir`, `outputs_dir`: same as Representation-Invariant.
- `assets`: list of symbols (SPY, USDJPY, CL).
- `episode`: explicit event/calm window set used by analysis (`2026_iran` or `2022_ukraine`).
- `ib`: `host`, `port`, `client_id`, `start_date`, `end_date`, `future_expiry_by_symbol` (e.g. CL: `"CONTFUT"` for IB continuous front-month futures).

## Notes

- **CL (WTI):** Recommended setting is `ib.future_expiry_by_symbol.CL: "CONTFUT"` so IB returns continuous front-month futures rather than one fixed far-month contract.
- **Timezone:** All bars are written in **America/New_York**.
- **Bar count per day:** SPY ≈78, CL ≈276, USDJPY 288; do not use a global fixed bar count when building rolling features.
- **Paper figure sync:** regenerate plots into `outputs/` and copy `outputs/SPY_timeline.png`, `outputs/QQQ_timeline.png`, `outputs/CL_timeline.png`, `outputs/USDJPY_timeline.png`, and `outputs/GLD_timeline.png` to the corresponding `paper/<asset>_timeline.png` paths before building the manuscript.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).

## Citation

If you use this code, please cite the accompanying paper and/or this repository. See [CITATION.cff](CITATION.cff).
