#!/usr/bin/env bash
# Reproduce every paper Table / Figure from the raw 5m CSVs in data/ and data_2022/.
# See README.md "Table / Figure -> Command -> Output" mapping for what each step generates.
#
# Prerequisite: 2026 panel data/{SPY,QQQ,USDJPY,CL,GLD}_5m.csv and
# 2022 panel data_2022/{SPY,USDJPY,GLD}_5m.csv (run
# `python run.py pipeline --download` first if data/ is empty; asset list
# is authoritative in config.yaml / config_2022.yaml).
set -euo pipefail
cd "$(dirname "$0")"

python run.py pipeline                                                        # 2026 main pipeline: Tables 1-2, Fig 1, A.1-A.13 (excl. A.6/A.7/A.11/A.12 sweeps -> covered by `extended`)
python run.py pipeline --episode 2022_ukraine --raw-dir data_2022 --outputs-dir outputs_2022   # 2022 OOS replica: Table A.14
python run.py extended                    # extended sweep: A.6/A.7/A.11 (K x window), A.12 (GLD block-sweep), A.15-A.18, hypothesis tests, simulation/GARCH-MS/K=3 baselines, disagree-day config, EM-restart placebo, per-asset full ML baseline
python run.py stress_vs_calm              # Table A.19 (stress vs calm paired test)
python run.py cross_asset                 # Supplementary figure: cross_asset_resonance.png

# Notebooks under notebooks/ are hand-maintained (each is substantively
# different); re-execute them in place so embedded Table/Figure cells render
# from the freshly populated outputs/.
if command -v jupyter >/dev/null 2>&1; then
    jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
else
    echo "jupyter not found on PATH -- skipping notebook execution."
fi
