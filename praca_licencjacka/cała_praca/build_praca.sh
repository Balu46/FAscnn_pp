#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="${SCRIPT_DIR}/../../.venv/bin/python"
AUTO_RESULTS_DIR="${SCRIPT_DIR}/../wyniki_auto"

"${VENV_PY}" "${SCRIPT_DIR}/../../visualisation/comparison.py" \
  --root "${SCRIPT_DIR}/../../results" \
  --pattern metrics.txt \
  --outdir "${SCRIPT_DIR}/../../results/plots" || true

mkdir -p "${AUTO_RESULTS_DIR}"
cp "${SCRIPT_DIR}/../../results/plots/metrics_summary.csv" "${AUTO_RESULTS_DIR}/metrics_summary.csv" || true

"${VENV_PY}" "${SCRIPT_DIR}/../scripts/generate_results_text.py" || true
# "${VENV_PY}" "${SCRIPT_DIR}/../scripts/generate_tensorboard_assets.py" || true
"${VENV_PY}" "${SCRIPT_DIR}/../scripts/sync_results_figures.py" || true

cd "${SCRIPT_DIR}"
pdflatex -interaction=nonstopmode praca_licencjacka.tex
biber praca_licencjacka
pdflatex -interaction=nonstopmode praca_licencjacka.tex
pdflatex -interaction=nonstopmode praca_licencjacka.tex
