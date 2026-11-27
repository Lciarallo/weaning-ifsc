#!/usr/bin/env bash
# Regera os artefatos Python a partir das evidências nativas Netica verificadas.
set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

"$PYTHON" src/generate_artifacts.py
"$PYTHON" src/export_models.py
"$PYTHON" src/build_notebook.py
"$PYTHON" -m nbconvert --to html notebooks/Weaning_Bayes_Fuzzy.ipynb \
  --output Weaning_Bayes_Fuzzy.html --output-dir notebooks
"$PYTHON" src/build_report.py
"$PYTHON" scripts/preparar_entrega.py

echo
echo "Concluído. Relatório: report/Relatorio_Weaning_Bayes_Fuzzy.pdf"
echo "Notebook: notebooks/Weaning_Bayes_Fuzzy.ipynb"
