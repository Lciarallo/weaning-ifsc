#!/usr/bin/env bash
# Regera os artefatos Python a partir das evidências nativas Netica verificadas.
set -euo pipefail
cd "$(dirname "$0")"

python3 src/generate_artifacts.py
python3 src/export_models.py
python3 src/build_notebook.py
python3 -m nbconvert --to html notebooks/Weaning_Bayes_Fuzzy.ipynb \
  --output Weaning_Bayes_Fuzzy.html --output-dir notebooks
python3 src/build_report.py
python3 scripts/preparar_entrega.py

echo
echo "Concluído. Relatório: report/Relatorio_Weaning_Bayes_Fuzzy.pdf"
echo "Notebook: notebooks/Weaning_Bayes_Fuzzy.ipynb"
