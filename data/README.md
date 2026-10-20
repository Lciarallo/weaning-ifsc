# Bases de dados

- `weaning_train.csv` / `.tsv`: 21 tentativas de treino (12 fracassos, 9 sucessos).
- `weaning_test.csv` / `.tsv`: 9 tentativas de teste (3 fracassos, 6 sucessos).
- `weaning_full.csv`: união das duas bases (30 casos), usada na avaliação fuzzy e nas tabelas consolidadas.

Colunas: `Base`, `IDnum`, `NIF`, `RR`, `VT`, `RSBI`, `Resultado`. Os arquivos TSV são os originais do material da disciplina; os CSV foram gerados a partir deles, com a coluna `Base`.

**Protocolo:** os 9 casos de teste não foram usados para definir limiares, estimar priors/CPTs, escolher funções de pertinência ou calibrar regras. Ver `README.md` e o relatório.

**Fonte:** ASHUTOSH, K. et al. *Critical Care Medicine*, v. 20, n. 9, p. 1295–1301, 1992.
