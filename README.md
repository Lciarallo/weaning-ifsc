# Desmame ventilatório: Naive Bayes (Netica) e Fuzzy Mamdani (QtFuzzyLite)

Trabalho da disciplina Sistemas Probabilísticos e Difusos, Especialização em Computação Científica para a Indústria (IFSC, Prof. Cesar Alberto Penz).

Autores: Luiz Eduardo do Nascimento Ciarallo e Renato Teodoro.

**Relatório:** [PDF](report/Relatorio_Weaning_Bayes_Fuzzy.pdf) · [HTML](report/Relatorio_Weaning_Bayes_Fuzzy.html)

## Resultados (30 casos; classe positiva = Sucesso)

| Modelo | Acurácia | Sensibilidade | Especificidade | Teste |
|---|---:|---:|---:|---:|
| Naive Bayes 1 (cortes clínicos) | 90,0% | 86,7% | 93,3% | 9/9 |
| Naive Bayes 2 (discretização do Netica) | 83,3% | 86,7% | 80,0% | 7/9 |
| Fuzzy Mamdani, 12 regras (limiar 60) | 86,7% | 80,0% | 93,3% | 9/9 |

## Estrutura

- `netica/`: redes `.dne`, arquivos de casos, capturas e saídas do Process Cases.
- `fuzzy/`: modelos `.fll`, `.fcl`, `.fis` e capturas do QtFuzzyLite 4.0.
- `data/`: bases de treino, teste e completa.
- `references/`: enunciado, base de conhecimento e arquivos originais da disciplina.
- `report/`: relatório.

O Netica não é distribuído neste repositório; o download oficial está em https://www.norsys.com/download.html.

Material acadêmico; não constitui recomendação clínica.

## Fonte dos dados

ASHUTOSH, K. et al. Prediction criteria for successful weaning from respiratory support: statistical and connectionist analyses. *Critical Care Medicine*, v. 20, n. 9, p. 1295-1301, 1992.
