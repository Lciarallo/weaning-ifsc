# Desmame ventilatório: Naive Bayes e Fuzzy Mamdani

Projeto acadêmico da disciplina **Sistemas Probabilísticos e Difusos** (IFSC — Prof. Cesar Alberto Penz), desenvolvido sobre a base de 30 tentativas de interrupção da ventilação mecânica descrita por Ashutosh et al. (*Critical Care Medicine*, 1992).

O projeto compara **duas redes Naive Bayes** — uma com discretização por conhecimento clínico e outra com dois intervalos automáticos no estilo Netica — e um **sistema de inferência fuzzy Mamdani** com quatro entradas, 81 regras e defuzzificação por centroide.

## Entregáveis principais

| Arquivo | Conteúdo |
|---|---|
| [`report/Relatorio_Weaning_Bayes_Fuzzy.pdf`](report/Relatorio_Weaning_Bayes_Fuzzy.pdf) | Relatório técnico-científico completo, 34 páginas, pronto para envio |
| [`notebooks/Weaning_Bayes_Fuzzy.ipynb`](notebooks/Weaning_Bayes_Fuzzy.ipynb) | Notebook executado de ponta a ponta, com tabelas e gráficos |
| [`notebooks/Weaning_Bayes_Fuzzy.html`](notebooks/Weaning_Bayes_Fuzzy.html) | Versão HTML do notebook, para leitura sem Jupyter |
| [`netica/Weaning_Rede1_Clinica.dne`](netica/Weaning_Rede1_Clinica.dne) | Rede 1 exportada para o Netica |
| [`netica/Weaning_Rede2_Netica.dne`](netica/Weaning_Rede2_Netica.dne) | Rede 2 exportada para o Netica |
| [`fuzzy/weaning_mamdani.fis`](fuzzy/weaning_mamdani.fis) | FIS para MATLAB/Octave |
| [`fuzzy/weaning_mamdani.fcl`](fuzzy/weaning_mamdani.fcl) | FIS para fuzzylite 4.0 |
| [`src/weaning_lib.py`](src/weaning_lib.py) | Implementação comentada de dados, Naive Bayes, Mamdani, métricas e validações |
| [`tables/`](tables/) | 21 tabelas em CSV e Markdown |
| [`figures/`](figures/) | 19 figuras em alta resolução |

## Arquivos recebidos

Os 9 arquivos enviados pelo usuário foram preservados sem alteração em [`references/originais/`](references/originais/). O manifesto com SHA-256 e tamanho de cada um está em [`references/MANIFEST_ORIGINAIS.csv`](references/MANIFEST_ORIGINAIS.csv), e a explicação em [`references/README_ORIGINAIS.md`](references/README_ORIGINAIS.md).

## Auditoria de conformidade

O arquivo [`CONFORMIDADE.md`](CONFORMIDADE.md) traz a auditoria ponto a ponto do enunciado, com status e ressalvas. Resumo: todos os itens têm entrega; as duas ressalvas são a execução nativa do Netica (software proprietário) e a citação individual dos vídeos.

## Resultados principais

Base completa (30 casos; classe positiva = Sucesso):

| Modelo | Acurácia | Sensibilidade | Especificidade | Brier |
|---|---:|---:|---:|---:|
| Naive Bayes 1 — clínica | **90,0%** (27/30) | **86,7%** | **93,3%** | **0,078** |
| Naive Bayes 2 — Netica | 83,3% (25/30) | **86,7%** | 80,0% | 0,137 |
| Fuzzy Mamdani — limiar 60 | 86,7% (26/30) | 80,0% | **93,3%** | — |

No teste (9 casos): Rede 1 = 9/9; Rede 2 = 7/9; Fuzzy = 9/9. O teste é muito pequeno: o intervalo de confiança de 95% para 9/9 vai de 70,1% a 100%.

## Como reproduzir

Requisitos: Python 3.11+ e as bibliotecas em [`requirements.txt`](requirements.txt). Para gerar novamente tabelas, figuras, redes exportadas, notebook e PDF:

```bash
python3 -m pip install -r requirements.txt
./run_all.sh
```

Ou, passo a passo:

```bash
python3 src/generate_artifacts.py   # tabelas, figuras, resultados.json
python3 src/export_models.py        # .dne, .fis, .fcl e casos Netica
python3 src/build_notebook.py       # constrói e executa o notebook
jupyter nbconvert --to html notebooks/Weaning_Bayes_Fuzzy.ipynb \
  --output Weaning_Bayes_Fuzzy.html --output-dir notebooks
python3 src/build_report.py         # gera HTML e PDF
```

O PDF é impresso pelo Chromium headless e recebe cabeçalho/rodapé com numeração via `pypdf` + `reportlab`.

## Estrutura

```
weaning-ifsc/
├── data/                 # bases de treino, teste e completa
├── notebooks/            # notebook executado (.ipynb) e HTML
├── src/                  # biblioteca, figuras, exportação, geração do relatório
├── netica/               # redes .dne, casos discretizados e instruções
├── fuzzy/                # FIS .fis/.fcl e instruções
├── figures/              # 19 figuras PNG
├── tables/               # tabelas CSV/Markdown
├── report/               # PDF, HTML e resultados.json
└── references/           # materiais de apoio e base de conhecimento
```

## Notas metodológicas

- **Antileakage:** os 9 casos de teste não participaram da escolha de limiares, da estimação de priors/CPTs, da definição das funções de pertinência nem da calibração das regras.
- **Rede 2 (Netica):** o Netica é proprietário e não estava disponível no ambiente. Os cortes automáticos da execução de referência foram reproduzidos em Python, auditados apenas no treino (partições 10/11 ou 11/10) e exportados em `.dne`. A captura de tela do Netica (P(Fracasso) = 56,5%, P(Sucesso) = 43,5%) é compatível com a suavização de Laplace α = 1 usada no projeto.
- **Redundância:** RSBI = RR/VT(L). A inclusão simultânea de RR, VT e RSBI viola a independência condicional do Naive Bayes; o relatório explicita o problema e quantifica seu efeito por validação leave-one-out.
- **Fuzzy:** o limiar de 60/100 é uma política conservadora de decisão, não um parâmetro ajustado ao teste. A faixa 40–60 é uma zona de cautela.
- **Validação:** Naive Bayes manual × `sklearn.CategoricalNB` com as mesmas priors = erro máximo da ordem de 10⁻¹⁶; Mamdani manual × `scikit-fuzzy` < 10⁻⁵ no centroide.

## Limitações e uso

A base é pequena, histórica e uniestruturada (uma medida por paciente), sem gasometria, sedação, força de tosse ou comorbidades. O caso TR1 — todos os indicadores favoráveis, mas fracasso real — mostra que as quatro variáveis não capturam todo o risco. **Este material é acadêmico e não constitui recomendação clínica individual.** Qualquer uso em pacientes exige validação prospectiva, aprovação ética e governança clínica.

## Fonte dos dados

ASHUTOSH, K. et al. Prediction criteria for successful weaning from respiratory support: statistical and connectionist analyses. *Critical Care Medicine*, v. 20, n. 9, p. 1295–1301, 1992. DOI: 10.1097/00003246-199209000-00017.

Materiais de apoio do professor Cesar Alberto Penz (IFSC), incluindo a “Base de Conhecimento Weaning” e os vídeos indicados no enunciado.
