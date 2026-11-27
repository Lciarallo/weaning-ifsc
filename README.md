# Desmame ventilatório: Naive Bayes e Fuzzy Mamdani

Projeto acadêmico do curso de **Especialização em Computação Aplicada à Indústria** — disciplina **Sistemas Probabilísticos e Difusos** (IFSC — Prof. Cesar Alberto Penz), desenvolvido sobre a base de 30 tentativas de interrupção da ventilação mecânica descrita por Ashutosh et al. (*Critical Care Medicine*, 1992).

Autores: **Luiz Eduardo do Nascimento Ciarallo** e **Renato Deodoro**.

O projeto compara **duas redes Naive Bayes** (nó-alvo `Resultado`) — uma com discretização por conhecimento clínico e outra com dois intervalos produzidos automaticamente pelo **Netica Application 7.01** — e um **sistema fuzzy Mamdani** com quatro entradas, saída `Resultado` (`Fracasso`/`Sucesso`), **12 regras compactas** (cobrindo 100% das $3^4 = 81$ combinações) e centroide, executado e validado no **QtFuzzyLite 4.0** e no `fuzzylite 4.0`. A Rede 2 foi aprendida pelo **Netica C API 5.04** usando apenas os 21 casos de treino, e ambas as redes foram avaliadas diretamente no **Netica 7.01** via **`Cases > Process Cases`** com `CTRL_Respira.txt`.

## Entregáveis principais

| Arquivo | Conteúdo |
|---|---|
| [`report/Relatorio_Weaning_Bayes_Fuzzy.pdf`](report/Relatorio_Weaning_Bayes_Fuzzy.pdf) | Relatório técnico com autoria da dupla, resultados e capturas reais do Netica 7.01 e do QtFuzzyLite 4.0 |
| [`notebooks/Weaning_Bayes_Fuzzy.ipynb`](notebooks/Weaning_Bayes_Fuzzy.ipynb) | Notebook executado de ponta a ponta, com tabelas, gráficos e capturas dos programas gráficos |
| [`notebooks/Weaning_Bayes_Fuzzy.html`](notebooks/Weaning_Bayes_Fuzzy.html) | Versão HTML do notebook, para leitura sem Jupyter |
| [`netica/Weaning_Rede1_Clinica.dne`](netica/Weaning_Rede1_Clinica.dne) | Rede 1 (discretização clínica, nó-alvo `Resultado`, níveis contínuos compatíveis com `CTRL_Respira.txt`) |
| [`netica/Weaning_Rede2_Netica.dne`](netica/Weaning_Rede2_Netica.dne) | Rede 2 (discretização automática no Netica 7.01, nó-alvo `Resultado`) |
| [`fuzzy/weaning_mamdani.fis`](fuzzy/weaning_mamdani.fis) | FIS para MATLAB/Octave (12 regras compactas) |
| [`fuzzy/weaning_mamdani.fcl`](fuzzy/weaning_mamdani.fcl) | FIS para `fuzzylite 4.0` (IEC 61131-7) |
| [`fuzzy/weaning_mamdani.fll`](fuzzy/weaning_mamdani.fll) | FIS nativo do **QtFuzzyLite 4.0** / `fuzzylite 4.0` |
| [`src/weaning_lib.py`](src/weaning_lib.py) | Implementação comentada de dados, conferência dos arquivos da disciplina, Naive Bayes, Mamdani, métricas e validações |
| [`tables/`](tables/) | Tabelas em CSV e Markdown, incluindo validação Netica × Python e acurácia balanceada |
| [`figures/`](figures/) | 20 figuras Python em alta resolução (incl. `fig05b_probabilidades_teste.png`) |
| [`netica/evidencias/`](netica/evidencias/) | Rede discretizada, capturas reais do Netica 7.01 (Rede 1, Rede 2, CPT, TE3), saídas do `Process Cases` com `CTRL_Respira.txt`, probabilidades da C API, log e hashes |
| [`fuzzy/evidencias/`](fuzzy/evidencias/) | Capturas reais do **QtFuzzyLite 4.0** (`Setup` com 12 regras, 4 mapas `Surface 2D` e simulações `Control` de TE5, TR10 e TE3) |

## Arquivos recebidos

Os 9 arquivos originais da disciplina foram preservados sem alteração em [`references/originais/`](references/originais/). O manifesto com SHA-256 e tamanho de cada um está em [`references/MANIFEST_ORIGINAIS.csv`](references/MANIFEST_ORIGINAIS.csv), e a explicação em [`references/README_ORIGINAIS.md`](references/README_ORIGINAIS.md).

## Auditoria de conformidade

O arquivo [`CONFORMIDADE.md`](CONFORMIDADE.md) confronta cada item do enunciado com as evidências. Todos os requisitos técnicos estão implementados e verificados, incluindo a execução nativa nos programas gráficos (**Netica Application 7.01** e **QtFuzzyLite 4.0**). O pacote `Entrega_Weaning_IFSC_Luiz_Ciarallo.zip` é gerado para encaminhamento; o envio por e-mail permanece pendente.

## Resultados principais

Base completa (30 casos; classe positiva = Sucesso):

| Modelo | Acurácia | Acurácia bal. | Sensibilidade | Especificidade | Brier |
|---|---:|---:|---:|---:|---:|
| Naive Bayes 1 — clínica | **90,0%** (27/30) | **90,0%** | **86,7%** | **93,3%** | **0,078** |
| Naive Bayes 2 — Netica | 83,3% (25/30) | 83,3% | **86,7%** | 80,0% | 0,137 |
| Fuzzy Mamdani — 12 regras (limiar 60) | 86,7% (26/30) | 86,7% | 80,0% | **93,3%** | — |

No teste (9 casos): Rede 1 = **9/9** (100%); Rede 2 = **7/9** (77,8%, acurácia bal. 66,7%); Fuzzy = **9/9** (100%). O teste é pequeno: o intervalo de confiança de 95% para 9/9 vai de 70,1% a 100%.

## Como reproduzir

Requisitos: Python 3.11+ (ou ambiente virtual `.venv` com Python 3.12) e as bibliotecas em [`requirements.txt`](requirements.txt). Para gerar novamente tabelas, figuras, modelos exportados, notebook, PDF e pacote de entrega:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run_all.sh
```

Esse comando usa automaticamente `.venv/bin/python` (se existir), verifica os hashes das evidências nativas e dos arquivos originais da disciplina, regenera os artefatos Python e prepara o ZIP. Para repetir as execuções nos programas gráficos, consulte [`netica/README_NETICA.md`](netica/README_NETICA.md) e [`fuzzy/README_FUZZY.md`](fuzzy/README_FUZZY.md).

Passo a passo manual:

```bash
.venv/bin/python src/generate_artifacts.py   # tabelas, 20 figuras, resultados.json
.venv/bin/python src/export_models.py        # .dne, .fis, .fcl, .fll e casos Netica
.venv/bin/python src/build_notebook.py       # constrói e executa o notebook
.venv/bin/python -m nbconvert --to html notebooks/Weaning_Bayes_Fuzzy.ipynb \
  --output Weaning_Bayes_Fuzzy.html --output-dir notebooks
.venv/bin/python src/build_report.py         # gera HTML e PDF
.venv/bin/python scripts/preparar_entrega.py # confere integridade e gera o ZIP
```

O PDF usa Chromium headless (ou o Chromium do Playwright; `CHROME_BIN` permite indicar outro executável), `pypdf` e `reportlab`.

## Estrutura

```
weaning-ifsc/
├── data/                 # bases de treino, teste e completa
├── notebooks/            # notebook executado (.ipynb) e HTML
├── src/                  # biblioteca, figuras, exportação, geração do relatório
├── netica/               # redes .dne, casos, evidências GUI/API e instruções
├── fuzzy/                # FIS .fis/.fcl/.fll, evidências QtFuzzyLite 4.0 e instruções
├── figures/              # 20 figuras PNG
├── tables/               # tabelas CSV/Markdown
├── report/               # PDF, HTML, manifesto e resultados.json
└── references/           # materiais de apoio, originais da disciplina e base de conhecimento
```

## Notas metodológicas

- **Antileakage:** os 9 casos de teste não participaram da escolha de limiares, da estimação de priors/CPTs, da definição das funções de pertinência nem da elaboração das regras.
- **Rede 2 (Netica 7.01):** execução real de `Modify > Auto-Discretize Nodes`, dois estados e 20% de arredondamento, com apenas as quatro entradas dos 21 casos de treino. O Python lê os cortes do `.dne` salvo pelo Netica. Ambas as redes foram executadas no Netica 7.01 via `Cases > Process Cases` com `CTRL_Respira.txt`.
- **Redundância:** RSBI = RR/VT(L). A inclusão simultânea de RR, VT e RSBI viola a independência condicional do Naive Bayes; o relatório explicita o problema e quantifica seu efeito por validação leave-one-out.
- **Fuzzy (QtFuzzyLite 4.0):** 12 regras compactas com saída binária linguística `Resultado` (`Fracasso` `[0,0,35,65]` e `Sucesso` `[35,65,100,100]`), cobrindo 100% do domínio; limiar conservador 60/100 (faixa 40–60 como zona de cautela).
- **Validação cruzada:** Netica C API 5.04 × Python (60/60 classes, erro ≤ 9,87 × 10⁻⁸); Netica 7.01 GUI `Process Cases` × Python (60/60 classes, erro ≤ 5,5 × 10⁻⁷); `CategoricalNB` × Python; `scikit-fuzzy` e `fuzzylite 4.0` × Python.

## Limitações e uso

A base é pequena, histórica e uniestruturada (uma medida por paciente), sem gasometria, sedação, força de tosse ou comorbidades. O caso TR1 — todos os indicadores favoráveis, mas fracasso real — mostra que as quatro variáveis não capturam todo o risco. **Este material é acadêmico e não constitui recomendação clínica individual.** Qualquer uso em pacientes exige validação prospectiva, aprovação ética e governança clínica.

## Fonte dos dados

ASHUTOSH, K. et al. Prediction criteria for successful weaning from respiratory support: statistical and connectionist analyses. *Critical Care Medicine*, v. 20, n. 9, p. 1295–1301, 1992. DOI: 10.1097/00003246-199209000-00017.

Materiais de apoio do professor Cesar Alberto Penz (IFSC), incluindo a “Base de Conhecimento Weaning”, os arquivos `Respira_DB_*.txt` / `CTRL_Respira.txt` e os vídeos indicados no enunciado.
