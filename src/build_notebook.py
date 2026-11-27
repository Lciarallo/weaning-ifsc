# -*- coding: utf-8 -*-
"""Constrói e executa o notebook didático do projeto."""
from __future__ import annotations

from pathlib import Path
import sys

import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


cells = []

cells.append(md(r"""
# Apoio à decisão na interrupção da ventilação mecânica
### Duas redes Naive Bayes e um sistema de inferência fuzzy Mamdani

**Curso:** Especialização em Computação Aplicada à Indústria — IFSC  
**Disciplina:** Sistemas Probabilísticos e Difusos  
**Autores:** Luiz Eduardo do Nascimento Ciarallo · Renato Deodoro  
**Professor:** Cesar Alberto Penz  
**Base de dados:** Ashutosh *et al.*, *Critical Care Medicine*, 1992 (21 casos de treino + 9 de teste)  
**Data:** novembro de 2026

---

Este notebook reproduz integralmente o projeto: verificação dos arquivos originais da disciplina, análise exploratória, discretização clínica e automática, redes Naive Bayes (nó-alvo `Resultado`), sistema Mamdani com **12 regras compactas** que cobrem integralmente o espaço de entrada ($3^4 = 81$ combinações), avaliação nos 30 casos, os três estudos de caso exigidos e as evidências gráficas reais obtidas no **Netica Application 7.01** e no **QtFuzzyLite 4.0**. Todo o código de apoio está em `src/weaning_lib.py`; as tabelas e figuras são gravadas em `tables/` e `figures/`.

> **Convenção:** a classe positiva é **Sucesso** (o paciente pode ser liberado). Assim, a sensibilidade mede a capacidade de reconhecer quem realmente pode sair do ventilador; a especificidade mede a capacidade de reter quem fracassaria. Falso positivo = liberação prematura; falso negativo = atraso na liberação.
"""))


cells.append(md(r"""
## Como este notebook está organizado

O notebook segue exatamente os itens do trabalho e exibe **as 20 figuras Python e as capturas reais dos programas gráficos (Netica 7.01 e QtFuzzyLite 4.0)**. Cada seção indica o passo correspondente:

| Seção | Passo do trabalho | O que será mostrado |
|---|---|---|
| 1 | Descrição do processo e base | Verificação dos arquivos da disciplina, análise exploratória, correlações e protocolo antileakage |
| 2 | Item 2(a–c) — Naive Bayes | Topologia (`Resultado`), as duas discretizações, CPTs, tabelas caso a caso, cálculo posterior passo a passo, matrizes de confusão, métricas (incluindo acurácia balanceada), probabilidades no teste, limiares, divergências e execução no **Netica 7.01** (`Process Cases` com `CTRL_Respira.txt`) |
| 3 | Item 3(a) — Fuzzy | Operadores Mamdani, universos, funções de pertinência, **12 regras compactas**, completude, avaliação nos 30 casos, comportamento de platô do centroide Mamdani, superfícies e capturas reais no **QtFuzzyLite 4.0** |
| 4 | Item 3(b) — Exemplos | TE5 (sucesso), TR10 (fracasso) e TE3 (crítico/duvidoso): fuzzificação, regras ativadas, agregação, centroide e simulações no **QtFuzzyLite 4.0** |
| 5 | Item 4 — Fechamento | Comparação final, conclusão e limitações |
| 6 | Reprodutibilidade | Validações cruzadas (Python, `sklearn`, `scikit-fuzzy`, Netica C API 5.04, Netica 7.01 GUI e `fuzzylite 4.0`), arquivos e checklist de conformidade |

**Mapa das figuras:** 1 distribuições; 2 topologia; 3 matrizes NB; 4 métricas NB; 5 divergência entre redes; 5b probabilidades de sucesso no teste; 6 cortes de discretização; 7 funções de pertinência; 8 regras; 9 superfície NIF×RR; 10 superfície VT×RSBI; 11 saída fuzzy nos 30 casos; 12 matrizes fuzzy; 13–18 casos TE5/TR10/TE3 (entrada e saída); 19 comparação final.

> **Execução nativa comprovada nos programas gráficos:**
> - **Netica Application 7.01:** discretização automática da Rede 2 (`Modify > Auto-Discretize Nodes`, 2 estados, 20% de arredondamento, 21 casos de treino), compilação das duas redes com nó-alvo `Resultado`, inferência interativa (TE3) e execução em lote (`Cases > Process Cases` com `CTRL_Respira.txt`), além da validação pela **Netica C API 5.04** (`netica/evidencias/`).
> - **QtFuzzyLite 4.0 / `fuzzylite 4.0`:** carregamento de `fuzzy/weaning_mamdani.fll` (12 regras válidas), geração dos 4 mapas `Surface 2D` e simulação interativa na aba `Control` para TE5, TR10 e TE3 (`fuzzy/evidencias/`).
"""))

cells.append(md(r"""
## 0. Preparação do ambiente

O notebook pressupõe Python 3.11+ com `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-fuzzy`, `scikit-learn` e `statsmodels`. O caminho do projeto é detectado automaticamente a partir do diretório do notebook.
"""))

cells.append(code(r"""
from pathlib import Path
import sys, json, itertools, warnings, math

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image

import weaning_lib as w
import plotting

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")

FIG = ROOT / "figures"
TAB = ROOT / "tables"
print("Projeto:", ROOT)
print("Python:", sys.version.split()[0])
print("pandas:", pd.__version__)
"""))

cells.append(md(r"""
## 1. Dados e análise exploratória

A base contém 21 tentativas de interrupção para treinamento (9 sucessos e 12 fracassos) e 9 para teste (3 fracassos e 6 sucessos). As variáveis são:

- **NIF/PImax** — pico negativo da pressão inspiratória (cm.H₂O); valores mais negativos indicam maior força.
- **RR** — frequência respiratória (resp/min).
- **VT** — volume tidal (mL).
- **RSBI** — *Rapid Shallow Breathing Index* = RR / VT(L) (resp/min/L).

O RSBI é uma variável derivada: `RSBI = RR / (VT/1000)`. Essa dependência será discutida na seção de Naive Bayes.
"""))

cells.append(code(r"""
train, test, full = w.load_data()
chk = w.verify_discipline_files()
print(f"Treino: {train.shape[0]} casos | Teste: {test.shape[0]} casos | Completa: {full.shape[0]} casos")
print("Conferência linha a linha com os arquivos originais da disciplina (Respira_DB_*):", chk)
display(train["Resultado"].value_counts().rename("Treino").to_frame().T)
display(test["Resultado"].value_counts().rename("Teste").to_frame().T)
display(full[["ID","NIF","RR","VT","RSBI","Resultado"]].head(10))
"""))

cells.append(code(r"""
desc = w.descriptive_table(full)
display(desc[["Variável","Desfecho","n","Média ± DP","Mínimo","Máximo"]])
"""))

cells.append(code(r"""
display(Markdown("**Figura 1 — distribuição das variáveis por desfecho.**"))
display(Image(filename=str(FIG / "fig01_distribuicoes.png"), width=1050))
"""))

cells.append(md(r"""
**Leitura clínica.** Há separação parcial entre os grupos: pacientes com sucesso tendem a apresentar NIF mais negativo, menor RR e maior VT. Há, porém, sobreposição importante — em especial VT, RSBI e RR —, o que justifica modelos que representem incerteza e não apenas cortes rígidos. O caso TR1 é um exemplo de sobreposição extrema: todos os indicadores parecem favoráveis, mas o desfecho real foi fracasso.
"""))

cells.append(code(r"""
corr = full[["NIF","RR","VT","RSBI"]].corr(method="spearman").round(2)
display(corr)

calc = full["RR"] / (full["VT"] / 1000)
print("Máxima diferença absoluta entre RSBI informado e RR/(VT/1000):", round(float((calc - full["RSBI"]).abs().max()), 3))
"""))

cells.append(md(r"""
A correlação de Spearman é muito alta entre RSBI e RR (0,91) e entre RSBI e VT (−0,94), como esperado pela definição. Isso antecipa o problema de dupla contagem no Naive Bayes: RR, VT e RSBI não são evidências condicionalmente independentes.

### 1.1 Protocolo antileakage

1. Os 9 casos de teste **não** participam da escolha de limiares, da estimação de CPTs, da escolha de funções de pertinência ou da calibração de regras.
2. A Rede 2 usa somente os 21 casos de treino para reproduzir os cortes automáticos.
3. As funções de pertinência fuzzy vêm da literatura clínica, não de ajuste aos dados.
4. O limiar operacional conservador do fuzzy (60) é uma decisão de política clínica, não um parâmetro ajustado ao teste.
5. A validação *leave-one-out* é feita exclusivamente no treino.
"""))

cells.append(md(r"""
## 2. Redes Naive Bayes

### 2.1 Topologia e regra de decisão

A rede tem um nó pai `Resultado` (estados `Fracasso` e `Sucesso`, idêntico aos arquivos `Respira_DB_*.txt` e `CTRL_Respira.txt` da disciplina) e quatro filhos: `NIF`, `RR`, `VT` e `RSBI`. A hipótese de independência condicional permite escrever

$$
P(C \mid x_1,\ldots,x_4) \propto P(C)\prod_{j=1}^{4}P(x_j \mid C),
$$

e a decisão é a classe de maior probabilidade posterior. As probabilidades são estimadas por máxima verossimilhança com **suavização de Laplace** ($\alpha=1$):

$$
P(x_j=s \mid C=c)=\frac{N_{cjs}+\alpha}{N_c+\alpha K_j}.
$$

A suavização evita probabilidades nulas em estados não observados no treino — situação provável com apenas 21 casos.
"""))

cells.append(code(r"""
display(Markdown("**Figura 2 — topologia das duas redes e suas grades de discretização.**"))
display(Image(filename=str(FIG / "fig02_topologia_nb.png"), width=1100))
"""))

cells.append(md(r"""
### 2.2 Nuance obrigatória: RR, VT e RSBI simultaneamente

O RSBI é uma função determinística de RR e VT. Incluir os três nós viola a hipótese de independência condicional: a mesma informação sobre o padrão respiratório é contada duas ou três vezes. Em termos práticos, o Naive Bayes pode produzir posteriores excessivamente extremos (perto de 0% ou 100%) e parecer mais confiável do que realmente é.

**Decisão adotada:** manter as quatro variáveis nas duas redes principais, para responder exatamente ao escopo do trabalho e permitir comparação direta entre as discretizações. Como análise de robustez, a validação *leave-one-out* no treino compara o conjunto completo com subconjuntos (Seção 2.10). A redundância é, portanto, explicitada e medida — não ignorada.
"""))

cells.append(md(r"""
### 2.3 Rede 1 — discretização por conhecimento clínico

Os cortes foram definidos a partir da base de conhecimento da disciplina, do III Consenso Brasileiro de Ventilação Mecânica, de Yang & Tobin (1991), Silva *et al.* (2015), Hess & Kacmarek e do material da Dräger. A tabela completa está em `tables/02_discretizacao_clinica.csv`.
"""))

cells.append(code(r"""
clinical = pd.read_csv(TAB / "02_discretizacao_clinica.csv")
with pd.option_context("display.max_colwidth", None):
    display(clinical)
"""))

cells.append(md(r"""
**Justificativa dos cortes.** O limiar de NIF em −30 cm.H₂O é clássico para força muscular; a faixa −30 a −20 é explicitamente uma zona de incerteza, e valores acima de −20 (menos negativos) associam-se a falha. Para RR, 30 resp/min é o limite superior da faixa de sucesso em várias referências, enquanto 20 delimita a faixa de repouso. Para VT, 350 mL aproxima 5 mL/kg em um adulto de 70 kg e 315 mL é referência de sucesso. Para RSBI, 105 é o corte clássico; 150 delimita a faixa de alto risco. As transições entre estados não são “naturais”: são escolhas clínicas explícitas, e por isso a Rede 2 será comparada.
"""))

cells.append(md(r"""
### 2.4 Rede 2 — discretização automática executada pelo Netica

A Rede 2 foi discretizada pelo comando **Modify > Auto-Discretize Nodes** do Netica Application 7.01, com **dois intervalos por variável e 20% de arredondamento**, usando somente `netica/treino_discretizacao.cas` (21 casos, quatro entradas, sem desfecho). A semântica é `[inferior, superior)`: o limite inferior pertence ao intervalo, o superior não (exceto a última faixa). Os níveis foram lidos do arquivo salvo pelo próprio Netica:

- NIF: `[−60, −25)` e `[−25, −10]`
- RR: `[13, 30)` e `[30, 46]`
- VT: `[120, 350)` e `[350, 950]`
- RSBI: `[10, 90)` e `[90, 270]`

O corte de RSBI 90 foi uma saída do Netica, cuja opção de arredondamento foi fixada em 20% antes da avaliação. A mediana do treino (86) aparece apenas como estatística descritiva. O Python lê os níveis da rede nativa, sem conter cortes alternativos. Os hashes e a comparação das entradas com o treino são verificados antes de gerar os artefatos.
"""))

cells.append(code(r"""
audit = pd.read_csv(TAB / "04_auditoria_netica.csv")
display(audit)
compare = pd.read_csv(TAB / "03_comparacao_limiares.csv")
with pd.option_context("display.max_colwidth", None):
    display(compare)
display(Markdown("**Execução real no Netica Application 7.01 — Rede 1 (discretização clínica, compilada com nó-alvo `Resultado`).**"))
display(Image(filename=str(ROOT / "netica/evidencias/05_netica_rede1_clinica.png"), width=860))
display(Markdown("**Execução real no Netica Application 7.01 — Rede 2 (discretização automática, compilada com `Automatic Updating` ativo).**"))
display(Image(filename=str(ROOT / "netica/evidencias/07_netica_rede2_automatica.png"), width=860))
"""))

cells.append(md(r"""
**Figura 6 — onde os cortes caem na distribuição.** A linha tracejada representa os cortes clínicos da Rede 1; a linha cheia representa o corte automático da Rede 2. Os pontos pretos são os 21 casos de treino e os triângulos cinza são os 9 casos de teste. Note que o corte de NIF em −25 cai no meio da faixa clínica de incerteza (−30 a −20), o corte de RR coincide com 30, o de VT coincide com 350 e o de RSBI (90) fica abaixo do clássico 105.
"""))

cells.append(code(r"""
display(Markdown("**Figura 6 — cortes de discretização: conhecimento clínico versus Netica.**"))
display(Image(filename=str(FIG / "fig06_limiares.png"), width=1050))
"""))

cells.append(md(r"""
A execução nativa obtém `Resultado` com 56,5% de fracasso e 43,5% de sucesso, exatamente como na captura original `Process Cases.jpg` do professor. Para aprender com Laplace $\alpha=1$, o Netica inicia cada linha com probabilidades uniformes e experiência igual ao número de estados, acrescentando as contagens dos 21 casos pelo método `ReviseCPTsByCaseFile_bn`. Ambas as redes também foram processadas diretamente na interface gráfica do **Netica 7.01** via **`Cases > Process Cases`** com o arquivo de controle da disciplina (`CTRL_Respira.txt`), gerando `netica/evidencias/process_cases_rede{1,2}_{treino,teste}.txt` com concordância integral frente ao Python (erro máximo ≤ 5,5 × 10⁻⁷).
"""))

cells.append(md(r"""
### 2.5 Probabilidades condicionais aprendidas

Abaixo, as priors e as tabelas de probabilidade condicional (CPTs) completas das quatro variáveis nas duas redes, estimadas apenas no treino com Laplace α = 1. Os mesmos valores estão em `tables/08_nb_cpt.csv`.
"""))

cells.append(code(r"""
models = {
    "Rede 1 (clínica)": w.fit_naive_bayes(train, "clinical"),
    "Rede 2 (Netica)": w.fit_naive_bayes(train, "netica"),
}
prior = pd.DataFrame([{"Rede": k, "P(Fracasso)": v.priors["Fracasso"], "P(Sucesso)": v.priors["Sucesso"]} for k, v in models.items()])
display(prior)

cpt = pd.read_csv(TAB / "08_nb_cpt.csv")
for var in w.VARIABLES:
    piv = cpt[cpt["Variável"] == var].pivot_table(index=["Rede","Estado"], columns="Classe", values="P(estado | classe)").reset_index()
    display(Markdown(f"**{var}**"))
    display(piv)
"""))

cells.append(md(r"""
### 2.6 Resultados caso a caso

As tabelas abaixo trazem os valores originais, o desfecho real, as probabilidades posteriores de cada rede e a classe predita com limiar de 50%. As probabilidades estão em porcentagem. **F** = Fracasso; **S** = Sucesso.
"""))

cells.append(code(r"""
def nb_case_view(df, p1, p2):
    out = df[["ID","NIF","RR","VT","RSBI","Resultado"]].copy().reset_index(drop=True)
    out["Real"] = out["Resultado"].map({"Fracasso":"F", "Sucesso":"S"})
    out["R1 P(S)%"] = (p1["p_sucesso"].to_numpy() * 100).round(1)
    out["R1 P(F)%"] = (p1["p_fracasso"].to_numpy() * 100).round(1)
    out["R1 pred."] = p1["predito"].map({"Fracasso":"F", "Sucesso":"S"}).to_numpy()
    out["R2 P(S)%"] = (p2["p_sucesso"].to_numpy() * 100).round(1)
    out["R2 P(F)%"] = (p2["p_fracasso"].to_numpy() * 100).round(1)
    out["R2 pred."] = p2["predito"].map({"Fracasso":"F", "Sucesso":"S"}).to_numpy()
    out["R1"] = p1["predito"].eq(out["Resultado"]).map({True:"✓", False:"✗"}).to_numpy()
    out["R2"] = p2["predito"].eq(out["Resultado"]).map({True:"✓", False:"✗"}).to_numpy()
    return out.drop(columns=["Resultado"])

p1_train = models["Rede 1 (clínica)"].predict(train)
p2_train = models["Rede 2 (Netica)"].predict(train)
p1_test = models["Rede 1 (clínica)"].predict(test)
p2_test = models["Rede 2 (Netica)"].predict(test)
p1_full = models["Rede 1 (clínica)"].predict(full)
p2_full = models["Rede 2 (Netica)"].predict(full)

display(Markdown("**Treinamento (21 casos)**"))
display(nb_case_view(train, p1_train, p2_train))
"""))

cells.append(code(r"""
display(Markdown("**Teste (9 casos)**"))
display(nb_case_view(test, p1_test, p2_test))
"""))

cells.append(md(r"""
### 2.6.1 Passo a passo de uma inferência Naive Bayes — TE3 na Rede 1

Para tornar o cálculo auditável, vamos abrir a inferência do caso **TE3** na Rede 1. O TE3 tem NIF −28, RR 24, VT 310 e RSBI 77. A rotina abaixo mostra, para cada classe, a prior, o estado observado de cada variável, a verossimilhança P(estado | classe) e o logaritmo correspondente. Depois soma os logaritmos, normaliza e compara com a saída da rede.
"""))

cells.append(code(r"""
def explain_nb(case_id, model, df):
    row = df.loc[df.ID == case_id].iloc[0]
    rec = w.discretize(pd.DataFrame([row]), model.method).iloc[0]
    rows = []
    for cls in w.CLASSES:
        rows.append({
            "Classe": cls, "Componente": "prior", "Variável": "—", "Estado": "—",
            "Probabilidade": model.priors[cls], "log": math.log(model.priors[cls]),
        })
        for var in w.VARIABLES:
            state = rec[var]
            prob = model.likelihoods[var][cls][state]
            rows.append({
                "Classe": cls, "Componente": f"P({var} = {state} | classe)",
                "Variável": var, "Estado": state,
                "Probabilidade": prob, "log": math.log(prob),
            })
    tab = pd.DataFrame(rows)
    display(tab.round(6))

    logp = {}
    for cls in w.CLASSES:
        logp[cls] = math.log(model.priors[cls]) + sum(
            math.log(model.likelihoods[var][cls][rec[var]]) for var in w.VARIABLES
        )
    m = max(logp.values())
    p = {c: math.exp(logp[c] - m) for c in w.CLASSES}
    z = sum(p.values())
    p = {c: p[c] / z for c in w.CLASSES}
    resumo = pd.DataFrame([
        {"Classe": c, "log P(classe) + Σ log P(x|classe)": logp[c], "P(classe | evidências)": p[c]}
        for c in w.CLASSES
    ])
    display(resumo.round(6))
    display(Markdown(
        f"**Decisão:** P(Sucesso | evidências) = {p['Sucesso']:.3f} e "
        f"P(Fracasso | evidências) = {p['Fracasso']:.3f}. "
        f"Como o limiar é 50%, a rede prediz **{'Sucesso' if p['Sucesso'] >= .5 else 'Fracasso'}**."
    ))

explain_nb("TE3", models["Rede 1 (clínica)"], full)
"""))

cells.append(md(r"""
### 2.7 Matrizes de confusão e métricas

A classe positiva é **Sucesso**. As matrizes têm linhas (real: Fracasso, Sucesso) e colunas (predito: Fracasso, Sucesso).
"""))

cells.append(code(r"""
display(Markdown("**Figura 3 — matrizes de confusão das redes Naive Bayes (treino e teste).**"))
display(Image(filename=str(FIG / "fig03_matrizes_confusao_nb.png"), width=760))
"""))

cells.append(code(r"""
metrics = pd.read_csv(TAB / "07_nb_metricas.csv")
show = metrics.copy()
for c in ["Acurácia","Acurácia bal.","Sensibilidade","Especificidade","PPV","NPV","F1","Brier","Certeza média"]:
    show[c] = (show[c] * 100).round(1).astype(str) + "%"
show = show.rename(columns={"Certeza média":"Certeza média (max P)"})
display(show[["Modelo","Conjunto","Acurácia","Acurácia bal.","Sensibilidade","Especificidade","PPV","NPV","F1","Brier","Certeza média (max P)","TN","FP","FN","TP"]])
"""))

cells.append(md(r"""
**Leitura.** A Rede 1 tem 85,7% de acerto no treino (acurácia balanceada 84,7%) e 100% no teste; na base completa, 90,0%. A Rede 2 tem o mesmo desempenho no treino (85,7%), mas cai para 77,8% no teste (acurácia balanceada 66,7%), com especificidade de apenas 33,3%: ela libera dois dos três pacientes que fracassariam. A certeza média da Rede 2 é maior (89,5% no treino; 93,1% no teste) — ou seja, ela é mais confiante e mais errada. Esse é um exemplo claro de má calibração.
"""))

cells.append(code(r"""
display(Markdown("**Figura 4 — métricas e calibração das redes Naive Bayes.**"))
display(Image(filename=str(FIG / "fig04_metricas_nb.png"), width=1050))
display(Markdown("**Figura 5b — probabilidades posteriores $P(\\text{Sucesso})$ das duas redes nos 9 casos de teste.**"))
display(Image(filename=str(FIG / "fig05b_probabilidades_teste.png"), width=900))
"""))

cells.append(md(r"""
### 2.8 Limiar de decisão e assimetria de riscos

O limiar padrão de 50% escolhe a classe de maior probabilidade posterior; ele não garante a máxima acurácia balanceada e não incorpora o custo assimétrico do erro. Liberar um paciente que fracassará (falso positivo) pode causar sofrimento, reintubação e aumento de mortalidade; atrasar a liberação (falso negativo) consome leito e aumenta exposição a infecções, mas é potencialmente reversível. Em uma política conservadora, pode-se exigir $P(\text{Sucesso}) \geq 60\%$ ou $75\%$ para liberar. A tabela de sensibilidade mostra o efeito no teste e na base completa.
"""))

cells.append(code(r"""
th = pd.read_csv(TAB / "09_nb_limiares.csv")
th_show = th[th["Conjunto"].isin(["Teste", "Completa"])].copy()
for c in ["acuracia","sensibilidade","especificidade"]:
    th_show[c] = (th_show[c] * 100).round(1)
with pd.option_context("display.max_rows", None):
    display(th_show[["Rede","Conjunto","Limiar","acuracia","sensibilidade","especificidade","fp","fn"]].rename(columns={"acuracia":"Acurácia (%)","sensibilidade":"Sensibilidade (%)","especificidade":"Especificidade (%)","fp":"FP","fn":"FN"}))
"""))

cells.append(md(r"""
Para a Rede 1, o limiar conservador não altera os resultados até 75%, porque as posteriores são extremas (a maioria fica abaixo de 45% ou acima de 75%). Para a Rede 2, subir o limiar para 60–75% não elimina nenhum falso positivo e acrescenta um falso negativo (TR13); só a partir de 80% o TE1 deixa de ser liberado, enquanto TR1 e TE3 continuam como falsos positivos. A escolha do limiar é, portanto, uma decisão de política clínica e não deve ser feita olhando apenas a acurácia.
"""))

cells.append(md(r"""
### 2.9 Por que as redes convergem ou divergem?

- **Convergem** quando o paciente é inequívoco: NIF muito negativo, RR baixo, VT alto e RSBI baixo geram evidência forte em ambas as redes. Exemplos: TR15, TR17, TR20, TE7, TE9.
- **Divergem** nos casos de fronteira, em que a perda de resolução da Rede 2 muda a contagem de verossimilhança. TR4 (NIF −22, RR 30, VT 350, RSBI 86) recebe 44,2% na Rede 1 e 17,0% na Rede 2: ambas retêm, mas com confiança distinta. TR16 (NIF −30, RR 30, VT 250, RSBI 120) é mais dramático: 76,7% na Rede 1 (sucesso) contra 10,9% na Rede 2 (fracasso), com desfecho real de sucesso. TE1 (NIF −15, RR 25, VT 450, RSBI 56) vai de 29,7% (fracasso) na Rede 1 para 77,2% (sucesso) na Rede 2, e o desfecho real é fracasso.
- **TE3** é o caso mais didático: NIF −28 e RR 24 são limítrofes, VT 310 é baixo e RSBI 77 é favorável. A Rede 1 fica em 44,2% (fracasso), a Rede 2 em 90,6% (sucesso) e o sistema fuzzy em 40,05 (fronteira de cautela / Fracasso). O desfecho real é fracasso. A Rede 2 erra por comprimir NIF, RR e VT em apenas duas faixas e multiplicar evidências favoráveis.
"""))

cells.append(code(r"""
d = p1_full[["ID","Resultado","p_sucesso"]].rename(columns={"p_sucesso":"Rede 1"}).merge(
    p2_full[["ID","p_sucesso"]].rename(columns={"p_sucesso":"Rede 2"}), on="ID")
d["Δ"] = (d["Rede 1"] - d["Rede 2"]).abs()
display(d.sort_values("Δ", ascending=False).head(12).round(3))
"""))

cells.append(code(r"""
display(Markdown("**Figura 5 — casos em que as redes mais divergem.**"))
display(Image(filename=str(FIG / "fig05_divergencia_probabilidades.png"), width=760))
display(Markdown("**Netica 7.01 — inferência interativa do caso TE3 na Rede 1 (Fracasso 55,8% / Sucesso 44,2%) e na Rede 2 (Fracasso 9,36% / Sucesso 90,6%).**"))
display(Image(filename=str(ROOT / "netica/evidencias/06_netica_rede1_te3.png"), width=820))
display(Image(filename=str(ROOT / "netica/evidencias/08_netica_rede2_te3.png"), width=820))
"""))

cells.append(md(r"""
### 2.10 Robustez e redundância: validação leave-one-out no treino

A validação leave-one-out refaz priors e CPTs em cada dobra, mantendo a discretização fixa. Ela não substitui o teste externo, mas mostra a estabilidade interna.
"""))

cells.append(code(r"""
loo = pd.read_csv(TAB / "10_loo_variaveis.csv")
for c in ["Acurácia LOO","Sensibilidade LOO","Especificidade LOO"]:
    loo[c] = (loo[c] * 100).round(1)
display(loo.rename(columns={"Acurácia LOO":"Acurácia (%)","Sensibilidade LOO":"Sensibilidade (%)","Especificidade LOO":"Especificidade (%)"}))
"""))

cells.append(md(r"""
O conjunto completo tem 76,2% de acurácia LOO na Rede 1, contra 85,7% para NIF+RR+VT e para NIF isolado. Isso não prova que “NIF sozinho é melhor”; prova que, com 21 casos, as quatro variáveis produzem muitas células de CPT e a redundância aumenta a variância. A Rede 1 mantém as quatro variáveis por fidelidade ao escopo do trabalho; o bom desempenho no teste é uma observação posterior, não um critério de escolha. A análise é um alerta contra concluir que “mais variáveis sempre melhora”.
"""))

cells.append(md(r"""
## 3. Sistema de inferência fuzzy Mamdani

### 3.1 Arquitetura e operadores

O FIS tem quatro entradas (`NIF`, `RR`, `VT`, `RSBI`) e uma saída `Resultado` (0–100) com dois valores linguísticos possíveis: `Fracasso` e `Sucesso`. O método é Mamdani, com:

| Etapa | Operador adotado | Implementação |
|---|---|---|
| Conectivo E | mínimo (`min`) | `μ_A(x) ∧ μ_B(x) = min(μ_A, μ_B)` |
| Conectivo OU | máximo (`max`) | não utilizado nas regras (todas conjuntivas) |
| Implicação | mínimo (`min`) | corte do consequente em α |
| Agregação | máximo (`max`) | união das regiões recortadas |
| Defuzzificação | centroide (COA) | $y^* = \int y\,\mu_{agg}(y)dy / \int \mu_{agg}(y)dy$ |

A implementação manual foi validada contra `scikit-fuzzy` (diferença máxima inferior a 10⁻⁵ no centroide).
"""))

cells.append(md(r"""
### 3.2 Universos, termos linguísticos e funções de pertinência

As funções trapezoidais nas extremidades representam saturação fisiológica; as triangulares no centro representam zonas de transição. Na saída `Resultado` (0–100), os dois conjuntos `Fracasso` (`[0, 0, 35, 65]`) e `Sucesso` (`[35, 65, 100, 100]`) se cruzam em 50 ($\mu = 0{,}5$). Os parâmetros foram definidos a priori, com base na literatura, e não ajustados aos 30 casos.
"""))

cells.append(code(r"""
mfs = pd.read_csv(TAB / "11_fuzzy_mfs.csv")
display(mfs)
display(Markdown("**Figura 7 — funções de pertinência de entrada e saída (`Resultado`: `Fracasso` e `Sucesso`).**"))
display(Image(filename=str(FIG / "fig07_funcoes_pertinencia.png"), width=1050))
"""))

cells.append(md(r"""
**Como as sobreposições capturam a incerteza.** O NIF tem 50% de pertinência a “Forte” e 50% a “Limítrofe” em −30 cm.H₂O, exatamente o limiar clássico; em −25, é totalmente limítrofe. O RSBI tem 50% em 105 (favorável vs. limítrofe) e 50% em 135 (limítrofe vs. desfavorável), criando uma faixa de cautela entre 105 e 150. A RR tem 50% em 26 (adequada vs. limítrofe) e 34 (limítrofe vs. elevada), em vez de um salto em 30. O VT tem transição entre 300 e 400 mL, capturando a diferença entre “baixo” e “adequado” na região de 5 mL/kg. Na saída `Resultado`, `Fracasso` e `Sucesso` compartilham a zona de transição entre 35 e 65.
"""))

cells.append(md(r"""
### 3.3 Base de regras compacta (12 regras)

A base de regras foi estruturada em **12 regras compactas** (4 concluindo `Sucesso` e 8 concluindo `Fracasso`), redigidas no formato **SE…ENTÃO** com omissão das variáveis irrelevantes em cada regra (equivalente ao uso de curingas sobre o espaço cartesiano $3^4 = 81$ combinações). Essa formulação evita a explosão combinatória e torna a auditoria clínica imediata:

- **Regras de `Sucesso` (R1–R4):** exigem combinação favorável de força muscular (`NIF`), padrão respiratório (`RR`, `RSBI`) e, quando `NIF` é `Limítrofe`, volume corrente `Adequado` (R4).
- **Regras de `Fracasso` (R5–R12):** atuam como travas de segurança clínica — qualquer fator crítico isolado (`NIF = Fraco`, `RR = Elevada`, `RSBI = Desfavorável`) ou dupla condição limítrofe/desfavorável ativa `Fracasso`.

A pontuação clínica de referência (`NIF`: +3/0/−2; `RR`: +2/0/−2; `VT`: +1/−1/0; `RSBI`: +2/0/−2) permanece coerente com o consequente de cada regra. A base é **completa** (cobre 100% do domínio de entrada) e **consistente** (sem contradições diretas nos antecedentes).
"""))

cells.append(code(r"""
rules = pd.read_csv(TAB / "12_fuzzy_regras.csv")
print(f"Total de regras: {len(rules)} | distribuição de consequentes: {rules['Consequente'].value_counts().to_dict()}")
with pd.option_context("display.max_rows", None, "display.max_colwidth", None):
    display(rules[["Regra","NIF","RR","VT","RSBI","Pontuação","Consequente"]])
    display(Markdown("**Base completa no formato SE…ENTÃO (12 regras compactas):**"))
    display(rules[["Regra","SE...ENTÃO"]])
"""))

cells.append(md(r"""
**Figura 8 — distribuição dos consequentes e geometria da base de regras.** O painel esquerdo mostra a distribuição conservadora das **12 regras compactas** para a variável `Resultado`: 8 regras concluem `Fracasso` e 4 concluem `Sucesso`. O painel direito fixa `VT` adequado e `RSBI` favorável e mostra que a pontuação cresce de forma monótona quando o `NIF` passa de fraco para forte e a `RR` de elevada para adequada.
"""))

cells.append(code(r"""
display(Markdown("**Figura 8 — consequentes das 12 regras compactas (`Fracasso` vs. `Sucesso`) e pontuação NIF × RR.**"))
display(Image(filename=str(FIG / "fig08_regras_consequentes.png"), width=1050))
"""))

cells.append(md(r"""
### 3.4 Completude e consistência

A verificação abaixo percorre uma malha densa de cada universo e confirma que existe pelo menos um termo com pertinência positiva em cada ponto. Embora compacta (12 regras), a base cobre todas as 81 combinações do espaço cartesiano $3^4$: qualquer ponto do domínio ativa ao menos uma regra e produz centroide finito.
"""))

cells.append(code(r"""
# Cobertura dos universos: para cada variável, max_t mu_t(x) > 0 em toda a malha
for var in w.VARIABLES:
    # Usa a biblioteca de plotting para obter as MFs na malha do universo.
    mus = np.vstack([plotting._mf_array(var, t) for t in w.FUZZY_TERMS[var]])
    mx = mus.max(axis=0)
    print(f"{var}: mínimo da cobertura máxima = {mx.min():.3f}; pontos sem cobertura = {(mx <= 1e-12).sum()}")
    assert (mx > 1e-12).all(), f"{var} tem pontos sem cobertura"

# Amostra de 2.000 pontos: nenhuma saída ausente/NaN
rng = np.random.default_rng(42)
for _ in range(2000):
    x = {
        "NIF": rng.uniform(-60, 0),
        "RR": rng.uniform(5, 60),
        "VT": rng.uniform(100, 1000),
        "RSBI": rng.uniform(0, 300),
    }
    val = w.mamdani_evaluate(x).centroid
    assert np.isfinite(val)
# Cantos do espaço de entrada (extremos de cada universo)
for corner in itertools.product(*(w.FUZZY_RANGES[v] for v in w.VARIABLES)):
    assert np.isfinite(w.mamdani_evaluate(dict(zip(w.VARIABLES, corner))).centroid)
print("2.000 avaliações aleatórias e 16 cantos do domínio: todas com regra ativada e saída finita.")
"""))

cells.append(md(r"""
### 3.5 Avaliação na base completa (30 casos)

Tanto com limiar 50 (ponto médio entre `Fracasso` e `Sucesso`) quanto com limiar conservador 60, o sistema classifica os mesmos **26/30 casos** corretamente (**17/21** no treino e **9/9** no teste).

> **Nota sobre platôs do centroide Mamdani:** quando apenas um dos dois trapézios de saída (`Fracasso` ou `Sucesso`) é ativado com $\alpha = 1{,}0$, o centroide assume exatamente **25,75** (`Fracasso`) ou **74,25** (`Sucesso`) — valores que refletem o centro de massa do trapézio inteiro, e não “erro numérico”. Quando há ativação parcial ($\alpha < 1$) ou competição simultânea entre regras de `Fracasso` e `Sucesso` (como em **TE3 = 40,05**), o centroide percorre continuamente a faixa intermediária.
"""))

cells.append(code(r"""
fuzzy = w.predict_fuzzy(full)
show = fuzzy[["ID","NIF","RR","VT","RSBI","Resultado","crisp","decisao","acerto_conservador"]].copy()
show["crisp"] = show["crisp"].round(2)
show["Resultado"] = show["Resultado"].map({"Fracasso":"F", "Sucesso":"S"})
show["decisao"] = show["decisao"].map({"Fracasso":"Fracasso", "Cautela":"Cautela", "Sucesso":"Sucesso"})
show["acerto_conservador"] = show["acerto_conservador"].map({True:"✓", False:"✗"})
display(show.rename(columns={"crisp":"Resultado (crisp)","decisao":"Decisão sugerida","acerto_conservador":"Acerto"}))
"""))

cells.append(md(r"""
**Figura 11 — saída `Resultado` nos 30 casos.** O eixo vertical é o centroide defuzzificado da variável `Resultado`; as cores indicam o desfecho real.
"""))

cells.append(code(r"""
display(Markdown("**Figura 11 — saída Fuzzy (`Resultado`) nos 30 casos, com os limiares 40, 50 e 60.**"))
display(Image(filename=str(FIG / "fig11_fuzzy_30casos.png"), width=1050))
"""))

cells.append(md(r"""
**Figura 12 — matrizes de confusão do Fuzzy.** Ambas as matrizes (limiar 50 e limiar 60) obtêm 26/30 de acurácia global (86,7%) e 9/9 no teste (100%).
"""))

cells.append(code(r"""
display(Markdown("**Figura 12 — matrizes de confusão do sistema Fuzzy (limiares 50 e 60).**"))
display(Image(filename=str(FIG / "fig12_fuzzy_confusao.png"), width=720))
"""))

cells.append(code(r"""
fm = pd.read_csv(TAB / "14_fuzzy_metricas.csv")
show = fm.copy()
cols_pct = [c for c in ["acuracia","acuracia_balanceada","sensibilidade","especificidade","ppv","npv","f1"] if c in show.columns]
for c in cols_pct:
    show[c] = (show[c] * 100).round(1)
display(show[["Conjunto","Limiar"] + cols_pct + ["tn","fp","fn","tp"]].rename(columns={
    "acuracia":"Acurácia (%)","acuracia_balanceada":"Acurácia bal. (%)","sensibilidade":"Sensibilidade (%)","especificidade":"Especificidade (%)",
    "ppv":"PPV (%)","npv":"NPV (%)","f1":"F1 (%)","tn":"TN","fp":"FP","fn":"FN","tp":"TP"}))
"""))

cells.append(md(r"""
**Resultado.** O FIS acerta 17/21 no treino (acurácia 81,0%; acurácia balanceada 79,2%), 9/9 no teste (100,0%) e 26/30 na base completa (86,7%). A sensibilidade é 80,0% e a especificidade 93,3%; há apenas um falso positivo (TR1) e três falsos negativos (TR13, TR14 e TR16).
"""))

cells.append(md(r"""
### 3.6 Superfícies de decisão e execução no QtFuzzyLite 4.0

As superfícies mostram como a saída `Resultado` varia com dois pares de variáveis, mantendo as demais fixas. As linhas escuras marcam 40 e 60. Em seguida, exibem-se as capturas reais do mesmo modelo (`fuzzy/weaning_mamdani.fll`) executado na interface gráfica **QtFuzzyLite 4.0**.
"""))

cells.append(md(r"""
**Figura 9 — superfície NIF × RR.** Com VT e RSBI fixos em valores adequados, a saída `Resultado` desloca-se de `Fracasso` para `Sucesso` quando o NIF se torna mais negativo e a RR diminui.
"""))

cells.append(code(r"""
display(Markdown("**Figura 9 — superfície de decisão em função de NIF e RR.**"))
display(Image(filename=str(FIG / "fig09_superficie_nif_rr.png"), width=720))
"""))

cells.append(md(r"""
**Figura 10 — superfície VT × RSBI.** Com NIF e RR fixos, volumes muito baixos mantêm a saída em `Fracasso` mesmo quando o RSBI parece favorável; volumes adequados/altos e RSBI baixo se combinam na região de `Sucesso`.
"""))

cells.append(code(r"""
display(Markdown("**Figura 10 — superfície de decisão em função de VT e RSBI.**"))
display(Image(filename=str(FIG / "fig10_superficie_vt_rsbi.png"), width=720))
display(Markdown("**Execução real no QtFuzzyLite 4.0 — aba `Setup` (`weaning_mamdani.fll`, 12 regras compactas válidas, saída `Resultado`: `Fracasso` e `Sucesso`) e 4 mapas `Surface 2D` (`NIF×RR`, `VT×RSBI`, `NIF×RSBI`, `RR×RSBI`).**"))
for img_name in (
    "01_qtfuzzylite_setup.png",
    "02_qtfuzzylite_surface_nif_rr.png",
    "03_qtfuzzylite_surface_vt_rsbi.png",
    "04_qtfuzzylite_surface_nif_rsbi.png",
    "05_qtfuzzylite_surface_rr_rsbi.png",
):
    w_px = 450 if "surface" in img_name else 980
    display(Image(filename=str(ROOT / "fuzzy/evidencias" / img_name), width=w_px))
"""))

cells.append(md(r"""
## 4. Estudos de caso

Os três casos foram escolhidos para cobrir: sucesso claro (`TE5`), fracasso claro (`TR10`) e situação crítica/duvidosa (`TE3`). Para cada um, apresentamos fuzzificação, regras ativadas passo a passo, agregação, centroide, interpretação clínica e a simulação interativa na aba `Control` do **QtFuzzyLite 4.0**.
"""))

cells.append(code(r"""
# Fuzzificação, regras ativadas, agregação e centroide de um caso (Figuras 11+2k e 12+2k).
def show_case(case_id, k, qt_img):
    case = full.loc[full.ID == case_id].iloc[0]
    x = {v: float(case[v]) for v in w.VARIABLES}
    res = w.mamdani_evaluate(x)
    entradas = ", ".join(f"{v} = {x[v]:g}" for v in w.VARIABLES)
    display(Markdown(f"**Entradas:** {entradas} · **desfecho real:** {case['Resultado']}"))
    display(Markdown("**Fuzzificação** — pertinência de cada entrada a cada termo linguístico:"))
    display(w.fuzzify_dataframe(res).round(4))
    act = pd.DataFrame(res.activations)
    for col in ["NIF", "RR", "VT", "RSBI", "Consequente"]:
        act[col] = act[col].map(lambda t: w.TERM_DISPLAY.get(t, "—"))
    display(Markdown(f"**Regras ativadas** — {len(act)} regras com α = min(μ) > 0:"))
    display(act[["Regra","NIF","RR","VT","RSBI","mu_NIF","mu_RR","mu_VT","mu_RSBI","alpha","Consequente"]].round(4))
    display(Markdown("**Agregação (máximo) e defuzzificação (centroide):**"))
    display(pd.DataFrame([w.defuzz_summary(res)]).round(4))
    display(Markdown(f"**Figura {11 + 2 * k} — {case_id}: fuzzificação das quatro entradas.**"))
    display(Image(filename=str(FIG / f"fig{11 + 2 * k}_caso{k}_entradas.png"), width=1000))
    display(Markdown(f"**Figura {12 + 2 * k} — {case_id}: agregação Mamdani, região recortada e centroide.**"))
    display(Image(filename=str(FIG / f"fig{12 + 2 * k}_caso{k}_saida.png"), width=1000))
    display(Markdown(f"**QtFuzzyLite 4.0 (`Control`) — simulação interativa para {case_id}:**"))
    display(Image(filename=str(ROOT / "fuzzy/evidencias" / qt_img), width=980))
"""))

cells.append(md(r"""
### 4.1 Caso 1 — TE5: sucesso claro

Paciente de teste com todos os indicadores favoráveis ou adequados; o sistema deve confirmar `Sucesso`.
"""))

cells.append(code(r"""
show_case("TE5", 1, "06_qtfuzzylite_caso1_te5_sucesso.png")
"""))

cells.append(md(r"""
**Caso 1 — interpretação.** NIF −48 é `Forte` ($\mu=1{,}000$); RR 24 é predominantemente `Adequada` ($0{,}750$) com $0{,}250$ de `Limítrofe`; VT 380 é $0{,}200$ `Baixo` e $0{,}457$ `Adequado`; RSBI 63 é totalmente `Favorável` ($1{,}000$). Duas regras disparam (**R1** com $\alpha=0{,}750$ e **R2** com $\alpha=0{,}250$), ambas concluindo `Sucesso`. A agregação recorta `Sucesso` em $0{,}750$ (área $40{,}312$, momento $2932{,}031$), resultando no centroide **72,73** (`1,000/Sucesso`; `72,732` no QtFuzzyLite 4.0), acima de 60 e de 50: recomendação clara de `Sucesso`, coerente com o desfecho real.
"""))

cells.append(md(r"""
### 4.2 Caso 2 — TR10: fracasso claro

Paciente com padrão respiratório rápido e superficial extremo; todas as regras ativadas devem levar a `Fracasso`.
"""))

cells.append(code(r"""
show_case("TR10", 2, "07_qtfuzzylite_caso2_tr10_fracasso.png")
"""))

cells.append(md(r"""
**Caso 2 — interpretação.** NIF −20 divide-se igualmente entre `Fraco` ($0{,}500$) e `Limítrofe` ($0{,}500$); RR 33 é $0{,}625$ `Limítrofe` e $0{,}375$ `Elevada`; VT 124 é totalmente `Baixo` ($1{,}000$); RSBI 266 é totalmente `Desfavorável` ($1{,}000$). Cinco regras disparam (**R5** = $0{,}500$, **R6** = $0{,}375$, **R7** = $1{,}000$, **R8** = $0{,}500$ e **R10** = $0{,}500$), todas concluindo `Fracasso`. Como R7 atinge $\alpha=1{,}000$, a agregação preserva o trapézio integral de `Fracasso` (área $50{,}000$, momento $1287{,}500$), produzindo o centroide **25,75** (`1,000/Fracasso`; `25,750` no QtFuzzyLite 4.0): recomendação inequívoca de `Fracasso`, coerente com o desfecho real.
"""))

cells.append(md(r"""
### 4.3 Caso 3 — TE3: crítico/duvidoso

NIF e RR em transição, VT baixo e RSBI favorável: sinais conflitantes, redes divergentes e desfecho real de fracasso.
"""))

cells.append(code(r"""
show_case("TE3", 3, "08_qtfuzzylite_caso3_te3_duvidoso.png")
"""))

cells.append(md(r"""
**Caso 3 — interpretação.** O paciente tem NIF −28 ($0{,}700$ `Limítrofe` e $0{,}300$ `Forte`), RR 24 ($0{,}750$ `Adequada` e $0{,}250$ `Limítrofe`), VT 310 ($0{,}900$ `Baixo` e $0{,}057$ `Adequado`) e RSBI 77 (totalmente `Favorável`, $1{,}000$). Cinco regras disparam em competição entre os dois consequentes: **R8** (`Fracasso`, $\alpha=0{,}700$) e **R10** (`Fracasso`, $\alpha=0{,}250$) contrapõem-se a **R1** (`Sucesso`, $\alpha=0{,}300$), **R2** (`Sucesso`, $\alpha=0{,}250$) e **R4** (`Sucesso`, $\alpha=0{,}057$). A agregação forma um polígono bimodal (recorte de `Fracasso` em $0{,}700$ e `Sucesso` em $0{,}300$; área $50{,}000$, momento $2002{,}400$), gerando o centroide **40,05** (`0,832/Fracasso + 0,168/Sucesso`; `40,048` no QtFuzzyLite 4.0), na fronteira da zona de cautela e abaixo de 50. A decisão é **Fracasso / não liberar**, acertando o desfecho real onde a Rede 2 errou (90,6% `Sucesso`).
"""))

cells.append(md(r"""
## 5. Comparação geral e conclusão

### 5.1 Desempenho na base completa
"""))

cells.append(code(r"""
final = pd.read_csv(TAB / "20_comparacao_final.csv")
show = final.copy()
cols_f = [c for c in ["Acurácia","Acurácia bal.","Sensibilidade","Especificidade","PPV","NPV","F1"] if c in show.columns]
for c in cols_f:
    show[c] = (show[c] * 100).round(1)
display(show[["Modelo"] + cols_f + ["TN","FP","FN","TP"]])
display(Markdown("**Figura 19 — comparação final dos três modelos na base completa.**"))
display(Image(filename=str(FIG / "fig19_comparacao_final.png"), width=850))
"""))

cells.append(md(r"""
### 5.2 Interpretação comparativa

| Critério | Naive Bayes 1 (clínica) | Naive Bayes 2 (Netica) | Fuzzy Mamdani |
|---|---|---|---|
| Acurácia (30) | **90,0%** | 83,3% | 86,7% |
| Acurácia balanceada (30) | **90,0%** | 83,3% | 86,7% |
| Acurácia no teste (9) | **100,0%** (9/9) | 77,8% (7/9) | **100,0%** (9/9) |
| Sensibilidade | **86,7%** | **86,7%** | 80,0% |
| Especificidade | **93,3%** | 80,0% | **93,3%** |
| Calibração | Brier 0,078 | Brier 0,137 | não probabilístico |
| Transparência | CPTs e 12 faixas clínicas | CPTs e 2 faixas Netica | **12 regras linguísticas compactas** |
| Incerteza no limiar | posteriores extremas | posteriores extremas | saída graduada + cautela |
| Principal fragilidade | dupla contagem RR/VT/RSBI | perda de resolução | escolha manual das MFs e limiar |

- **Desempenho:** a Rede 1 é a melhor na base completa (27/30) e gabarita o teste (9/9); a Rede 2 é mais confiante e menos específica (25/30; 7/9 no teste); o fuzzy é competitivo (26/30; 9/9 no teste), empatando na especificidade (93,3%) e protegendo contra falsos positivos.
- **Interpretabilidade:** a Rede 1 permite ler diretamente “P(NIF forte | sucesso)” e explicar cada contribuição; o fuzzy permite auditar diretamente as 12 regras linguísticas no formato “SE NIF é Forte E RR é Adequada…”. A Rede 2 é mais simples, mas essa simplicidade custa informação clínica.
- **Incerteza:** as redes produzem probabilidades, mas com 21 casos e forte dependência entre preditores elas são pontiagudas demais. O fuzzy produz um potencial graduado, onde a faixa 40–60 funciona como zona explícita de cautela (exemplificada por TE3 = 40,05).
- **Correlação:** o fuzzy não exige independência, mas ao usar RR, VT e RSBI também reforça o mesmo padrão. A diferença é que isso é uma decisão de modelagem visível, não uma hipótese probabilística implícita.
- **Risco clínico:** nenhum modelo deve liberar um paciente sozinho. A recomendação é usar a Rede 1 como triagem, o fuzzy como segunda opinião nos casos limítrofes e sempre integrar gasometria, consciência, secreção, tosse, estabilidade hemodinâmica e avaliação do fisioterapeuta/médico.

### 5.3 Limitações

1. Amostra pequena e histórica (1992), com apenas 30 tentativas e 9 no teste; os intervalos de confiança são largos.
2. Variáveis medidas em um único momento, sem tendência temporal.
3. Não há informações de sedação, balanço hídrico, comorbidades, força de tosse, PaO₂/FiO₂ ou modo ventilatório.
4. A definição de sucesso/fracasso e o protocolo de interrupção não são detalhados na base.
5. A Rede 2 foi discretizada pelo Netica 7.01 e aprendida pelo Netica C API 5.04; versões e opções diferentes podem alterar os resultados. O leave-one-out mantém os cortes fixos e não valida novamente a discretização em cada dobra.
6. O limiar fuzzy de 60 foi adotado por política conservadora; outros serviços podem adotar 50 ou 70, com trade-offs distintos.
7. Não foi feita validação externa nem análise de curvas de decisão.

### 5.4 Conclusão

As duas abordagens são complementares. A Rede 1, com cortes clínicos e suavização, obteve o melhor equilíbrio preditivo (90,0% global; 9/9 no teste) e a maior transparência probabilística. O sistema Mamdani entregou uma saída graduada, uma zona explícita de cautela e desempenho competitivo (86,7% global; 9/9 no teste) com apenas **12 regras compactas** totalmente auditáveis no **QtFuzzyLite 4.0**. A Rede 2 mostrou que a discretização automática em duas faixas no **Netica 7.01** é rápida e reproduzível, porém perde resolução clínica e pode produzir confiança excessiva em casos limítrofes.

Para uso real, a recomendação é um fluxo em duas camadas: (i) Rede 1 para estimar probabilidade e sinalizar casos claros; (ii) fuzzy para os casos de fronteira, com decisão humana obrigatória. O caso TE3 — sucesso previsto pela Rede 2 (90,6%), cautela/fracasso pelo fuzzy (40,05) e fracasso real — resume por que a decisão não pode ser terceirizada a um único modelo.
"""))

cells.append(md(r"""
## 6. Reprodutibilidade

Arquivos principais:

- `src/weaning_lib.py` — dados, verificação dos arquivos da disciplina, discretizações, Naive Bayes, Mamdani manual, métricas e validações.
- `src/plotting.py` — todas as 20 figuras.
- `src/generate_artifacts.py` — gera tabelas, figuras e `report/resultados.json`.
- `src/export_models.py` — exporta redes Netica (`.dne`) com nó-alvo `Resultado` e FIS (`.fis`, `.fcl`, `.fll`).
- `netica/README_NETICA.md` e `netica/evidencias/` — execução real no Netica Application 7.01 (`Auto-Discretize Nodes`, `Process Cases` com `CTRL_Respira.txt`, capturas e validação pela C API 5.04).
- `fuzzy/weaning_mamdani.{fis,fcl,fll}` e `fuzzy/evidencias/` — modelo e capturas reais no QtFuzzyLite 4.0 (`Setup`, 4 `Surface 2D`, `Control` para TE5, TR10 e TE3).
- `report/Relatorio_Weaning_Bayes_Fuzzy.pdf` — relatório técnico.

Para reproduzir tudo (tabelas, figuras, modelos, notebook, PDF e pacote de entrega): `./run_all.sh`.
"""))

cells.append(code(r"""
# Validações finais de consistência
print("Arquivos originais da disciplina conferidos:", w.verify_discipline_files())
print("Naive Bayes manual x sklearn (erro absoluto máximo):")
print("  Rede 1:", f"{w.nb_validation_against_sklearn(train, 'clinical'):.2e}")
print("  Rede 2:", f"{w.nb_validation_against_sklearn(train, 'netica'):.2e}")
print("Netica nativo (C API 5.04) x Python (ambas as redes, treino e teste):")
val = w.validate_against_netica()
val["Erro máximo |P(S) Netica − Python|"] = val["Erro máximo |P(S) Netica − Python|"].map("{:.2e}".format)
val["Tolerância"] = val["Tolerância"].map("{:.0e}".format)
display(val)
print("Fuzzy manual x scikit-fuzzy:")
fz = w.validate_fuzzy_against_skfuzzy(full, ["TE1", "TE3", "TR10"])
fz["Diferença absoluta"] = fz["Diferença absoluta"].map("{:.1e}".format)
display(fz)
print("Regras:", len(w.FUZZY_RULES), "| consequentes:", pd.Series([r['consequent'] for r in w.FUZZY_RULES]).value_counts().to_dict())
"""))

cells.append(md(r"""
## 7. Checklist de conformidade com o trabalho

A tabela abaixo confronta o que foi pedido com o que está efetivamente entregue. Os status são: **cumprido**, **cumprido com ressalva** e **não se aplica**.

| Item do trabalho | Status | Onde está | Observação |
|---|---|---|---|
| Descrição do processo e riscos da interrupção prematura/tardia | **Cumprido** | Seção 1 e relatório | Contexto clínico e variáveis |
| Duas redes Naive Bayes | **Cumprido** | Seções 2.1–2.6 | Rede 1 clínica e Rede 2 Netica (`Resultado`) |
| Variáveis NIF, RR, VT e RSBI | **Cumprido** | Seções 1 e 2 | As quatro variáveis nas duas redes |
| Discretização da Rede 1 por conhecimento especialista | **Cumprido** | Seção 2.3 e Figura 6 | 12 faixas justificadas |
| Discretização automática da Rede 2 via Netica | **Cumprido** | Seção 2.4 e `netica/evidencias/` | `Auto-Discretize Nodes` executado no Netica 7.01 com 21 casos; rede nativa, capturas e hashes |
| Avaliação das redes no treino e no teste | **Cumprido** | Seções 2.6–2.7 | Tabelas caso a caso, matrizes, `Process Cases` no Netica 7.01 (`CTRL_Respira.txt`) |
| Probabilidades de sucesso e fracasso por caso | **Cumprido** | Seção 2.6 e Figura 5b | Tabelas completas (treino e teste) |
| Comparação entre as duas redes | **Cumprido** | Seções 2.7–2.10 e Figura 5 | Acurácia, acurácia balanceada, sensibilidade, especificidade, Brier e certeza |
| Discussão da independência condicional com RR, VT e RSBI | **Cumprido** | Seção 2.2 e 2.10 | Correlação e leave-one-out |
| Sistema de inferência Mamdani | **Cumprido** | Seção 3 | min/min/max/centroide; validado no QtFuzzyLite 4.0 |
| Variáveis, termos linguísticos e funções de pertinência | **Cumprido** | Seções 3.2–3.3 e Figura 7 | Parâmetros exatos, justificativa e 4 mapas `Surface 2D` no QtFuzzyLite 4.0 |
| Base de regras completa e consistente | **Cumprido** | Seções 3.3–3.4 | **12 regras compactas** cobrindo integralmente as 81 combinações ($3^4$) |
| Avaliação do Fuzzy nos 30 casos | **Cumprido** | Seção 3.5 | Tabela, Figura 11, Figura 12 e métricas |
| Limiar de decisão justificado | **Cumprido** | Seções 3.5 e 5.3 | Limiar conservador 60; 50 como referência |
| Exemplo de sucesso | **Cumprido** | Seção 4.1 (TE5) | Figuras 13–14 + captura QtFuzzyLite 4.0 |
| Exemplo de fracasso | **Cumprido** | Seção 4.2 (TR10) | Figuras 15–16 + captura QtFuzzyLite 4.0 |
| Exemplo crítico/duvidoso | **Cumprido** | Seção 4.3 (TE3) | Figuras 17–18 + captura QtFuzzyLite 4.0 |
| Fuzzificação, regras ativadas e defuzzificação dos exemplos | **Cumprido** | Seções 4.1–4.3 | Tabelas, polígono agregado e aba `Control` do QtFuzzyLite 4.0 |
| Relatório em PDF com gráficos e tabelas | **Cumprido** | `report/` | Regenerado com autoria da dupla e evidências Netica 7.01 + QtFuzzyLite 4.0 |
| Código-fonte e artefatos | **Cumprido** | `src/`, `netica/`, `fuzzy/`, `tables/`, `figures/` | Reproduzível por `run_all.sh` |
| Envio de PDF e arquivos por e-mail | **Pendente de envio** | Pacote de entrega | Disponibilidade dos arquivos não comprova envio |

### 7.1 Ressalvas que merecem transparência

1. **Programas gráficos:** tanto o **Netica Application 7.01** (discretização automática, compilação, inferência TE3 e `Cases > Process Cases` com `CTRL_Respira.txt`) quanto o **QtFuzzyLite 4.0** (12 regras em `.fll`, 4 superfícies 2D e simulações de TE5, TR10 e TE3) foram executados diretamente, com capturas reais e validação cruzada frente ao Python.
2. **Vídeos:** o conteúdo clínico foi fundamentado na base de conhecimento, no artigo de Ashutosh et al. (1992), em Yang & Tobin (1991), no III Consenso, em Silva et al. (2015), Hess & Kacmarek e Dräger. Os vídeos indicados no enunciado não são citados um a um.
3. **Amostra:** o teste tem apenas 9 casos; os intervalos de confiança são largos e não há validação externa. O caso TR1 mostra que as quatro variáveis não explicam todos os desfechos.

### 7.2 Conclusão da conformidade

Os requisitos técnicos de modelagem, discretização via Netica, execução nos programas gráficos (Netica 7.01 e QtFuzzyLite 4.0), avaliação e documentação estão implementados e verificados. O envio por e-mail é uma etapa de entrega separada e permanece pendente.
"""))

cells.append(md(r"""
---

**Nota final.** Este notebook foi executado de ponta a ponta. As tabelas em CSV preservam os valores completos; as figuras em PNG estão em alta resolução. O relatório em PDF apresenta a mesma análise em formato técnico-científico, com apêndices de regras, CPTs, evidências dos programas gráficos e instruções de reprodução.
"""))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"},
    "title": "Weaning — Naive Bayes e Fuzzy",
}

out = ROOT / "notebooks" / "Weaning_Bayes_Fuzzy.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("Notebook escrito:", out)

manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client = NotebookClient(nb, km=manager, timeout=900, resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbf.write(nb, out)
print("Notebook executado e salvo.")
