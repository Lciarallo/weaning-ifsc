# -*- coding: utf-8 -*-
"""Gera o relatório técnico em HTML e PDF (via Chromium headless)."""
from __future__ import annotations

import base64
import html
import io
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import weaning_lib as w

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
REPORT.mkdir(exist_ok=True)
FIG = ROOT / "figures"
TAB = ROOT / "tables"

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
train, test, full = w.load_data()
nb_train = pd.read_csv(TAB / "05_nb_treino.csv")
nb_test = pd.read_csv(TAB / "06_nb_teste.csv")
nb_metrics = pd.read_csv(TAB / "07_nb_metricas.csv")
cpt = pd.read_csv(TAB / "08_nb_cpt.csv")
cpt["Rede"] = cpt["Rede"].str.replace(" (clínica)", " clínica", regex=False).str.replace(" (Netica)", " Netica", regex=False)
th_nb = pd.read_csv(TAB / "09_nb_limiares.csv")
loo = pd.read_csv(TAB / "10_loo_variaveis.csv")
mfs = pd.read_csv(TAB / "11_fuzzy_mfs.csv")
rules = pd.read_csv(TAB / "12_fuzzy_regras.csv")
fuzzy = pd.read_csv(TAB / "13_fuzzy_30casos.csv")
fuzzy_metrics = pd.read_csv(TAB / "14_fuzzy_metricas.csv")
fuzzy_th = pd.read_csv(TAB / "15_fuzzy_limiares.csv")
case_fuzz = pd.read_csv(TAB / "16_caso_fuzzificacao.csv")
case_rules = pd.read_csv(TAB / "17_caso_regras.csv")
case_def = pd.read_csv(TAB / "18_caso_defuzz.csv")
case_meta = pd.read_csv(TAB / "19_caso_meta.csv")
final = pd.read_csv(TAB / "20_comparacao_final.csv")
ci = pd.read_csv(TAB / "21_intervalos_confianca.csv")
desc = pd.read_csv(TAB / "01_dados_descritivos.csv")
audit = pd.read_csv(TAB / "04_auditoria_netica.csv")
clinical = pd.read_csv(TAB / "02_discretizacao_clinica.csv")
compare = pd.read_csv(TAB / "03_comparacao_limiares.csv")

FIGNUM: dict[str, int] = {}
TABNUM: dict[str, int] = {}


def e(x) -> str:
    return html.escape(str(x), quote=False)


def n(x: float, d: int = 1) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{d}f}".replace(".", ",")


def pct(x: float, d: int = 1) -> str:
    return n(100 * float(x), d) + "%"


def pct_cell(x: float, d: int = 1) -> str:
    return f'<td class="num">{pct(x, d)}</td>'


def num_cell(x: float, d: int = 1) -> str:
    return f'<td class="num">{n(x, d)}</td>'


def img_uri(path: Path, max_width: int = 1800) -> str:
    im = Image.open(path).convert("RGB")
    if im.width > max_width:
        ratio = max_width / im.width
        im = im.resize((max_width, int(im.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def fig(key: str, filename: str, caption: str, source: str = "Fonte: elaboração própria com os dados do trabalho (Ashutosh et al., 1992).", width: str = "100%") -> str:
    FIGNUM[key] = len(FIGNUM) + 1
    return f"""
<figure id="fig-{key}">
  <img src="{img_uri(FIG / filename)}" style="width:{width}">
  <figcaption><span class="figlabel">Figura {FIGNUM[key]}</span> — {caption}
  <span class="src">{source}</span></figcaption>
</figure>
"""


def tab(key: str, caption: str, head: str, body: str, source: str | None = None, cls: str = "", colgroup: str = "", note: str | None = None) -> str:
    TABNUM[key] = len(TABNUM) + 1
    src = f'<div class="tablesrc">{source}</div>' if source else ""
    nt = f'<div class="tablenote">{note}</div>' if note else ""
    return f"""
<div class="tableblock {cls}">
<table id="tab-{key}" class="{cls}">
  <caption><span class="tablabel">Tabela {TABNUM[key]}</span> — {caption}</caption>
  {colgroup}
  <thead>{head}</thead>
  <tbody>{body}</tbody>
</table>
{src}{nt}
</div>
"""


def simple_table(key, caption, headers, rows, source=None, cls="", numeric_from=1, note=None, row_classes=None):
    head = "<tr>" + "".join(f'<th class="{"num" if i >= numeric_from else ""}">{h}</th>' for i, h in enumerate(headers)) + "</tr>"
    body = []
    for i, row in enumerate(rows):
        klass = f' class="{row_classes[i]}"' if row_classes and row_classes[i] else ""
        cells = []
        for j, cell in enumerate(row):
            cells.append(f'<td class="{"num" if j >= numeric_from else ""}">{cell}</td>')
        body.append(f"<tr{klass}>" + "".join(cells) + "</tr>")
    return tab(key, caption, head, "".join(body), source=source, cls=cls, note=note)


def confusion_table(key, caption, cm, source=None, note=None):
    tn, fp, fn, tp = np.asarray(cm).ravel()
    head = '<tr><th></th><th class="num">Predito Fracasso</th><th class="num">Predito Sucesso</th><th class="num">Total</th></tr>'
    body = f"""
<tr><th class="rowhead">Real Fracasso</th><td class="num">{tn}</td><td class="num">{fp}</td><td class="num">{tn+fp}</td></tr>
<tr><th class="rowhead">Real Sucesso</th><td class="num">{fn}</td><td class="num">{tp}</td><td class="num">{fn+tp}</td></tr>
<tr><th class="rowhead">Total</th><td class="num">{tn+fn}</td><td class="num">{fp+tp}</td><td class="num">{tn+fp+fn+tp}</td></tr>
"""
    return tab(key, caption, head, body, source=source, cls="cm", note=note)


def placeholder(key: str, kind: str) -> str:
    if kind == "fig":
        return f"Figura {FIGNUM[key]}"
    return f"Tabela {TABNUM[key]}"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = r"""
@page { size: A4; margin: 19mm 15mm 18mm 15mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 9.5pt; line-height: 1.46; color: #1F2933; }
p { margin: 0 0 5.2pt 0; text-align: justify; }
h1 { font-size: 18pt; line-height: 1.15; color: #12355B; margin: 26pt 0 10pt 0; padding-bottom: 4pt; border-bottom: 2.2pt solid #D98E04; page-break-before: auto; page-break-after: avoid; }
h1.newpage { page-break-before: always; margin-top: 0; }
h1.first { page-break-before: avoid; }
h2 { font-size: 12.4pt; color: #12355B; margin: 13pt 0 5pt 0; page-break-after: avoid; }
h3 { font-size: 10.6pt; color: #23646C; margin: 10pt 0 4pt 0; page-break-after: avoid; }
h4 { font-size: 9.8pt; color: #12355B; margin: 8pt 0 3pt 0; page-break-after: avoid; }
ul, ol { margin: 2pt 0 6pt 0; padding-left: 15pt; }
li { margin: 2pt 0; text-align: justify; }
strong { color: #102A43; }
.small { font-size: 8.3pt; }
.mono { font-family: 'DejaVu Sans Mono', monospace; font-size: 8pt; }
.center { text-align: center; }
.muted { color: #52606D; }
.section-lead { font-size: 10.2pt; color: #334E68; margin-bottom: 8pt; }
.cover { page-break-after: always; padding-top: 12mm; }
.cover .kicker { font-size: 8.5pt; letter-spacing: 0.12em; text-transform: uppercase; color: #52606D; }
.cover .rule { height: 4pt; background: linear-gradient(90deg,#12355B 0 40%, #D98E04 40% 100%); margin: 8pt 0 22pt 0; }
.cover h1.title { page-break-before: avoid; border: none; font-family: 'DejaVu Serif', Georgia, serif; font-size: 27pt; line-height: 1.12; color: #12355B; margin: 0 0 12pt 0; padding: 0; }
.cover .subtitle { font-family: 'DejaVu Serif', Georgia, serif; font-size: 14.5pt; line-height: 1.35; color: #23646C; margin-bottom: 24pt; }
.cover .meta { border-left: 4pt solid #2A9D8F; padding-left: 10pt; margin: 20mm 0 14mm 0; font-size: 10pt; line-height: 1.65; }
.cover .meta b { color: #12355B; }
.cover .abstract { background: #F3F6F8; border-top: 1.5pt solid #D9E2EC; padding: 10pt 12pt; font-size: 8.6pt; line-height: 1.45; }
.cover .abstract b { color: #12355B; }
.cover .foot { margin-top: 16mm; font-size: 8.5pt; color: #52606D; }
.toc { page-break-after: always; }
.toc h1 { page-break-before: avoid; }
.toc ol { list-style: none; padding-left: 0; font-size: 10pt; }
.toc li { margin: 4pt 0; border-bottom: 0.4pt dotted #CBD2D9; }
.toc .lvl2 { padding-left: 14pt; font-size: 9pt; color: #334E68; border-bottom: none; margin: 2pt 0; }
table { width: 100%; border-collapse: collapse; margin: 5pt 0 3pt 0; font-size: 8.1pt; line-height: 1.28; }
caption { caption-side: top; text-align: left; font-size: 8.7pt; color: #12355B; font-weight: bold; padding: 0 0 4pt 0; }
.tablabel, .figlabel { color: #12355B; font-weight: 700; }
thead { display: table-header-group; }
th { background: #12355B; color: #FFFFFF; font-weight: 600; padding: 3pt 3pt; border: 0.4pt solid #12355B; vertical-align: bottom; }
td { padding: 2.5pt 3pt; border-bottom: 0.4pt solid #D9E2EC; vertical-align: top; }
tbody tr:nth-child(even) td { background: #F7F9FB; }
.rowhead { text-align: left; background: #EDF2F7 !important; color: #12355B; font-weight: 600; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
table.tiny { font-size: 7.05pt; line-height: 1.22; }
table.tiny th, table.tiny td { padding: 2pt 2.1pt; }
table.tiny caption { font-size: 8.3pt; }
table.cm { width: 76%; margin-left: auto; margin-right: auto; font-size: 8.6pt; }
table.cm td { text-align: center; }
table.cm th { text-align: center; }
tr.err td { background: #FCE8EA !important; color: #7A1F2B; }
tr.ok td { background: #F2FAF6 !important; }
.tableblock { margin: 7pt 0 10pt 0; page-break-inside: auto; }
.tableblock.keep { page-break-inside: avoid; }
.tablesrc { font-size: 6.9pt; color: #52606D; margin-top: 2pt; }
.tablenote { font-size: 7.2pt; color: #52606D; margin-top: 2pt; font-style: italic; }
figure { margin: 10pt 0 11pt 0; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; height: auto; border: 0.4pt solid #D9E2EC; }
figcaption { font-size: 7.9pt; color: #334E68; text-align: justify; margin-top: 4pt; line-height: 1.35; }
figcaption .src { display: block; font-size: 6.8pt; color: #616E7C; margin-top: 1pt; }
.box { background: #EEF7F6; border-left: 4pt solid #2A9D8F; padding: 7pt 9pt; margin: 8pt 0 10pt 0; font-size: 9pt; }
.box.warn { background: #FDF1F2; border-left-color: #B23A48; }
.box.neutral { background: #F3F6F8; border-left-color: #486581; }
.box h4 { margin: 0 0 3pt 0; font-size: 9.2pt; }
.box p:last-child { margin-bottom: 0; }
.kpis { display: flex; gap: 5pt; margin: 8pt 0 10pt 0; }
.kpi { flex: 1; border: 0.5pt solid #D9E2EC; border-top: 3pt solid #2A9D8F; background: #FAFCFD; padding: 6pt 6pt; }
.kpi b { display: block; font-size: 15pt; color: #12355B; line-height: 1.1; }
.kpi span { font-size: 7.4pt; color: #52606D; }
.formula { font-family: 'DejaVu Serif', Georgia, serif; text-align: center; font-size: 10.1pt; margin: 7pt 0; color: #102A43; }
.refs { font-size: 7.7pt; line-height: 1.28; }
.refs p { margin: 0 0 3.6pt 0; padding-left: 14pt; text-indent: -14pt; text-align: left; }
.rules { column-count: 3; column-gap: 5mm; font-size: 6.15pt; line-height: 1.22; margin-top: 4pt; }
.rule { break-inside: avoid; padding: 0.8pt 0; border-bottom: 0.3pt solid #E4E9EE; text-align: left; }
.rule b { color: #12355B; }
.code { font-family: 'DejaVu Sans Mono', monospace; background: #F5F7FA; border: 0.4pt solid #D9E2EC; padding: 7pt 8pt; font-size: 7.1pt; line-height: 1.35; white-space: pre-wrap; }
.pagebreak { page-break-before: always; }
.nobreak { page-break-inside: avoid; }
.sig { margin-top: 16pt; font-size: 8.5pt; color: #52606D; }
"""

# ---------------------------------------------------------------------------
# Componentes de texto
# ---------------------------------------------------------------------------
def bullets(items):
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def box(title, text, cls=""):
    return f'<div class="box {cls}"><h4>{title}</h4>{text}</div>'


def fig_topologia():
    return fig("topologia", "fig02_topologia_nb.png", "Topologia das duas redes Naive Bayes e grades de discretização adotadas. O nó pai é compartilhado; as CPTs e priors são estimadas apenas com os 21 casos de treino. A Rede 2 usa dois intervalos automáticos por variável.", width="98%")


def fig_dist():
    return fig("dist", "fig01_distribuicoes.png", "Distribuição das quatro variáveis fisiológicas segundo o desfecho real na base completa (n = 30). As linhas tracejadas indicam os cortes clínicos usados na Rede 1; os pontos mostram os casos individuais.", width="96%")


def fig_conf_nb():
    return fig("conf_nb", "fig03_matrizes_confusao_nb.png", "Matrizes de confusão das duas redes Naive Bayes no treino e no teste, com limiar de 50%. A classe positiva é Sucesso; o falso positivo é a liberação prematura.", width="68%")


def fig_met_nb():
    return fig("met_nb", "fig04_metricas_nb.png", "Métricas de classificação (à esquerda) e calibração/confiança (à direita) das redes Naive Bayes. A Rede 2 é mais confiante, porém com pior especificidade no teste.", width="98%")


def fig_div():
    return fig("div", "fig05_divergencia_probabilidades.png", "Casos em que as duas redes mais divergem em P(Sucesso | evidências), ordenados pela diferença absoluta. A linha tracejada marca 50% e a pontilhada, o limiar conservador de 60%.", width="77%")


def fig_prob_teste():
    return fig("prob_teste", "fig05b_probabilidades_teste.png", "Probabilidade posterior de sucesso nos nove casos de teste (TE1–TE9), comparando a Rede 1 (limiares clínicos) e a Rede 2 (automática via Netica). A linha tracejada indica o ponto de decisão de 50%.", width="98%")


def fig_lim():
    return fig("lim", "fig06_limiares.png", "Cortes de discretização da Rede 1 (linhas tracejadas) e da Rede 2 (linha cheia) sobre os casos de treino e teste. Os cortes automáticos foram produzidos pelo Netica 7.01 com dois estados e 20% de arredondamento, usando somente o treino.", width="98%")


def fig_mf():
    return fig("mf", "fig07_funcoes_pertinencia.png", "Funções de pertinência das quatro entradas e da saída do sistema Mamdani. Trapézios nas extremidades e triângulos nas transições; as sobreposições representam explicitamente a incerteza clínica.", width="86%")


def fig_regras():
    return fig("regras", "fig08_regras_consequentes.png", "Distribuição dos consequentes das 12 regras compactas (Fracasso e Sucesso) e mapa de decisão no plano NIF × RR, fixando VT adequado e RSBI favorável.", width="86%")


def fig_surf1():
    return fig("surf1", "fig09_superficie_nif_rr.png", "Superfície de decisão do FIS em função de NIF e RR, com VT = 450 mL e RSBI = 90. As linhas escuras marcam os níveis 40 e 60 do resultado defuzzificado.", width="73%")


def fig_surf2():
    return fig("surf2", "fig10_superficie_vt_rsbi.png", "Superfície de decisão do FIS em função de VT e RSBI, com NIF = −30 cm.H₂O e RR = 25. Observa-se a interação entre volume e padrão respiratório.", width="73%")


def fig_fuzzy30():
    return fig("fuzzy30", "fig11_fuzzy_30casos.png", "Resultado defuzzificado para os 30 casos, com os limiares 40, 50 e 60. Em vermelho, fracassos reais; em azul, sucessos reais.", width="98%")


def fig_fuzzy_conf():
    return fig("fuzzy_conf", "fig12_fuzzy_confusao.png", "Matrizes de confusão do sistema Fuzzy na base completa com limiares 50 e 60. Os três falsos negativos do treino e o único falso positivo são os mesmos nos dois limiares.", width="64%")


def fig_case(cid):
    specs = {
        "TE5": ("fig13_caso1_entradas.png", "fig14_caso1_saida.png", "Caso 1 (TE5)"),
        "TR10": ("fig15_caso2_entradas.png", "fig16_caso2_saida.png", "Caso 2 (TR10)"),
        "TE3": ("fig17_caso3_entradas.png", "fig18_caso3_saida.png", "Caso 3 (TE3)"),
    }
    a, b, lab = specs[cid]
    f1 = fig(f"case_{cid}_in", a, f"{lab} — fuzzificação das quatro entradas. A linha preta marca o valor do paciente; as caixas mostram as pertinências não nulas.", width="96%")
    f2 = fig(f"case_{cid}_out", b, f"{lab} — agregação Mamdani, região recortada e centroide (linha vertical). As barras laterais mostram a força de disparo α de cada regra ativada.", width="96%")
    return f1 + f2


def fig_final():
    return fig("final", "fig19_comparacao_final.png", "Comparação final dos três modelos na base completa (n = 30), com a classe positiva Sucesso.", width="82%")



# ---------------------------------------------------------------------------
# Conteúdo
# ---------------------------------------------------------------------------
parts: list[str] = []

# ---------------------------------------------------------------- Capa
parts.append(f"""
<section class="cover">
  <div class="kicker">Instituto Federal de Santa Catarina · Campus Florianópolis · Departamento Acadêmico de Eletrotécnica</div>
  <div class="rule"></div>
  <h1 class="title">Apoio à decisão na interrupção da ventilação mecânica</h1>
  <div class="subtitle">Duas redes Naive Bayes e um sistema de inferência fuzzy Mamdani aplicados ao desmame ventilatório (com implementação e validação nos programas gráficos Netica 7.01 e QtFuzzyLite 4.0)</div>
  <div class="meta">
    <b>Autores:</b> Luiz Eduardo do Nascimento Ciarallo · Renato Deodoro<br>
    <b>Disciplina:</b> Sistemas Probabilísticos e Difusos<br>
    <b>Professor:</b> Cesar Alberto Penz<br>
    <b>Curso:</b> Especialização em Computação Aplicada à Indústria<br>
    <b>Base de dados:</b> Ashutosh et al. (1992), <i>Critical Care Medicine</i> — 21 casos de treino e 9 de teste<br>
    <b>Data:</b> novembro de 2026
  </div>
  <div class="abstract">
    <b>Resumo.</b> Este trabalho desenvolve, implementa nos programas gráficos oficiais da disciplina (<b>Netica Application 7.01</b> e <b>QtFuzzyLite 4.0</b>) e valida numericamente em Python dois modelos probabilísticos (redes Naive Bayes) e um sistema de inferência fuzzy Mamdani para apoiar a decisão de interrupção da ventilação mecânica. A base contém 21 tentativas de treinamento (9 sucessos e 12 fracassos) e 9 de teste (3 fracassos e 6 sucessos), conferidas linha a linha contra os arquivos fornecidos na disciplina (<span class="mono">Respira_DB_Treinamento.txt</span>, <span class="mono">Respira_DB_Teste.txt</span> e <span class="mono">CTRL_Respira.txt</span>), descritas por NIF, RR, VT e RSBI. A Rede 1 usa cortes clínicos (NIF −30/−20 cm.H₂O; RR 20/30 resp/min; VT 350/600 mL; RSBI 105/150 resp/min/L); a Rede 2 usa dois intervalos automáticos produzidos diretamente pelo Netica 7.01 (cortes −25, 30, 350 e 90). Ambas usam o nó pai <b>Resultado</b> (<i>Fracasso</i>/<i>Sucesso</i>) e foram processadas via <i>Cases → Process Cases</i> com <span class="mono">CTRL_Respira.txt</span> (concordância total com Python, erro máximo 5,5 × 10<sup>−7</sup>). O sistema fuzzy Mamdani emprega quatro entradas, saída <b>Resultado</b> (<b>Fracasso</b> e <b>Sucesso</b>) e base enxuta e completa de 12 regras clínicas, validada no motor <b>fuzzylite 4.0</b> e simulada na interface <b>QtFuzzyLite 4.0</b>. Na base completa (30 casos), a Rede 1 obteve 90,0% de acurácia (9/9 no teste; especificidade 93,3%; acurácia balanceada 90,0%), a Rede 2 obteve 83,3% (7/9 no teste; especificidade no teste 33,3%) e o Fuzzy obteve 86,7% (9/9 no teste; especificidade 93,3%). Os estudos de caso TE5 (sucesso claro), TR10 (fracasso claro) e TE3 (crítico/duvidoso) detalham fuzzificação, regras ativadas, agregação, centroide e capturas gráficas reais.
  </div>
  <div class="foot">Relatório técnico-científico · versão reprodutível com programas gráficos (Netica 7.01 e QtFuzzyLite 4.0), notebook, scripts, redes .dne e modelos .fcl/.fis/.fll.</div>
</section>
""")

# ---------------------------------------------------------------- Sumário
parts.append(f"""
<section class="toc">
<h1 class="first">Sumário</h1>
<ol>
  <li>1. Introdução e contextualização clínica</li>
  <li>2. Dados, análise exploratória e protocolo de avaliação</li>
  <li>3. Modelagem probabilística: duas redes Naive Bayes</li>
  <li class="lvl2">3.1 Topologia, hipótese de independência e redundância entre RR, VT e RSBI</li>
  <li class="lvl2">3.2 Discretização por conhecimento clínico (Rede 1)</li>
  <li class="lvl2">3.3 Discretização automática executada pelo Netica (Rede 2)</li>
  <li class="lvl2">3.4 Probabilidades condicionais e suavização de Laplace</li>
  <li class="lvl2">3.5 Resultados no treino e no teste</li>
  <li class="lvl2">3.6 Matrizes de confusão, métricas e calibração</li>
  <li class="lvl2">3.7 Limiar de decisão e assimetria de riscos</li>
  <li class="lvl2">3.8 Comparação crítica e análise dos casos de fronteira</li>
  <li class="lvl2">3.9 Robustez e redundância: validação leave-one-out</li>
  <li class="lvl2">3.10 Execução no programa gráfico Netica 7.01 e Process Cases</li>
  <li>4. Modelagem difusa: sistema de inferência Mamdani</li>
  <li class="lvl2">4.1 Arquitetura, operadores e universo de discurso</li>
  <li class="lvl2">4.2 Funções de pertinência e justificativa clínica</li>
  <li class="lvl2">4.3 Base de regras (12 regras compactas), completude e consistência</li>
  <li class="lvl2">4.4 Avaliação na base completa e limiar de decisão</li>
  <li class="lvl2">4.5 Estudo de caso 1: TE5 — sucesso claro</li>
  <li class="lvl2">4.6 Estudo de caso 2: TR10 — fracasso claro</li>
  <li class="lvl2">4.7 Estudo de caso 3: TE3 — crítico/duvidoso</li>
  <li class="lvl2">4.8 Execução no programa gráfico QtFuzzyLite 4.0 e síntese da modelagem difusa</li>
  <li>5. Comparação geral, limitações e conclusão</li>
  <li class="lvl2">5.1 Desempenho preditivo</li>
  <li class="lvl2">5.2 Interpretabilidade, incerteza e adequação clínica</li>
  <li class="lvl2">5.3 Limitações</li>
  <li class="lvl2">5.4 Conclusão</li>
  <li>Referências</li>
  <li>Apêndice A — Base de regras completa (12 regras compactas)</li>
  <li>Apêndice B — Probabilidades condicionais das redes</li>
  <li>Apêndice C — Reprodutibilidade, validações e evidências gráficas (Netica 7.01 e QtFuzzyLite 4.0)</li>
</ol>
<div class="box neutral" style="margin-top:14pt">
<h4>Como ler este relatório</h4>
<p><b>Sucesso</b> é a classe positiva. Portanto, <b>sensibilidade</b> é a proporção de pacientes que podiam ser liberados e foram corretamente reconhecidos; <b>especificidade</b> é a proporção de pacientes que fracassariam e foram corretamente retidos; e <b>acurácia balanceada</b> é a média aritmética entre sensibilidade e especificidade. O <b>falso positivo</b> (prever sucesso quando houve fracasso) é o erro mais grave, pois leva à extubação prematura. O <b>falso negativo</b> (prever fracasso quando houve sucesso) prolonga o suporte, com custo e risco infeccioso, mas é potencialmente reversível.</p>
</div>
</section>
""")

# ---------------------------------------------------------------- 1 Introdução
parts.append(f"""
<h1 id="sec-intro">1. Introdução e contextualização clínica</h1>
<p class="section-lead">A decisão de interromper a ventilação mecânica é uma das mais frequentes e mais incertas na unidade de terapia intensiva. Ela exige integrar força muscular, drive respiratório, padrão ventilatório e reserva funcional — sinais que se sobrepõem e não obedecem a limiares rígidos.</p>

<h2>1.1 O problema clínico</h2>
<p>A ventilação mecânica invasiva sustenta a troca gasosa e reduz o trabalho respiratório, mas prolonga a exposição a pneumonia associada à ventilação, lesão diafragmática, sedação e eventos adversos. O desmame é o processo de retirada gradual ou abrupta desse suporte; a interrupção pode ocorrer por extubação após um teste de respiração espontânea (TRE), por redução progressiva de pressão de suporte ou por traqueostomia em casos selecionados. A pergunta prática é: <b>este paciente tem condições de respirar sem auxílio, de forma estável, sem sofrimento e sem necessidade de reintubação?</b></p>
<p>Duas falhas são possíveis e têm custos diferentes. A <b>interrupção prematura</b> (falso positivo) pode causar desconforto, hipoxemia, hipercapnia, aspiração, parada cardiorrespiratória e reintubação de urgência; a reintubação associa-se a maior mortalidade, maior tempo de UTI e pior desfecho funcional. A <b>interrupção tardia</b> (falso negativo) desperdiça um recurso escasso, prolonga a sedação e a imobilidade, aumenta o risco de infecção nosocomial e de fraqueza adquirida na UTI. Em geral, o primeiro erro é mais grave e menos reversível; essa assimetria deve influenciar o limiar de decisão.</p>

<h2>1.2 Variáveis fisiológicas e critérios de referência</h2>
<p>As variáveis disponíveis são de obtenção relativamente simples à beira-leito e cobrem três dimensões: força muscular, drive/pattern respiratório e eficiência da ventilação.</p>
<ul>
<li><b>Pico negativo da pressão inspiratória (NIF/PImax, cm.H₂O):</b> medida de força dos músculos inspiratórios. Valores mais negativos indicam maior força; o limiar clássico é ≤ −30 cm.H₂O. A faixa entre −30 e −20 é explicitamente uma zona de incerteza; valores acima de −20 (menos negativos) associam-se a provável falha.</li>
<li><b>Taxa respiratória (RR, resp/min):</b> expressa o drive e o custo metabólico da respiração. RR persistentemente acima de 30–35 resp/min sugere desconforto e maior probabilidade de falha; RR &lt; 30 é referência de sucesso.</li>
<li><b>Volume tidal (VT, mL):</b> volume corrente espontâneo. Volumes muito baixos (&lt; 5 mL/kg, aproximadamente &lt; 350 mL em 70 kg) sugerem fraqueza e padrão superficial; valores &gt; 315 mL são referência de sucesso, e volumes elevados (&gt; 600 mL) exigem avaliação de esforço e risco de lesão.</li>
<li><b>RSBI = RR / VT(L) (resp/min/L):</b> índice combinado de respiração rápida e superficial. O corte clássico é 105: valores &lt; 105 favorecem sucesso; valores ≥ 105 aumentam a probabilidade de falha, com risco crescente acima de 150.</li>
</ul>

<h2>1.3 Objetivos</h2>
<ol>
<li>Construir duas redes Naive Bayes para estimar P(Sucesso | NIF, RR, VT, RSBI), uma com discretização baseada em conhecimento clínico e outra com limiares automáticos obtidos pelo Netica, estimando todas as probabilidades exclusivamente com os 21 casos de treino.</li>
<li>Avaliar as redes caso a caso no treino e no teste, construir matrizes de confusão e comparar acurácia, sensibilidade, especificidade e grau de certeza.</li>
<li>Projetar um sistema de inferência fuzzy Mamdani com quatro entradas, funções de pertinência justificadas pela literatura, base de regras completa e defuzzificação por centroide.</li>
<li>Avaliar o FIS nos 30 casos e demonstrar detalhadamente fuzzificação, regras ativadas e defuzzificação em três casos representativos: sucesso claro, fracasso claro e caso crítico/duvidoso.</li>
<li>Discutir comparativamente as abordagens, suas limitações e seu papel em um fluxo de apoio à decisão — nunca como substituto do julgamento clínico.</li>
</ol>

<h2>1.4 Contribuições e organização</h2>
<p>Além da comparação solicitada, o trabalho contribui com três elementos de rigor: (i) auditoria explícita de <i>data leakage</i>, com todos os cortes e CPTs restritos ao treino; (ii) análise da violação da independência condicional causada por RR, VT e RSBI simultâneos, com validação leave-one-out de subconjuntos; (iii) validação cruzada da implementação manual de Naive Bayes contra <span class="mono">sklearn.CategoricalNB</span> e do motor Mamdani contra <span class="mono">scikit-fuzzy</span>. O restante do relatório está organizado em dados e métodos (Seção 2), redes Naive Bayes (Seção 3), sistema fuzzy e estudos de caso (Seção 4), comparação e conclusão (Seção 5), referências e apêndices.</p>
""")

# ---------------------------------------------------------------- 2 Dados
# T1 composição
rows = [
    ["Treinamento", "21", "12", "9", "57,1% / 42,9%"],
    ["Teste", "9", "3", "6", "33,3% / 66,7%"],
    ["Total", "30", "15", "15", "50,0% / 50,0%"],
]
t1 = simple_table("composicao", "Composição da base de dados por conjunto e desfecho.", ["Conjunto", "n", "Fracassos", "Sucessos", "Fracasso / Sucesso"], rows, source="Fonte: base do trabalho (Ashutosh et al., 1992).", cls="keep", numeric_from=1)

desc_rows = []
for var in w.VARIABLES:
    for cls in w.CLASSES:
        r = desc[(desc["Variável"] == var) & (desc["Desfecho"] == cls)].iloc[0]
        label = "Fracasso" if cls == "Fracasso" else "Sucesso"
        desc_rows.append([var, label, str(int(r["n"])), f'{n(r["Média"],1)} ± {n(r["DP"],1)}', f'{n(r["Mínimo"],0)}–{n(r["Máximo"],0)}'])
t2 = simple_table("descritiva", "Estatística descritiva das variáveis na base completa, por desfecho (média ± desvio-padrão e faixa observada).", ["Variável", "Desfecho", "n", "Média ± DP", "Mín–Máx"], desc_rows, source="Fonte: elaboração própria. DP = desvio-padrão amostral.", cls="keep", numeric_from=2)

corr = full[["NIF", "RR", "VT", "RSBI"]].corr(method="spearman")
corr_rows = []
for i, var in enumerate(["NIF", "RR", "VT", "RSBI"]):
    corr_rows.append([var] + [n(corr.iloc[i, j], 2) for j in range(4)])
t3 = simple_table("correlacao", "Correlação de Spearman entre as variáveis de entrada na base completa (n = 30).", ["", "NIF", "RR", "VT", "RSBI"], corr_rows, source="Fonte: elaboração própria. Correlações altas entre RSBI, RR e VT refletem a definição determinística do índice.", cls="keep", numeric_from=1)

parts.append(f"""
<h1 id="sec-dados">2. Dados, análise exploratória e protocolo de avaliação</h1>

<h2>2.1 Base de dados e composição</h2>
<p>A base foi organizada a partir do material da disciplina, que reproduz o conjunto de 30 tentativas de interrupção descrito por Ashutosh et al. (1992). Cada linha corresponde a uma tentativa, com quatro preditores medidos antes da interrupção e o desfecho binário (sucesso ou fracasso). A divisão é a fornecida no enunciado: 21 casos para treinamento e 9 para teste (conjunto prospectivo do estudo de referência). Os quatro arquivos de casos fornecidos na disciplina (<span class="mono">Respira_DB_Treinamento.txt</span>, <span class="mono">Respira_DB_Teste.txt</span>, <span class="mono">Respira_DB_Treinamento_Process_Cases.txt</span> e <span class="mono">Respira_DB_Teste_Process_Cases.txt</span>) foram conferidos linha a linha contra <span class="mono">data/weaning_train.csv</span> e <span class="mono">data/weaning_test.csv</span> (função <span class="mono">verify_discipline_files()</span>) e coincidem integralmente, assim como o controlador <span class="mono">CTRL_Respira.txt</span> preservado em <span class="mono">references/originais/</span>. A separação treino/teste foi mantida de forma estrita.</p>
{t1}
<p>O treino é levemente desbalanceado (12 fracassos para 9 sucessos), enquanto o teste tem 3 fracassos e 6 sucessos. Essa diferença é relevante: uma rede que aprenda a privilegiar a classe majoritária do treino (Fracasso) pode parecer conservadora no treino e ter dificuldade no teste; o contrário também é possível. Por isso, além da acurácia, o relatório reporta sensibilidade, especificidade, PPV, NPV e F1.</p>

<h2>2.2 Análise exploratória</h2>
<p>{t2}</p>
<p>Os grupos se separam parcialmente: pacientes com sucesso têm NIF médio de −38,8 cm.H₂O, contra −21,0 nos fracassos; RR médio de 22,5 contra 33,3; VT médio de 552,7 mL contra 326,1; e RSBI médio de 50,0 contra 117,9. Há, no entanto, sobreposição importante em todas as variáveis. O caso TR1 é o exemplo mais extremo: NIF −30, RR 17, VT 500 e RSBI 34 — todos os critérios aparentemente favoráveis —, mas o desfecho real foi fracasso. Esse caso antecipa o limite de qualquer modelo baseado apenas nessas quatro variáveis.</p>
{fig_dist()}
<p>{t3}</p>
<p>A correlação de Spearman entre RSBI e RR é 0,91, e entre RSBI e VT é −0,94. Isso não é coincidência: o RSBI é definido como RR dividido pelo VT em litros. A máxima diferença absoluta entre o RSBI informado e o valor recalculado foi 0,5 unidade, compatível com arredondamento. A dependência é, portanto, determinística e precisa ser tratada na modelagem probabilística.</p>

<h2>2.3 Protocolo antileakage e de avaliação</h2>
<p>O protocolo foi desenhado para evitar que informação do teste contaminasse a construção do modelo:</p>
<ul>
<li><b>Treino:</b> apenas os 21 casos foram usados para definir os limiares automáticos da Rede 2, estimar priors, estimar CPTs e realizar a validação leave-one-out.</li>
<li><b>Teste:</b> os 9 casos foram mantidos fora de qualquer decisão de projeto e usados somente na avaliação final. O mesmo vale para as 30 observações na análise do fuzzy: as funções de pertinência e as regras foram definidas por conhecimento clínico, não por ajuste aos dados.</li>
<li><b>Rede 2:</b> os cortes foram produzidos pelo Netica 7.01 a partir do arquivo com as quatro entradas dos 21 casos de treino. Rede nativa, capturas e hashes permitem verificar a execução e sua entrada.</li>
<li><b>Validação interna:</b> leave-one-out no treino, refazendo priors e CPTs em cada dobra, para avaliar estabilidade e redundância. Os cortes da Rede 2 ficam fixos; portanto, essa análise é condicional à discretização e não é uma validação aninhada de todo o processo. O teste não é usado nessa etapa.</li>
</ul>
<p>As métricas adotadas são:</p>
<div class="formula">
Acurácia = (TP + TN)/N &nbsp;&nbsp;·&nbsp;&nbsp; Sensibilidade = TP/(TP + FN) &nbsp;&nbsp;·&nbsp;&nbsp; Especificidade = TN/(TN + FP)
</div>
<div class="formula">
PPV = TP/(TP + FP) &nbsp;&nbsp;·&nbsp;&nbsp; NPV = TN/(TN + FN) &nbsp;&nbsp;·&nbsp;&nbsp; F1 = 2TP/(2TP + FP + FN)
</div>
<p>onde TP = verdadeiro positivo (sucesso previsto e real), TN = verdadeiro negativo (fracasso previsto e real), FP = falso positivo (sucesso previsto, fracasso real) e FN = falso negativo (fracasso previsto, sucesso real). O <b>escore de Brier</b> é a média de (p<sub>i</sub> − y<sub>i</sub>)<sup>2</sup> e mede calibração e acurácia conjuntamente; valores menores são melhores. O <b>grau de certeza médio</b> é a média de max(P(Sucesso), P(Fracasso)): valores altos indicam confiança, não necessariamente acerto. O <b>log-loss</b> também foi calculado e está nos artefatos.</p>
<div class="box warn">
<h4>Assimetria de custos</h4>
<p>Um falso positivo pode levar à reintubação e a eventos potencialmente fatais; um falso negativo geralmente pode ser corrigido com reavaliação e prolongamento do suporte. Por isso, analisamos também limiares conservadores (exigir probabilidade ou potencial mais altos para liberar). A escolha do limiar não é apenas estatística: é uma decisão clínica e institucional.</p>
</div>
""")

# ---------------------------------------------------------------- 3 Naive Bayes
def nb_case_table(key, caption, df):
    head = """
<tr>
  <th rowspan="2">ID</th><th rowspan="2">NIF</th><th rowspan="2">RR</th><th rowspan="2">VT</th><th rowspan="2">RSBI</th><th rowspan="2">Real</th>
  <th colspan="3">Rede 1 — clínica</th><th colspan="3">Rede 2 — Netica</th><th colspan="2">Acerto</th>
</tr>
<tr><th class="num">P(S)</th><th class="num">P(F)</th><th>Pred.</th><th class="num">P(S)</th><th class="num">P(F)</th><th>Pred.</th><th>R1</th><th>R2</th></tr>
"""
    body = []
    for _, r in df.iterrows():
        real = "S" if r["Resultado"] == "Sucesso" else "F"
        p1 = "F" if r["Rede 1: predito"] == "Fracasso" else "S"
        p2 = "F" if r["Rede 2: predito"] == "Fracasso" else "S"
        ok1 = "✓" if r["Rede 1: acerto"] == "Acerto" else "✗"
        ok2 = "✓" if r["Rede 2: acerto"] == "Acerto" else "✗"
        cls = "err" if (ok1 == "✗" or ok2 == "✗") else ""
        body.append(f"""<tr class="{cls}">
<td>{e(r['ID'])}</td><td class="num">{n(r['NIF'],0)}</td><td class="num">{n(r['RR'],0)}</td><td class="num">{n(r['VT'],0)}</td><td class="num">{n(r['RSBI'],0)}</td><td class="center">{real}</td>
<td class="num">{n(100*r['Rede 1: P(Sucesso)'],1)}</td><td class="num">{n(100*r['Rede 1: P(Fracasso)'],1)}</td><td class="center">{p1}</td>
<td class="num">{n(100*r['Rede 2: P(Sucesso)'],1)}</td><td class="num">{n(100*r['Rede 2: P(Fracasso)'],1)}</td><td class="center">{p2}</td>
<td class="center">{ok1}</td><td class="center">{ok2}</td></tr>""")
    note = "F = fracasso; S = sucesso; P(S) e P(F) em %. Linhas destacadas contêm pelo menos um erro de classificação. Limiar de 50%."
    source = "Fonte: elaboração própria. Probabilidades calculadas com priors e CPTs estimadas apenas no treino (Laplace α = 1)."
    return tab(key, caption, head, "".join(body), source=source, cls="tiny nb", note=note)


# Tabela 4 — discretização clínica
c_rows = []
for _, r in clinical.iterrows():
    c_rows.append([r["Variável"], r["Estado"], e(r["Faixa"]), e(r["Justificativa clínica"])])
t4 = tab(
    "disc_clinica",
    "Discretização da Rede 1 por conhecimento clínico: estados, limiares e justificativa fisiológica.",
    "<tr><th>Variável</th><th>Estado</th><th>Limiares numéricos</th><th>Justificativa clínica</th></tr>",
    "".join(f"<tr>{''.join(f'<td>{x}</td>' for x in row)}</tr>" for row in c_rows),
    source="Fonte: base de conhecimento da disciplina, III Consenso Brasileiro de Ventilação Mecânica, Yang & Tobin (1991), Silva et al. (2015), Hess & Kacmarek e material da Dräger.",
    cls="tiny keep",
    colgroup='<colgroup><col style="width:8%"><col style="width:12%"><col style="width:17%"><col style="width:63%"></colgroup>',
    note="NIF mais negativo = mais forte. Os estados limítrofes são deliberadamente faixas de incerteza, não apenas intervalos estatísticos.",
)

# Tabela 5 — auditoria Netica
a_rows = []
for _, r in audit.iterrows():
    a_rows.append([r["Variável"], n(r["Corte adotado"], 0), n(r["Mediana treino"], 0), str(int(r["n faixa inferior"])), str(int(r["n faixa superior"])), str(int(r["Diferença"])), e(r["Intervalos"])])
t5 = simple_table("audit_netica", "Descrição no treino dos cortes automáticos efetivamente produzidos pelo Netica 7.01.", ["Variável", "Corte", "Mediana treino", "n inferior", "n superior", "|dif.|", "Intervalos"], a_rows, source="Fonte: Netica Application 7.01, Auto-Discretize Nodes, dois estados e 20% de arredondamento; somente 21 casos de treino.", cls="tiny keep", numeric_from=1, note="A mediana é descritiva: os cortes foram lidos do arquivo nativo salvo pelo Netica, não estimados em Python.")

# Tabela 6 — comparação de limiares
cmp_rows = [[r["Variável"], e(r["Rede 1 (clínica)"]), e(r["Rede 2 (Netica)"]), e(r["Efeito"])] for _, r in compare.iterrows()]
t6 = tab(
    "cmp_limiares",
    "Comparação entre os pontos de corte da discretização clínica (Rede 1) e da automática (Rede 2).",
    "<tr><th>Variável</th><th>Rede 1 — clínica</th><th>Rede 2 — Netica</th><th>Efeito da diferença</th></tr>",
    "".join(f"<tr>{''.join(f'<td>{x}</td>' for x in row)}</tr>" for row in cmp_rows),
    source="Fonte: elaboração própria com base na captura de tela do Netica e na auditoria do treino.",
    cls="tiny keep",
    colgroup='<col style="width:9%"><col style="width:20%"><col style="width:20%"><col style="width:51%"></colgroup>',
)

# Tabela 7 — priors
priors = pd.DataFrame(
    [
        ["Rede 1 — clínica", "12", "9", n(12.0/21, 3), n(9.0/21, 3), n(13.0/23, 3), n(10.0/23, 3)],
        ["Rede 2 — Netica", "12", "9", n(12.0/21, 3), n(9.0/21, 3), n(13.0/23, 3), n(10.0/23, 3)],
    ]
)
t7 = simple_table("priors", "Contagens e priors das classes no treino, antes e depois da suavização de Laplace (α = 1).", ["Rede", "n Fracasso", "n Sucesso", "P(F) empírica", "P(S) empírica", "P(F) Laplace", "P(S) Laplace"], priors.values.tolist(), source="Fonte: elaboração própria. A soma das priors suavizadas é 1.", cls="keep", numeric_from=1)

p1_train = "novo"  # marcadores para clareza do código; as tabelas vêm dos CSVs
t8 = nb_case_table("nb_treino", "Resultados da Rede 1 e da Rede 2 nos 21 casos de treinamento, com probabilidades posteriores e classe predita (limiar 50%).", nb_train)
t9 = nb_case_table("nb_teste", "Resultados da Rede 1 e da Rede 2 nos 9 casos de teste, com probabilidades posteriores e classe predita (limiar 50%).", nb_test)

# Tabela 10 — métricas
met_rows = []
for _, r in nb_metrics.iterrows():
    met_rows.append([
        e(r["Modelo"]), e(r["Conjunto"]),
        pct(r["Acurácia"]), pct(r["Acurácia bal."]), pct(r["Sensibilidade"]), pct(r["Especificidade"]), pct(r["PPV"]), pct(r["NPV"]), pct(r["F1"]),
        n(r["Brier"], 3), pct(r["Certeza média"]),
        str(int(r["TN"])), str(int(r["FP"])), str(int(r["FN"])), str(int(r["TP"])),
    ])
t10 = simple_table(
    "met_nb",
    "Métricas de classificação (incluindo acurácia balanceada) e calibração das duas redes Naive Bayes no treino, no teste e na base completa.",
    ["Rede", "Conjunto", "Acurácia", "Ac. bal.", "Sens.", "Espec.", "PPV", "NPV", "F1", "Brier", "Certeza", "TN", "FP", "FN", "TP"],
    met_rows,
    source="Fonte: elaboração própria. Ac. bal. = (Sensibilidade + Especificidade)/2; Sens. = sensibilidade; Espec. = especificidade; Certeza = média de max(P(S),P(F)).",
    cls="tiny",
    note="A classe positiva é Sucesso. O Brier combina calibração e acurácia; a certeza média mede confiança, não correção.",
)

# Tabela 11 — limiares NB
th_full = th_nb[th_nb["Conjunto"] == "Completa"].copy()
th_rows = []
for _, r in th_full.iterrows():
    th_rows.append([e(r["Rede"]), n(r["Limiar"], 2), pct(r["acuracia"]), pct(r["sensibilidade"]), pct(r["especificidade"]), str(int(r["fp"])), str(int(r["fn"]))])
t11 = simple_table("th_nb", "Sensibilidade ao limiar de decisão das redes Naive Bayes na base completa (n = 30).", ["Rede", "Limiar", "Acurácia", "Sensibilidade", "Especificidade", "FP", "FN"], th_rows, source="Fonte: elaboração própria. O limiar padrão é 0,50.", cls="tiny keep")

# Tabela 12 — LOO
loo_rows = [[e(r["Rede"]), e(r["Variáveis"]), pct(r["Acurácia LOO"]), pct(r["Sensibilidade LOO"]), pct(r["Especificidade LOO"])] for _, r in loo.iterrows()]
t12 = simple_table("loo", "Validação leave-one-out no treino: efeito da inclusão simultânea de RR, VT e RSBI e de subconjuntos de variáveis.", ["Rede", "Variáveis", "Acurácia", "Sensibilidade", "Especificidade"], loo_rows, source="Fonte: elaboração própria. A discretização é fixa; priors e CPTs são refeitas em cada dobra. Nenhum caso de teste participa.", cls="tiny", note="A análise é exploratória: com 21 casos, uma diferença de um acerto altera a acurácia em 4,8 pontos percentuais.")

# Tabela 13 — casos críticos
crit_ids = ["TR1", "TR4", "TR13", "TR14", "TR16", "TE1", "TE3"]
nb_all = pd.concat([nb_train, nb_test], ignore_index=True)
crit_comments = {
    "TR1": "Todos os critérios favoráveis; fracasso real. Limite das quatro variáveis.",
    "TR4": "NIF, RR e VT no limite; redes retêm, mas com confiança distinta.",
    "TR13": "VT baixo e RR limítrofe; sucesso real. A Rede 1 erra.",
    "TR14": "VT baixo, NIF limítrofe e RSBI 97, próximo do corte 105; sucesso real. Ambas erram.",
    "TR16": "VT 250 e RSBI 120; sucesso real. A Rede 2 erra com alta confiança.",
    "TE1": "NIF fraco e RSBI favorável; fracasso real. A Rede 2 erra.",
    "TE3": "Caso crítico: NIF/RR limítrofes, VT baixo, RSBI favorável; fracasso real.",
}
c13_rows = []
for cid in crit_ids:
    r = nb_all[nb_all["ID"] == cid].iloc[0]
    fz = fuzzy[fuzzy["ID"] == cid].iloc[0]
    c13_rows.append([cid, "S" if r["Resultado"] == "Sucesso" else "F", n(r["NIF"],0), n(r["RR"],0), n(r["VT"],0), n(r["RSBI"],0), pct(r["Rede 1: P(Sucesso)"]), pct(r["Rede 2: P(Sucesso)"]), n(fz["crisp"],1), crit_comments[cid]])
t13 = simple_table("casos_criticos", "Casos de fronteira e divergências entre as redes, com a saída do sistema fuzzy.", ["ID", "Real", "NIF", "RR", "VT", "RSBI", "R1 P(S)", "R2 P(S)", "Fuzzy", "Comentário"], c13_rows, source="Fonte: elaboração própria. Fuzzy = saída Resultado (0–100); decisão conservadora se ≥ 60.", cls="tiny", numeric_from=1, note="F = fracasso; S = sucesso. Estes casos são discutidos no texto e no estudo de caso TE3.")

netica_r1_fig = fig(
    "netica_r1", str(ROOT / "netica/evidencias/05_netica_rede1_clinica.png"),
    "Rede 1 (clínica) aberta e compilada no Netica Application 7.01, com o nó pai Resultado (Fracasso 56,5% / Sucesso 43,5%) e os quatro nós filhos contínuos discretizados pelos limiares clínicos.",
    source="Fonte: captura real da execução no Netica Application 7.01 (Limited Mode).", width="76%"
)
netica_r1_cpt_fig = fig(
    "netica_r1_cpt", str(ROOT / "netica/evidencias/06b_netica_rede1_cpt.png"),
    "Tabela de probabilidade condicional (CPT) do nó NIF na Rede 1 visualizada no Netica Application 7.01 (Table → View/Edit).",
    source="Fonte: captura real da execução no Netica Application 7.01.", width="68%"
)
netica_r2_fig = fig(
    "netica_r2", str(ROOT / "netica/evidencias/07_netica_rede2_automatica.png"),
    "Rede 2 (automática) aberta e compilada no Netica Application 7.01 com Automatic Updating ativado: cortes −25, 30, 350 e 90 e marginais idênticas à captura de referência da disciplina.",
    source="Fonte: captura real da execução no Netica Application 7.01 (Limited Mode).", width="76%"
)
netica_te3_r1_fig = fig(
    "netica_te3_r1", str(ROOT / "netica/evidencias/06_netica_rede1_te3.png"),
    "Inferência gráfica do caso crítico TE3 (NIF −28, RR 24, VT 310, RSBI 77; desfecho real Fracasso) na Rede 1 no Netica 7.01: Resultado = 55,8% Fracasso / 44,2% Sucesso (retém corretamente).",
    source="Fonte: captura real no Netica Application 7.01 via Cases → Get Case (Respira_DB_Teste_Process_Cases.txt, IDnum = 3).", width="74%"
)
netica_te3_r2_fig = fig(
    "netica_te3_r2", str(ROOT / "netica/evidencias/08_netica_rede2_te3.png"),
    "Inferência gráfica do mesmo caso TE3 na Rede 2 no Netica 7.01: Resultado = 9,36% Fracasso / 90,6% Sucesso (falso positivo de alta confiança causado pela discretização binária grosseira).",
    source="Fonte: captura real no Netica Application 7.01 via Cases → Get Case (Respira_DB_Teste_Process_Cases.txt, IDnum = 3).", width="74%"
)

parts.append(f"""
<h1 id="sec-nb">3. Modelagem probabilística: duas redes Naive Bayes</h1>

<h2>3.1 Topologia, hipótese de independência e redundância entre RR, VT e RSBI</h2>
<p>A topologia adotada é a de um classificador Naive Bayes: o nó pai <b>Resultado</b>, com estados <i>Fracasso</i> e <i>Sucesso</i> (exatamente os identificadores consultados pelo controlador <span class="mono">CTRL_Respira.txt</span>), condiciona quatro nós filhos — <b>NIF</b>, <b>RR</b>, <b>VT</b> e <b>RSBI</b>. Cada filho tem uma distribuição condicional ao estado do pai, e a rede impõe que os filhos sejam condicionalmente independentes dado o desfecho. A decisão é a classe com maior probabilidade posterior:</p>
<div class="formula">P(C | x<sub>1</sub>,…,x<sub>4</sub>) ∝ P(C) · Π<sub>j=1</sub><sup>4</sup> P(x<sub>j</sub> | C).</div>
{fig_topologia()}
<p>A hipótese de independência é uma aproximação. Em particular, <b>RR, VT e RSBI não são independentes</b>: o RSBI é exatamente RR dividido por VT(L). Incluir os três nós faz com que o mesmo padrão respiratório seja contado mais de uma vez, podendo produzir posteriores excessivamente extremas e uma falsa sensação de certeza. A correlação de Spearman de 0,91 entre RR e RSBI e de −0,94 entre VT e RSBI (Tabela [[tab:correlacao]]) confirma a redundância.</p>
<p><b>Decisão metodológica.</b> Mantivemos as quatro variáveis nas duas redes principais, tanto para cumprir o escopo do trabalho quanto para permitir comparação direta com a literatura e com a base de conhecimento, que lista os quatro preditores. A violação é explicitamente reconhecida, e a Seção 3.9 quantifica seu efeito por validação leave-one-out. Não se afirma, portanto, que a independência condicional seja verdadeira; afirma-se que o modelo é uma aproximação útil e interpretável, cujas limitações são medidas.</p>
<p>A estimação usa <b>suavização de Laplace</b> com α = 1 tanto para as priors quanto para as verossimilhanças:</p>
<div class="formula">P(C = c) = (N<sub>c</sub> + 1) / (N + 2) &nbsp;&nbsp;·&nbsp;&nbsp; P(x<sub>j</sub> = s | C = c) = (N<sub>cjs</sub> + 1) / (N<sub>c</sub> + K<sub>j</sub>).</div>
<p>O uso de α = 1 é coerente com a execução de referência do Netica: a captura de tela fornecida mostra P(Fracasso) = 56,5% e P(Sucesso) = 43,5%, exatamente (12+1)/(21+2) e (9+1)/(21+2). A mesma lógica aparece nas marginais das variáveis (por exemplo, 47,8%/52,2% para NIF, compatível com 10/11 casos suavizados). A suavização evita probabilidades nulas em células com poucos casos — um risco real com 21 observações.</p>

<h2>3.2 Discretização por conhecimento clínico (Rede 1)</h2>
<p>A Rede 1 usa três estados por variável, com cortes escolhidos para preservar a informação clínica e separar explicitamente as zonas de incerteza. A Tabela [[tab:disc_clinica]] apresenta os limiares e a justificativa; a Figura [[fig:lim]] os posiciona sobre a distribuição dos dados.</p>
{t4}
<p>O NIF é a variável com sinal negativo: quanto mais negativo, maior a força. Por isso, “Forte” é ≤ −30, “Limítrofe” é −30 a −20 e “Fraco” é &gt; −20. A faixa −30 a −20 é uma zona de incerteza deliberada: a literatura usa −30 como critério de força e > −30 (sobretudo > −20) como preditor de falha, mas não há um único ponto universal. O mesmo raciocínio se aplica ao RSBI: 105 é o corte clássico, mas valores entre 105 e 150 não são equivalentes a valores muito altos; por isso a faixa limítrofe. No arquivo <span class="mono">Weaning_Rede1_Clinica.dne</span>, os nós filhos são declarados como contínuos discretizados por níveis (<span class="mono">levels</span>) em ASCII puro, permitindo ao Netica receber diretamente os valores numéricos de <span class="mono">Respira_DB_Treinamento_Process_Cases.txt</span> e <span class="mono">Respira_DB_Teste_Process_Cases.txt</span>.</p>
{fig_lim()}

<h2>3.3 Discretização automática executada pelo Netica (Rede 2)</h2>
<p>A Rede 2 foi discretizada automaticamente no <b>Netica Application 7.01</b>, pelo comando <i>Modify → Auto-Discretize Nodes</i>, com <b>dois estados e 20% de arredondamento</b>. A única entrada foi <span class="mono">netica/treino_discretizacao.cas</span>: quatro variáveis, 21 casos de treino e nenhum desfecho. A rede com os níveis gerados foi salva pelo próprio Netica em <span class="mono">netica/evidencias/02_discretizacao_automatica.dne</span>. O Python lê esses níveis, sem cortes alternativos. A Tabela [[tab:audit_netica]] descreve a distribuição resultante no treino; a Tabela [[tab:cmp_limiares]] compara os cortes com os da Rede 1.</p>
{t5}
{t6}
<p>Três diferenças são particularmente importantes. Primeiro, o corte de NIF em −25 cai dentro da zona de incerteza clínica e elimina a distinção entre “forte” e “limítrofe”. Segundo, o VT é dividido apenas em “baixo” e “não baixo”, de modo que um VT de 350 mL (limite) passa a ser tratado como favorável, enquanto um VT de 900 mL não recebe nenhum crédito adicional. Terceiro, o RSBI é cortado em 90, abaixo do clássico 105; isso reclassifica RSBI 90–104 como desfavorável, mas também torna a rede sensível ao arredondamento da mediana amostral (86).</p>
<div class="box neutral">
<h4>Sobre a reprodutibilidade do Netica</h4>
<p>A Rede 2 foi aprendida e salva pelo <b>Netica C API 5.04 Linux</b> com contagem dos 21 casos e um pseudocaso por estado (Laplace α = 1), e ambas as redes (<span class="mono">Weaning_Rede1_Clinica.dne</span> e <span class="mono">Weaning_Rede2_Netica.dne</span>) foram compiladas e processadas diretamente na interface gráfica <b>Netica Application 7.01</b> via <i>Cases → Process Cases</i> usando o controlador oficial <span class="mono">CTRL_Respira.txt</span>. Todas as 60 probabilidades (21 treino + 9 teste em cada rede) concordam com o Python dentro de 5,5 × 10<sup>−7</sup> (arredondamento textual de 6 dígitos significativos do Netica), com 100% das classes iguais. Capturas reais, arquivos de saída do <i>Process Cases</i>, log e hashes SHA-256 estão em <span class="mono">netica/evidencias/</span>.</p>
</div>

<h2>3.4 Probabilidades condicionais e suavização de Laplace</h2>
<p>As priors empíricas do treino são 12/21 = 57,1% para fracasso e 9/21 = 42,9% para sucesso. Após a suavização, tornam-se 13/23 = 56,5% e 10/23 = 43,5%, respectivamente. A Tabela [[tab:priors]] registra o cálculo. As CPTs completas estão no Apêndice B; a Tabela [[tab:met_nb]] e as tabelas de resultados caso a caso permitem auditar o efeito de cada estado.</p>
{t7}
<p>Com 21 casos e até 12 células por variável (3 estados × 2 classes × 2 redes), várias células têm contagens pequenas. A suavização impede que uma única célula vazia zere a verossimilhança de uma classe inteira. O custo é um leve encolhimento das probabilidades em direção ao centro, sobretudo nas caudas.</p>

<h2>3.5 Resultados no treino e no teste</h2>
<p>As Tabelas [[tab:nb_treino]] e [[tab:nb_teste]] apresentam todos os casos, com os valores originais, o desfecho real, as probabilidades posteriores de sucesso e fracasso em cada rede e a classe predita com limiar de 50%. As linhas destacadas contêm pelo menos um erro. A Figura [[fig:prob_teste]] compara diretamente P(Sucesso | evidências) nos nove casos de teste; a Figura [[fig:conf_nb]] resume as matrizes de confusão; e a Tabela [[tab:met_nb]] e a Figura [[fig:met_nb]] consolidam as métricas.</p>
{t8}
{t9}
{fig_prob_teste()}
{fig_conf_nb()}
{t10}
{fig_met_nb()}

<h2>3.6 Matrizes de confusão, métricas e calibração</h2>
<p>A Rede 1 acerta 18/21 no treino (acurácia 85,7%, balanceada 84,7%) e 9/9 no teste (100,0%, balanceada 100,0%); na base completa, 27/30 (90,0%, balanceada 90,0%). A Rede 2 acerta 18/21 no treino (85,7%, balanceada 84,7%), mas apenas 7/9 no teste (77,8%, balanceada 66,7%); na base completa, 25/30 (83,3%, balanceada 83,3%). O padrão é claro: a Rede 2 mantém a sensibilidade (detecta todos os sucessos do teste), mas sua especificidade no teste é de apenas 33,3% — ela libera dois dos três pacientes que fracassariam (TE1 e TE3). Como o falso positivo é o erro mais grave, esse é um problema clinicamente relevante.</p>
<p>A Rede 2, no entanto, apresenta <b>maior certeza média</b>: 89,5% no treino e 93,1% no teste, contra 86,8% e 85,4% da Rede 1. O Brier score confirma a pior calibração: 0,128/0,159 (Rede 2) contra 0,095/0,038 (Rede 1). Em outras palavras, a Rede 2 é mais confiante e mais errada. Esse é um argumento contra escolher um modelo apenas pela “confiança” das probabilidades.</p>
<div class="box">
<h4>Por que a Rede 1 vai tão bem no teste?</h4>
<p>Os 9 casos de teste têm padrões relativamente bem separados: os três fracassos têm NIF ≥ −28, VT ≤ 450 e RR ≥ 24; os seis sucessos têm NIF ≤ −34 e RSBI ≤ 63. Com cortes clínicos em −30/105, a Rede 1 consegue separar os grupos. Isso não significa que a Rede 1 seja universalmente superior: com apenas 9 casos, o intervalo de confiança de 95% para a acurácia de 100% vai de 70,1% a 100% (Tabela [[tab:ci]]). O resultado é encorajador, mas não é prova de generalização.</p>
</div>

<h2>3.7 Limiar de decisão e assimetria de riscos</h2>
<p>O limiar padrão de 50% escolhe a classe de maior probabilidade posterior; ele não garante máxima acurácia balanceada e não incorpora custos assimétricos. A Tabela [[tab:th_nb]] mostra o efeito de exigir 60%, 75% ou 90% de probabilidade. Para a Rede 1, o limiar não altera os resultados até 75%. Na Rede 2, subir o limiar para 60–75% não elimina nenhum falso positivo e acrescenta um falso negativo (TR13, 59,2%); a partir de 80%, TE1 (77,2%) deixa de ser liberado, mas TR1 (95,8%) e TE3 (90,6%) continuam como falsos positivos. Mesmo com limiar de 90%, a Rede 2 acerta 8/9 no teste; essa é uma observação posterior sobre nove casos e não deve orientar a escolha do limiar.</p>
{t11}

<h2>3.8 Comparação crítica e análise dos casos de fronteira</h2>
<p>As redes convergem quando a evidência é inequívoca: NIF muito negativo, RR baixo, VT alto e RSBI baixo produzem posteriores de sucesso acima de 89% em ambas. Divergem quando a discretização grosseira muda a contagem de verossimilhanças. A Tabela [[tab:casos_criticos]] e a Figura [[fig:div]] isolam esses casos.</p>
{t13}
{fig_div()}
<p><b>TR1</b> é o caso mais desconfortável: NIF −30, RR 17, VT 500 e RSBI 34, com fracasso real. Todas as quatro variáveis parecem favoráveis; a Rede 1 estima 90,6% de sucesso e a Rede 2, 95,8%. O erro é irredutível com essas entradas e sugere que há fatores não medidos (força de tosse, secreção, estado mental, reserva cardiorrespiratória, critério de reintubação) ou que o rótulo incorpora um evento tardio. Esse caso é o principal argumento contra o uso autônomo do modelo.</p>
<p><b>TR16</b> mostra o efeito da perda de resolução: NIF −30, RR 30, VT 250 e RSBI 120; a Rede 1 estima 76,7% de sucesso (acerta), enquanto a Rede 2 estima 10,9% (erra). O VT baixo e o RSBI 120, combinados em duas faixas, dominam a verossimilhança da Rede 2. <b>TE1</b> é o espelho: NIF −15, RR 25, VT 450 e RSBI 56; a Rede 1 estima 29,7% (retém corretamente), a Rede 2 estima 77,2% (libera incorretamente). <b>TE3</b> combina NIF e RR limítrofes, VT baixo e RSBI favorável; a Rede 1 fica em 44,2% (retém), a Rede 2 em 90,6% (libera), e o desfecho real é fracasso — o caso é detalhado na Seção 4.7.</p>
<p>A comparação, portanto, não é “qual rede tem a maior acurácia média”, mas “qual delas erra de forma mais segura”. A Rede 1 erra apenas em TR1, TR13 e TR14; a Rede 2 erra em TR1, TR14, TR16, TE1 e TE3, com confiança alta em vários deles. A Rede 1 é mais conservadora onde importa.</p>

<h2>3.9 Robustez e redundância: validação leave-one-out</h2>
<p>A validação leave-one-out refaz priors e CPTs em cada dobra e mantém a discretização fixa. A Tabela [[tab:loo]] mostra que o conjunto completo tem 76,2% de acurácia LOO na Rede 1, contra 85,7% para NIF+RR+VT e para NIF isolado. Na Rede 2, o conjunto completo tem 81,0%, contra 71,4% para NIF+RR+VT e 76,2% para NIF isolado. O resultado não é uma demonstração de que “menos variáveis é melhor”; é a constatação de que, com 21 casos, a inclusão de três variáveis dependentes aumenta a variância das estimativas. O desempenho no teste da Rede 1 (9/9) mostra que o modelo completo ainda é útil, mas a análise reforça a necessidade de validação externa e de modelos mais parcimoniosos ou regularizados.</p>
{t12}

<h2>3.10 Execução no programa gráfico Netica 7.01 e Process Cases</h2>
<p>Ambas as redes em <span class="mono">netica/</span> foram projetadas para uso direto no programa gráfico <b>Netica Application 7.01</b> com os arquivos originais da disciplina: <span class="mono">Weaning_Rede1_Clinica.dne</span> (discretização clínica em três níveis contínuos por variável) e <span class="mono">Weaning_Rede2_Netica.dne</span> (discretização automática em dois níveis produzida por <i>Modify → Auto-Discretize Nodes</i> com 20% de arredondamento). Nas duas redes, o nó pai chama-se <b>Resultado</b> (estados <i>Fracasso</i> e <i>Sucesso</i>), compatível com o arquivo de controle <span class="mono">CTRL_Respira.txt</span> (<span class="mono">IDnum()</span>, <span class="mono">bel (Resultado,Fracasso)</span>, <span class="mono">bel (Resultado,Sucesso)</span>).</p>
<p><b>Fluxo operacional no Netica 7.01:</b> (1) abrir o arquivo <span class="mono">.dne</span> e executar <i>Network → Compile</i> com <i>Network → Automatic Updating</i> ativo; (2) para inspecionar um paciente específico na interface gráfica, usar <i>Cases → Get Case…</i> (F8) apontando para <span class="mono">Respira_DB_Teste_Process_Cases.txt</span> e informar o <span class="mono">IDnum</span> desejado; (3) para avaliação em lote de toda a base de treino ou teste, executar <i>Cases → Process Cases</i> fornecendo, nesta ordem, <span class="mono">CTRL_Respira.txt</span>, o arquivo de casos contínuos da disciplina (<span class="mono">Respira_DB_Treinamento_Process_Cases.txt</span> ou <span class="mono">Respira_DB_Teste_Process_Cases.txt</span>) e o arquivo de saída. Os quatro arquivos gerados pelo Netica 7.01 (<span class="mono">process_cases_rede&#123;1,2&#125;_&#123;treino,teste&#125;.txt</span>) concordam em 100% das 60 classificações com o Python (erro máximo 5,46 × 10<sup>−7</sup>).</p>
{netica_r1_fig}
{netica_r1_cpt_fig}
{netica_r2_fig}
<p>As Figuras [[fig:netica_te3_r1]] e [[fig:netica_te3_r2]] ilustram diretamente na tela do Netica 7.01 a divergência clínica no caso crítico <b>TE3</b> (NIF −28, RR 24, VT 310, RSBI 77; desfecho real <i>Fracasso</i>): enquanto a Rede 1 retém corretamente o paciente (55,8% Fracasso / 44,2% Sucesso), a Rede 2 o classifica com excesso de confiança como sucesso (90,6% Sucesso / 9,36% Fracasso).</p>
{netica_te3_r1_fig}
{netica_te3_r2_fig}
""")

# ---------------------------------------------------------------- 4 Fuzzy
# Tabela 14 — operadores
ops = [
    ["Método de inferência", "Mamdani", "Consequentes são conjuntos fuzzy; não há função linear."],
    ["Conectivo E (AND)", "mínimo (min)", "α = min(μ<sub>1</sub>,…,μ<sub>k</sub>) para os antecedentes presentes na regra."],
    ["Conectivo OU (OR)", "máximo (max)", "Não utilizado: todas as 12 regras compactas são conjuntivas."],
    ["Implicação", "mínimo (min)", "Corte do conjunto consequente na altura α."],
    ["Agregação", "máximo (max)", "União das regiões recortadas de todas as regras ativadas."],
    ["Defuzzificação", "centroide (COA)", "y* = ∫ y μ<sub>agg</sub>(y) dy / ∫ μ<sub>agg</sub>(y) dy."],
]
t14 = simple_table("operadores", "Arquitetura e operadores do sistema de inferência fuzzy Mamdani.", ["Etapa", "Operador", "Implementação"], ops, source="Fonte: elaboração própria. Implementação validada contra scikit-fuzzy e fuzzylite 4.0.", cls="keep", numeric_from=3)

# Tabela 15 — MFs
mf_rows = []
for _, r in mfs.iterrows():
    mf_rows.append([e(r["Variável"]), e(r["Termo"]), e(r["Tipo"]), e(r["Parâmetros [a,b,c(,d)]"]), e(r["Universo"])])
t15 = simple_table("mfs", "Universos de discurso, termos linguísticos, formatos e parâmetros exatos das funções de pertinência.", ["Variável", "Termo", "Forma", "Parâmetros", "Universo"], mf_rows, source="Fonte: elaboração própria com base em Yang & Tobin (1991), III Consenso, Hess & Kacmarek e Dräger.", cls="tiny keep", numeric_from=4)

# Tabela 16 — base completa de 12 regras compactas
rep_rows = []
for _, r in rules.iterrows():
    rep_rows.append([str(int(r["Regra"])), e(r["NIF"]), e(r["RR"]), e(r["VT"]), e(r["RSBI"]), str(int(r["Pontuação"])), e(r["Consequente"])])
t16 = simple_table("regras_rep", "Base completa de 12 regras clínicas Mamdani (4 de Sucesso e 8 de Fracasso). O traço (—) indica cláusula parcial (qualquer termo naquela variável), garantindo cobertura integral do espaço 3⁴ = 81.", ["#", "NIF", "RR", "VT", "RSBI", "Pontos", "Consequente"], rep_rows, source="Fonte: elaboração própria; formato SE…ENTÃO no Apêndice A e nos arquivos .fcl/.fis/.fll.", cls="tiny keep")

# Tabela 17 — fuzzy 30 casos
fz_rows = []
for _, r in fuzzy.iterrows():
    real = "S" if r["Resultado"] == "Sucesso" else "F"
    decis = {"Sucesso": "Sucesso", "Cautela": "Cautela", "Fracasso": "Fracasso"}[r["decisao"]]
    ok = "✓" if bool(r["acerto_conservador"]) else "✗"
    cls = "err" if ok == "✗" else ""
    fz_rows.append(([e(r["ID"]), n(r["NIF"],0), n(r["RR"],0), n(r["VT"],0), n(r["RSBI"],0), real, n(r["crisp"],2), decis, ok], cls))
head17 = '<tr><th>ID</th><th class="num">NIF</th><th class="num">RR</th><th class="num">VT</th><th class="num">RSBI</th><th>Real</th><th class="num">Resultado</th><th>Decisão sugerida</th><th>Acerto</th></tr>'
body17 = "".join(f'<tr class="{c}">' + "".join(f'<td class="{"num" if i in (1,2,3,4,6) else ""}">{x}</td>' for i, x in enumerate(row)) + "</tr>" for row, c in fz_rows)
t17 = tab("fuzzy30tab", "Avaliação do sistema Fuzzy nos 30 casos: entradas, desfecho real, saída Resultado defuzzificada, decisão sugerida e acerto da política conservadora (≥ 60).", head17, body17, source="Fonte: elaboração própria. A saída é o centroide de Resultado (0–100); a decisão conservadora só libera com Resultado ≥ 60.", cls="tiny", note="F = fracasso; S = sucesso. Cautela (40–60) é tratada como não liberação na avaliação binária conservadora.")

# Tabela 18 — métricas fuzzy
fm_rows = []
for _, r in fuzzy_metrics.iterrows():
    if r["Limiar"] not in (50, 60):
        continue
    fm_rows.append([e(r["Conjunto"]), str(int(r["Limiar"])), pct(r["acuracia"]), pct(r["acuracia_balanceada"]), pct(r["sensibilidade"]), pct(r["especificidade"]), pct(r["ppv"]), pct(r["npv"]), pct(r["f1"]), str(int(r["tn"])), str(int(r["fp"])), str(int(r["fn"])), str(int(r["tp"]))])
t18 = simple_table("met_fuzzy", "Métricas do sistema Fuzzy (incluindo acurácia balanceada) com limiares 50 e 60, no treino, no teste e na base completa.", ["Conjunto", "Limiar", "Acurácia", "Ac. bal.", "Sens.", "Espec.", "PPV", "NPV", "F1", "TN", "FP", "FN", "TP"], fm_rows, source="Fonte: elaboração própria. A classe positiva é Sucesso.", cls="tiny keep", note="O limiar 60 é a política conservadora adotada; 50 é mantido como referência.")

# Tabela 19 — metadados dos casos
meta_rows = []
for _, r in case_meta.iterrows():
    meta_rows.append([e(r["Caso"]), "S" if r["Desfecho real"] == "Sucesso" else "F", n(r["NIF"],0), n(r["RR"],0), n(r["VT"],0), n(r["RSBI"],0), pct(r["NB1 P(Sucesso)"]), pct(r["NB2 P(Sucesso)"]), n(r["crisp"],2), e(r["decisão"])])
t19 = simple_table("casos_meta", "Síntese dos três casos estudados: entradas, desfecho real, probabilidades das redes e saída fuzzy.", ["Caso", "Real", "NIF", "RR", "VT", "RSBI", "NB1 P(S)", "NB2 P(S)", "Fuzzy", "Decisão"], meta_rows, source="Fonte: elaboração própria. Fuzzy = saída Resultado (0–100); decisão conservadora se ≥ 60.", cls="tiny keep", numeric_from=2)

# Funções auxiliares dos casos
def case_fuzz_table(key, caption, cid):
    df = case_fuzz[case_fuzz["Caso"] == cid]
    rows = []
    for _, r in df.iterrows():
        cls = "" if r["μ(x)"] > 1e-9 else "zero"
        rows.append(([e(r["Variável"]), n(r["Valor"], 0), e(r["Termo"]), n(r["μ(x)"], 4)], cls))
    head = '<tr><th>Variável</th><th class="num">Valor</th><th>Termo linguístico</th><th class="num">μ(x)</th></tr>'
    body = "".join(f'<tr class="{c}">' + "".join(f'<td class="{"num" if i in (1,3) else ""}">{x}</td>' for i, x in enumerate(row)) + "</tr>" for row, c in rows)
    return tab(key, caption, head, body, source="Fonte: elaboração própria. μ(x) calculado com os parâmetros exatos da Tabela [[tab:mfs]] antes do arredondamento.", cls="tiny keep", note="Termos com μ = 0 são mostrados para completar a fuzzificação; não contribuem para nenhuma regra.")


def case_rules_table(key, caption, cid):
    df = case_rules[case_rules["Caso"] == cid].copy()
    rows = []
    for _, r in df.iterrows():
        terms = f"{w.TERM_DISPLAY[r['NIF']]} / {w.TERM_DISPLAY[r['RR']]} / {w.TERM_DISPLAY[r['VT']]} / {w.TERM_DISPLAY[r['RSBI']]}"
        active_vals = [n(r[f"mu_{v}"], 3) for v in w.VARIABLES if r[v] != "Qualquer"]
        calc = f"min({'; '.join(active_vals)})"
        rows.append([str(int(r["Regra"])), e(terms), calc, n(r["alpha"],4), e(w.TERM_DISPLAY[r["Consequente"]])])
    head = '<tr><th>#</th><th>Antecedentes (NIF / RR / VT / RSBI)</th><th>Cálculo de α</th><th class="num">α</th><th>Consequente</th></tr>'
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i==3 else ""}">{x}</td>' for i, x in enumerate(row)) + "</tr>" for row in rows)
    return tab(key, caption, head, body, source="Fonte: elaboração própria. α = min(μ antecedentes presentes na regra), conforme o conectivo E de Mamdani.", cls="tiny keep", note="O traço (—) indica variável livre na regra compacta. A implicação corta o conjunto consequente em min(α, μ<sub>consequente</sub>(y)).")


def case_def_table(key, caption, cid):
    r = case_def[case_def["Caso"] == cid].iloc[0]
    rows = [
        ["Área agregada", f"∫ μ<sub>agg</sub>(y) dy", n(r["Área agregada (∫μ dy)"], 4)],
        ["Momento", f"∫ y · μ<sub>agg</sub>(y) dy", n(r["Momento (∫y·μ dy)"], 4)],
        ["Centroide", "y* = momento / área", n(r["Centroide y* = momento/área"], 4)],
        ["Regras com α > 0", "—", str(int(r["Regras com α>0"]))],
        ["Decisão conservadora", "≥ 60", e(r["Decisão conservadora"])],
    ]
    head = "<tr><th>Grandeza</th><th>Definição</th><th class=\"num\">Valor</th></tr>"
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i==2 else ""}">{x}</td>' for i, x in enumerate(row)) + "</tr>" for row in rows)
    return tab(key, caption, head, body, source="Fonte: elaboração própria. Integração por trapézios na malha de 0,01 unidade.", cls="cm keep")


t20 = case_fuzz_table("fz_te5", "Caso 1 (TE5) — fuzzificação completa das quatro entradas.", "TE5")
t21 = case_rules_table("rules_te5", "Caso 1 (TE5) — regras ativadas (α > 0) e cálculo do mínimo.", "TE5")
t22 = case_def_table("def_te5", "Caso 1 (TE5) — agregação e defuzzificação pelo centroide.", "TE5")
t23 = case_fuzz_table("fz_tr10", "Caso 2 (TR10) — fuzzificação completa das quatro entradas.", "TR10")
t24 = case_rules_table("rules_tr10", "Caso 2 (TR10) — regras ativadas (α > 0) e cálculo do mínimo.", "TR10")
t25 = case_def_table("def_tr10", "Caso 2 (TR10) — agregação e defuzzificação pelo centroide.", "TR10")
t26 = case_fuzz_table("fz_te3", "Caso 3 (TE3) — fuzzificação completa das quatro entradas.", "TE3")
t27 = case_rules_table("rules_te3", "Caso 3 (TE3) — regras ativadas (α > 0) e cálculo do mínimo.", "TE3")
t28 = case_def_table("def_te3", "Caso 3 (TE3) — agregação e defuzzificação pelo centroide.", "TE3")

# Valores dinâmicos dos 3 casos para eliminar qualquer dessincronia no texto
_d_te5 = case_def[case_def["Caso"] == "TE5"].iloc[0]
_d_tr10 = case_def[case_def["Caso"] == "TR10"].iloc[0]
_d_te3 = case_def[case_def["Caso"] == "TE3"].iloc[0]

parts.append(f"""
<h1 id="sec-fuzzy">4. Modelagem difusa: sistema de inferência Mamdani</h1>

<h2>4.1 Arquitetura, operadores e universo de discurso</h2>
<p>O sistema de inferência fuzzy foi construído com quatro entradas contínuas — NIF, RR, VT e RSBI — e uma saída, <b>Resultado</b>, no universo 0–100, com dois termos linguísticos possíveis: <b>Fracasso</b> e <b>Sucesso</b>. Quanto maior o valor defuzzificado de <b>Resultado</b>, maior o grau de pertinência ao desfecho de sucesso na interrupção da ventilação mecânica. A escolha de manter as quatro variáveis permite comparar diretamente com as redes e respeita o escopo do trabalho; a redundância do RSBI é tratada como uma decisão de modelagem, não como uma hipótese de independência.</p>
<p>O método é Mamdani. Cada regra tem antecedentes conjuntivos e um consequente linguístico (<b>Fracasso</b> ou <b>Sucesso</b>); o conectivo E é o mínimo, a implicação é o mínimo (corte), a agregação é o máximo e a defuzzificação é o centroide (COA). A Tabela [[tab:operadores]] resume a arquitetura. A implementação manual foi validada tanto contra <span class="mono">scikit-fuzzy</span> (diferença máxima &lt; 10<sup>−5</sup>) quanto no motor oficial <b>fuzzylite 4.0</b> e na interface gráfica <b>QtFuzzyLite 4.0</b>.</p>
{t14}

<h2>4.2 Funções de pertinência e justificativa clínica</h2>
<p>A Tabela [[tab:mfs]] apresenta os universos, os termos e os parâmetros exatos; a Figura [[fig:mf]] mostra as funções. As extremidades são trapezoidais para representar saturação fisiológica (por exemplo, NIF ≤ −35 é “forte” com pertinência 1; NIF ≥ −15 é “fraco” com pertinência 1). As transições são triangulares nas entradas e trapezoidais simétricas na saída (<b>Fracasso</b> em [0, 0, 35, 65] e <b>Sucesso</b> em [35, 65, 100, 100]), de modo que um valor próximo ao corte compartilhe pertinência entre os dois termos. As sobreposições são a representação formal da incerteza clínica: em vez de decidir que −30,1 é “forte” e −29,9 é “limítrofe”, o sistema atribui graus.</p>
{t15}
{fig_mf()}
<p>As duas superfícies da Figura [[fig:surf1]] e da Figura [[fig:surf2]] ajudam a visualizar a interação entre as variáveis. A primeira fixa VT e RSBI em valores adequados e mostra como o resultado defuzzificado cresce em direção a <b>Sucesso</b> com a força inspiratória e com a redução da frequência respiratória. A segunda fixa NIF e RR e evidencia que o VT e o RSBI não são redundantes na prática: volumes baixos deslocam a superfície para <b>Fracasso</b> mesmo com RSBI aparentemente favorável. As linhas de nível 40 e 60 delimitam a faixa de transição entre <b>Fracasso</b> e <b>Sucesso</b>.</p>
{fig_surf1()}
{fig_surf2()}
<ul>
<li><b>NIF:</b> “Forte” satura em ≤ −35 e decai até −25; “Limítrofe” tem pico em −25; “Fraco” cresce a partir de −25 e satura em ≥ −15. Os cruzamentos de 50% ocorrem em −30 e −20, exatamente os cortes clínicos. A zona −30 a −20 é, portanto, uma região de pertinência compartilhada.</li>
<li><b>RR:</b> “Adequada” satura até 22 e decai até 30; “Limítrofe” tem pico em 30; “Elevada” cresce de 30 a 38 e satura acima de 38. A transição é mais larga que o corte único de 30, refletindo a variabilidade fisiológica e a diferença entre os limiares 30 e 35 da literatura.</li>
<li><b>VT:</b> “Baixo” satura até 300 e decai até 400; “Adequado” tem pico em 475; “Elevado” começa em 600 e satura em 750. A faixa 300–400 mL é a transição em torno de 5 mL/kg; o limite superior de 600 mL evita tratar volumes muito altos como automaticamente melhores.</li>
<li><b>RSBI:</b> “Favorável” satura até 90 e decai até 120; “Limítrofe” tem pico em 120; “Desfavorável” começa em 120 e satura em 150. O cruzamento favorável/limítrofe em 105 recupera o corte clássico; o cruzamento limítrofe/desfavorável em 135 cria a faixa de cautela.</li>
<li><b>Saída (Resultado):</b> dois termos linguísticos — <b>Fracasso</b> ([0, 0, 35, 65]) e <b>Sucesso</b> ([35, 65, 100, 100]), cruzando μ = 0,5 exatamente em 50. A sobreposição em [35, 65] permite que a transição entre fracasso e sucesso seja contínua quando regras de ambos os desfechos são ativadas simultaneamente.</li>
</ul>

<h2>4.3 Base de regras (12 regras compactas), completude e consistência</h2>
<p>A base foi condensada em <b>12 regras clínicas Mamdani</b> — <b>4 regras de Sucesso</b> (R1–R4) e <b>8 regras de Fracasso</b> (R5–R12) — utilizando cláusulas parciais (variáveis livres indicadas por “—”) de forma equivalente à partição clínica do espaço completo 3<sup>4</sup> = 81. A hierarquia de pesos subjacente prioriza força inspiratória (NIF), frequência respiratória (RR) e RSBI, com VT atuando como fator de validação e proteção contra hipoventilação ou esforço extremo. Qualquer sinal claramente desfavorável (NIF Fraco, RR Elevada ou RSBI Desfavorável) dispara individualmente uma regra de <b>Fracasso</b> (R5–R7), garantindo comportamento conservador e auditável.</p>
<p>A Tabela [[tab:regras_rep]] lista integralmente as 12 regras; a Figura [[fig:regras]] mostra a distribuição de consequentes e o mapa de decisão no plano NIF × RR; e o Apêndice A traz todas as frases completas no formato SE…ENTÃO. A completude foi verificada computacionalmente: cada universo tem cobertura positiva em toda a malha, e em 2.000 avaliações aleatórias mais os 16 cantos extremos do domínio nenhuma entrada ficou sem regra ativada. Além disso, a base enxuta permite inspecionar todas as 12 regras simultaneamente em uma única tela do <b>QtFuzzyLite 4.0</b>.</p>
{t16}
{fig_regras()}
<div class="box neutral">
<h4>Comportamento do centroide Mamdani com dois termos de saída</h4>
<p>Uma propriedade geométrica importante do método Mamdani merece destaque: quando apenas um dos trapézios de saída é ativado com altura α = 1,0, o centroide coincide com o baricentro desse conjunto — <b>25,75</b> para <b>Fracasso</b> e <b>74,25</b> para <b>Sucesso</b> —, independentemente de quantas regras do mesmo sinal disparem. Consequentemente, casos inequivocamente ruins (como TR2, TR5–TR12, TE1–TE2) ou inequivocamente bons (TR15, TR17, TR20, TE7, TE9) agrupam-se nesses platôs, enquanto casos de fronteira em que regras de <b>Fracasso</b> e de <b>Sucesso</b> disparam simultaneamente (como <b>TE3 = 40,05</b>) produzem interpolação contínua na zona de cautela (40–60). Caso se deseje discriminação mais fina entre graus internos de risco (por exemplo, diferenciar fracasso moderado de extremo), a alternativa natural é expandir a saída para cinco termos linguísticos (muito baixo a muito alto), ao custo de maior complexidade de especificação.</p>
</div>

<h2>4.4 Avaliação na base completa e limiar de decisão</h2>
<p>Tanto com o limiar simétrico de <b>50/100</b> (ponto de cruzamento entre <b>Fracasso</b> e <b>Sucesso</b>) quanto com o limiar conservador de <b>60/100</b>, o sistema classifica exatamente os mesmos casos na base. A faixa 40–60 em torno de 50 representa a zona de transição/cautela onde ambos os termos (<b>Fracasso</b> e <b>Sucesso</b>) disputam o centroide. A Tabela [[tab:fuzzy30tab]] apresenta os 30 casos, a Figura [[fig:fuzzy30]] os posiciona em relação aos limiares e a Tabela [[tab:met_fuzzy]] consolida as métricas.</p>
{t17}
{fig_fuzzy30()}
<p>O FIS acerta 17/21 no treino (acurácia 81,0%, balanceada 79,2%), <b>9/9 no teste (100,0%, balanceada 100,0%)</b> e 26/30 na base completa (86,7%, balanceada 86,7%), tanto no limiar 50 quanto no limiar 60. A sensibilidade é 80,0%, a especificidade 93,3%, o PPV 92,3%, o NPV 82,4% e o F1 85,7%. Há apenas um falso positivo (TR1) e três falsos negativos (TR13, TR14 e TR16).</p>
{fig_fuzzy_conf()}
{t18}
<p>Os três falsos negativos merecem leitura clínica. TR13 (NIF −24, RR 24, VT 300, RSBI 80) e TR14 (NIF −23, RR 29, VT 300, RSBI 97) são pacientes com VT baixo e NIF limítrofe; o sistema classifica como <b>Fracasso</b>. TR16 (NIF −30, RR 30, VT 250, RSBI 120) tem VT muito baixo, RR no pico de “Limítrofe” e RSBI limítrofe; o NIF −30 fica no cruzamento entre “Forte” e “Limítrofe” (μ = 0,5 em cada) e não compensa esses sinais, de modo que o sistema prefere não liberar. Em uma política conservadora, esses erros são aceitáveis: o paciente permanece no ventilador e é reavaliado, em vez de ser extubado e precisar de reintubação. O único falso positivo, TR1, é o caso paradoxal já discutido.</p>
<div class="box warn">
<h4>Leitura do limiar</h4>
<p>O ponto médio entre os trapézios <b>Fracasso</b> e <b>Sucesso</b> é 50,0, enquanto a faixa 40–60 indica ativação simultânea importante de regras de fracasso e de sucesso. O importante é que o limiar seja definido <i>a priori</i>, documentado e auditado, e que os casos próximos a 40–60 sejam submetidos a reavaliação clínica criteriosa.</p>
</div>

<h2>4.5 Estudo de caso 1: TE5 — sucesso claro</h2>
<p>O caso TE5 é um paciente do conjunto de teste com NIF −48 cm.H₂O, RR 24 resp/min, VT 380 mL e RSBI 63 resp/min/L; o desfecho real foi <b>Sucesso</b>. Todos os indicadores são favoráveis ou adequados, e as duas redes também acertam (P(Sucesso) = 89,4% e 95,8%). A Tabela [[tab:casos_meta]] sintetiza os três casos; as Tabelas [[tab:fz_te5]]–[[tab:def_te5]] e as Figuras [[fig:case_TE5_in]]–[[fig:case_TE5_out]] detalham o processo.</p>
{t19}
{t20}
<p><b>Fuzzificação.</b> NIF −48 é totalmente “Forte” (μ = 1,000). RR 24 é 0,750 “Adequada” e 0,250 “Limítrofe”, pois está na transição. VT 380 é 0,200 “Baixo” e 0,457 “Adequado”. RSBI 63 é totalmente “Favorável” (μ = 1,000). Os demais termos têm μ = 0 e não contribuem para nenhuma regra.</p>
{t21}
<p><b>Regras ativadas.</b> Duas regras disparam, ambas com consequente <b>Sucesso</b>: <b>R1</b> (NIF Forte E RR Adequada E RSBI Favorável → Sucesso) com α = min(1,000; 0,750; 1,000) = 0,750, e <b>R2</b> (NIF Forte E RR Limítrofe E RSBI Favorável → Sucesso) com α = min(1,000; 0,250; 1,000) = 0,250. Como a agregação usa o máximo sobre o mesmo consequente, a regra dominante R1 recorta o trapézio <b>Sucesso</b> na altura 0,750.</p>
{t22}
<p><b>Defuzzificação.</b> A região agregada tem área <b>{n(_d_te5['Área agregada (∫μ dy)'], 3)}</b> e momento <b>{n(_d_te5['Momento (∫y·μ dy)'], 3)}</b>, resultando em <b>y* = {n(_d_te5['Centroide y* = momento/área'], 3)}</b> (100% no platô de <b>Sucesso</b>; no QtFuzzyLite 4.0: 72,732). O valor é claramente superior a 60 (e a 50); a decisão é <b>Sucesso</b>, coerente com o desfecho. A interpretação clínica é de força inspiratória preservada, drive normal e padrão respiratório eficiente.</p>
{fig_case("TE5")}

<h2>4.6 Estudo de caso 2: TR10 — fracasso claro</h2>
<p>O caso TR10 é um paciente do treino com NIF −20 cm.H₂O, RR 33 resp/min, VT 124 mL e RSBI 266 resp/min/L; o desfecho real foi <b>Fracasso</b>. É o oposto clínico de TE5: força reduzida, taquipneia, volume corrente muito baixo e RSBI extremamente alto. As duas redes estimam probabilidade de sucesso próxima de zero (0,97% e 1,80%).</p>
{t23}
<p><b>Fuzzificação.</b> NIF −20 divide-se igualmente entre “Fraco” (0,500) e “Limítrofe” (0,500). RR 33 é 0,625 “Limítrofe” e 0,375 “Elevada”. VT 124 é totalmente “Baixo” (1,000) e RSBI 266 é totalmente “Desfavorável” (1,000).</p>
{t24}
<p><b>Regras ativadas.</b> Cinco regras disparam, todas com consequente <b>Fracasso</b>: <b>R7</b> (RSBI Desfavorável) com α = 1,000; <b>R5</b> (NIF Fraco), <b>R8</b> (NIF Limítrofe E VT Baixo) e <b>R10</b> (NIF Limítrofe E RR Limítrofe) com α = 0,500; e <b>R6</b> (RR Elevada) com α = 0,375. Como R7 dispara com força plena (1,000), a região agregada pelo máximo é o trapézio completo <b>Fracasso</b>.</p>
{t25}
<p><b>Defuzzificação.</b> A área é <b>{n(_d_tr10['Área agregada (∫μ dy)'], 3)}</b>, o momento <b>{n(_d_tr10['Momento (∫y·μ dy)'], 3)}</b> e o centroide <b>y* = {n(_d_tr10['Centroide y* = momento/área'], 3)}</b> (100% no platô de <b>Fracasso</b>; idêntico no QtFuzzyLite 4.0: 25,750). A decisão é <b>Fracasso</b>, coerente com o desfecho. A interpretação é de padrão rápido e superficial grave, com força inspiratória insuficiente; a liberação seria claramente insegura.</p>
{fig_case("TR10")}

<h2>4.7 Estudo de caso 3: TE3 — crítico/duvidoso</h2>
<p>O caso TE3 é o mais importante para a discussão. Paciente do teste com NIF −28 cm.H₂O, RR 24 resp/min, VT 310 mL e RSBI 77 resp/min/L; o desfecho real foi <b>Fracasso</b>. Há conflito entre os sinais: NIF e RR estão em transição, VT está baixo e RSBI é favorável. A Rede 1 estima 44,2% de sucesso (retém), a Rede 2 estima 90,6% (libera incorretamente) e o sistema fuzzy produz <b>Resultado = {n(_d_te3['Centroide y* = momento/área'], 2)}</b> (ativando simultaneamente <b>Fracasso</b> com α = 0,700 e <b>Sucesso</b> com α = 0,300, situando-se na fronteira de <b>Cautela / Fracasso</b>). Este é o exemplo mais claro de por que a discretização e o tratamento da incerteza importam.</p>
{t26}
<p><b>Fuzzificação.</b> NIF −28 é 0,700 “Limítrofe” e 0,300 “Forte”. RR 24 é 0,750 “Adequada” e 0,250 “Limítrofe”. VT 310 é 0,900 “Baixo” e 0,057 “Adequado”. RSBI 77 é totalmente “Favorável” (1,000). O caso combina uma variável favorável (RSBI), duas mistas (NIF e RR) e uma desfavorável (VT), exatamente o tipo de conflito que um corte rígido não resolve bem.</p>
{t27}
<p><b>Regras ativadas.</b> Cinco regras disparam com consequentes opostos. Pelo lado de <b>Fracasso</b>, <b>R8</b> (NIF Limítrofe E VT Baixo) dispara com α = min(0,700; 0,900) = 0,700 e <b>R10</b> (NIF Limítrofe E RR Limítrofe) com α = 0,250. Pelo lado de <b>Sucesso</b>, <b>R1</b> (NIF Forte E RR Adequada E RSBI Favorável) dispara com α = 0,300, <b>R2</b> com α = 0,250 e <b>R4</b> com α = 0,057. A agregação pelo máximo preserva a competição bimodal entre <b>Fracasso</b> (altura 0,700) e <b>Sucesso</b> (altura 0,300).</p>
{t28}
<p><b>Defuzzificação.</b> A área é <b>{n(_d_te3['Área agregada (∫μ dy)'], 3)}</b>, o momento <b>{n(_d_te3['Momento (∫y·μ dy)'], 3)}</b> e o centroide <b>y* = {n(_d_te3['Centroide y* = momento/área'], 3)}</b> (&lt; 50, com pertinência 0,832 em <b>Fracasso</b> e 0,168 em <b>Sucesso</b>; idêntico no QtFuzzyLite 4.0: 40,048). A recomendação é <b>Fracasso / não liberar (faixa de cautela)</b>, acertando o desfecho real onde a Rede 2 errou com 90,6% de confiança.</p>
{fig_case("TE3")}

<h2>4.8 Execução no programa gráfico QtFuzzyLite 4.0 e síntese da modelagem difusa</h2>
<p>O sistema Mamdani foi exportado nos formatos <span class="mono">fuzzy/weaning_mamdani.fll</span> (nativo FuzzyLite), <span class="mono">fuzzy/weaning_mamdani.fcl</span> (IEC 61131-7) e <span class="mono">fuzzy/weaning_mamdani.fis</span> (MATLAB/Octave), todos validados com código de saída zero no motor oficial <b>fuzzylite 4.0</b> (<span class="mono">fuzzylite.exe</span>) e carregados diretamente na interface gráfica <b>QtFuzzyLite 4.0</b> (<span class="mono">qtfuzzylite.exe</span>). Na aba <i>Setup</i>, as 12 regras são verificadas sem erros (<i>Good Rules: 12, Bad Rules: 0</i>); na aba <i>Control</i>, os sliders de NIF, RR, VT e RSBI exibem em tempo real a fuzzificação, as regras ativadas em verde com sua força <i>Degree</i> e o centroide de <b>Resultado</b>; e em <i>Tools → Surface 2D</i> são gerados os mapas bidimensionais de decisão (capturas reais apresentadas no Apêndice C.6).</p>
<p>Em síntese, o FIS entrega três vantagens: (i) uma variável de saída <b>Resultado</b> diretamente interpretável nos dois desfechos clínicos (<b>Fracasso</b> e <b>Sucesso</b>); (ii) transição contínua na faixa intermediária quando regras opostas competem (como em TE3); e (iii) base enxuta de 12 regras linguisticamente auditáveis com cobertura total. O desempenho é de 26/30 (86,7%) na base completa e 9/9 (100,0%) no teste, com especificidade de 93,3%.</p>
""")

# ---------------------------------------------------------------- 5 Comparação
fin_rows = []
for _, r in final.iterrows():
    fin_rows.append([e(r["Modelo"]), pct(r["Acurácia"]), pct(r["Acurácia bal."]), pct(r["Sensibilidade"]), pct(r["Especificidade"]), pct(r["PPV"]), pct(r["NPV"]), pct(r["F1"]), str(int(r["TN"])), str(int(r["FP"])), str(int(r["FN"])), str(int(r["TP"]))])
t29 = simple_table("final", "Comparação final dos três modelos na base completa (n = 30), com a classe positiva Sucesso.", ["Modelo", "Acurácia", "Ac. bal.", "Sens.", "Espec.", "PPV", "NPV", "F1", "TN", "FP", "FN", "TP"], fin_rows, source="Fonte: elaboração própria. O Fuzzy usa a política conservadora de liberar apenas com Resultado ≥ 60.", cls="tiny keep")
qual = [
    ["Saída", "Probabilidade posterior de sucesso, sem calibração independente", "Probabilidade posterior de sucesso, sem calibração independente", "Saída contínua Resultado (0–100) + faixa de cautela (40–60)"],
    ["Transparência", "CPTs e priors auditáveis; leitura por variável", "CPTs e priors auditáveis; duas faixas por variável", "12 regras clínicas compactas (cobertura integral) e MFs"],
    ["Incerteza no limiar", "Posteriores extremas; pode parecer mais segura do que é", "Posteriores extremas; pior calibração (Brier 0,137)", "Transições graduadas; competição explícita na fronteira"],
    ["Independência", "Hipótese violada por RR/VT/RSBI", "Hipótese violada por RR/VT/RSBI", "Não exige independência, mas reforça o padrão respiratório"],
    ["Desempenho (30)", "90,0% de acurácia (bal. 90,0%); 86,7% sens.; 93,3% espec.", "83,3% de acurácia (bal. 83,3%); 86,7% sens.; 80,0% espec.", "86,7% de acurácia (bal. 86,7%); 80,0% sens.; 93,3% espec."],
    ["Principal fragilidade", "Redundância e confiança excessiva em casos extremos", "Perda de resolução clínica; falsos positivos", "Platôs em 25,75/74,25 em casos extremos; sensibilidade menor"],
    ["Uso sugerido", "Triagem inicial e explicação probabilística (Netica)", "Comparação/benchmark; não usar sozinha", "Segunda opinião em casos 40–60 e decisão conservadora (QtFuzzyLite)"],
]
t30 = simple_table("qualitativa", "Comparação qualitativa dos modelos: interpretabilidade, tratamento da incerteza e papel no fluxo de decisão.", ["Critério", "Naive Bayes 1 (clínica)", "Naive Bayes 2 (Netica)", "Fuzzy Mamdani"], qual, source="Fonte: elaboração própria.", cls="tiny keep", numeric_from=4)
ci_rows = []
for _, r in ci.iterrows():
    ci_rows.append([e(r["Modelo"]), str(int(r["Acertos"])) + "/9", pct(r["IC95% inferior"]), pct(r["IC95% superior"])])
t31 = simple_table("ci", "Intervalos de confiança de Wilson (95%) para a acurácia no conjunto de teste (n = 9).", ["Modelo", "Acertos", "IC95% inferior", "IC95% superior"], ci_rows, source="Fonte: elaboração própria. A amplitude reflete o tamanho reduzido do teste.", cls="keep", numeric_from=1)

parts.append(f"""
<h1 id="sec-conclusao">5. Comparação geral, limitações e conclusão</h1>

<h2>5.1 Desempenho preditivo</h2>
<p>A Tabela [[tab:final]] e a Figura [[fig:final]] consolidam a comparação. A Rede 1 é o melhor modelo global: 90,0% de acurácia (balanceada 90,0%), 86,7% de sensibilidade, 93,3% de especificidade e F1 de 89,7% na base completa; no teste, 9/9 (100,0%). A Rede 2 tem a mesma sensibilidade na base completa, mas especificidade de 80,0% e acurácia de 83,3%; no teste, cai para 77,8% (balanceada 66,7%), com dois falsos positivos. O Fuzzy, com limiar 60 (ou 50), fica em 86,7% de acurácia (balanceada 86,7%), 80,0% de sensibilidade e 93,3% de especificidade; no teste, 9/9 (100,0%). Em termos de segurança (especificidade), Rede 1 e Fuzzy empatam; em sensibilidade, Rede 1 e Rede 2 empatam; em acurácia e F1, a Rede 1 lidera.</p>
{t29}
{fig_final()}
<p>É importante não superinterpretar a diferença de 27/30 versus 26/30 versus 25/30: com 30 casos, um único caso muda a acurácia em 3,3 pontos percentuais. A Tabela [[tab:ci]] mostra que o intervalo de 95% para 9/9 no teste vai de 70,1% a 100%; o de 7/9, de 45,3% a 93,7%. As diferenças observadas são compatíveis com grande incerteza amostral. O que parece robusto é o padrão qualitativo: a discretização clínica e o fuzzy conservador erram menos na direção perigosa (liberar quem fracassaria) do que a discretização automática grosseira.</p>
{t31}

<h2>5.2 Interpretabilidade, incerteza e adequação clínica</h2>
<p><b>Interpretabilidade.</b> A Rede 1 permite responder perguntas como “qual a probabilidade de sucesso se o NIF for forte e o RSBI for favorável?” diretamente a partir das CPTs no Netica. A Rede 2 tem a mesma estrutura, mas a leitura é menos clínica porque as faixas agregam estados heterogêneos. O Fuzzy oferece a linguagem mais próxima do raciocínio médico — “SE o NIF é forte E a frequência é adequada E o RSBI é favorável, ENTÃO Resultado é Sucesso” —, com uma base compacta de apenas 12 regras legíveis integralmente em tela única no QtFuzzyLite 4.0.</p>
<p><b>Incerteza.</b> As redes produzem probabilidades, mas com 21 casos e forte dependência entre preditores elas são pontiagudas demais. A Rede 2, em particular, é mais confiante e menos calibrada. O Fuzzy não produz uma probabilidade; produz um grau contínuo de <b>Resultado</b> entre 0 e 100. Isso é uma vantagem para comunicação clínica (“40,05/100, zona de cautela”) e admite sinalizar dúvida em vez de forçar alta confiança em casos limítrofes.</p>
<p><b>Correlação.</b> O Fuzzy não exige independência condicional, mas ao usar RR, VT e RSBI também reforça o mesmo padrão. A diferença é que essa decisão é visível na estrutura das 12 regras, enquanto no Naive Bayes ela é uma hipótese implícita. Nenhum dos modelos “resolve” a fisiologia: eles apenas tornam explícitas diferentes formas de simplificá-la.</p>
<p><b>Fluxo recomendado.</b> Em um serviço real, a Rede 1 pode funcionar como triagem probabilística e ferramenta de explicação; o Fuzzy, como segunda opinião nos casos de fronteira (por exemplo, 40–60 em Resultado ou 40–60% de probabilidade). A decisão final deve integrar gasometria, nível de consciência, força de tosse, volume de secreção, estabilidade hemodinâmica, balanço hídrico, comorbidades e a evolução clínica. O modelo é um apoio, não um árbitro.</p>
{t30}

<h2>5.3 Limitações</h2>
<ol>
<li><b>Amostra pequena e histórica:</b> 30 tentativas, 9 no teste, de um estudo de 1992; a prevalência e o protocolo de desmame podem não refletir a prática atual. Os intervalos de confiança são largos.</li>
<li><b>Variáveis estáticas:</b> cada caso tem uma única medida; não há tendência temporal, resposta a um teste de respiração espontânea, gasometria, relação PaO₂/FiO₂, força de tosse ou avaliação de secreção.</li>
<li><b>Independência condicional:</b> RR, VT e RSBI são dependentes; as posteriores podem ser excessivamente extremas. A validação leave-one-out mostra instabilidade com o conjunto completo.</li>
<li><b>Definição do desfecho:</b> o critério de sucesso/fracasso e o protocolo de reintubação não estão detalhados na base; o caso TR1 sugere que variáveis omitidas podem ser decisivas.</li>
<li><b>Discretização automática:</b> a execução usa Netica Application 7.01, dois estados e 20% de arredondamento; outras opções podem produzir cortes diferentes. A rede nativa e os hashes registram as opções efetivamente usadas.</li>
<li><b>Limiar e saída fuzzy:</b> 60 é uma escolha de política conservadora (equivalente a 50 nesta base); a saída com dois termos concentra casos extremos nos platôs 25,75 e 74,25, reservando a graduação contínua para conflitos de fronteira como TE3.</li>
<li><b>Ausência de validação externa:</b> não há coorte independente, análise de curvas de decisão, calibração prospectiva ou comparação com o julgamento de especialistas.</li>
</ol>

<h2>5.4 Conclusão</h2>
<p>O trabalho mostrou que é possível construir, com 21 casos de treino, modelos interpretáveis e reprodutíveis para apoiar a decisão de desmame, implementados tanto em código auditável quanto nos programas gráficos oficiais <b>Netica 7.01</b> e <b>QtFuzzyLite 4.0</b>. A Rede 1, com cortes clínicos e suavização de Laplace, foi o modelo com melhor equilíbrio entre acurácia, sensibilidade, especificidade e calibração (90,0% geral; 9/9 no teste). O sistema Mamdani com 12 regras compactas entregou desempenho competitivo (86,7% geral; 9/9 no teste; especificidade 93,3%) e explicitou a incerteza no caso crítico TE3. A Rede 2, embora rápida e automática, evidenciou o custo clínico de uma discretização binária grosseira: maior confiança aparente, porém menor especificidade (33,3% no teste) e falsos positivos graves.</p>
<p>A comparação mais importante não é numérica, mas clínica. TE3 — sucesso previsto com 90,6% pela Rede 2, retido com 44,2% pela Rede 1, sinalizado em cautela (40,05) pelo Fuzzy e fracasso real — sintetiza o argumento: um modelo único, especialmente se excessivamente confiante, pode falhar justamente onde a decisão é mais difícil. A recomendação é usar os modelos como apoio, com limiar conservador, faixa de cautela e decisão humana obrigatória.</p>
""")

# ---------------------------------------------------------------- Referências
parts.append(r"""
<h1 id="refs">Referências</h1>
<div class="refs">
<p>ASHUTOSH, K. et al. Prediction criteria for successful weaning from respiratory support: statistical and connectionist analyses. <i>Critical Care Medicine</i>, v. 20, n. 9, p. 1295–1301, 1992. DOI: 10.1097/00003246-199209000-00017.</p>
<p>YANG, K. L.; TOBIN, M. J. A prospective study of indexes predicting the outcome of trials of weaning from mechanical ventilation. <i>New England Journal of Medicine</i>, v. 324, n. 21, p. 1445–1450, 1991. DOI: 10.1056/NEJM199105233242101.</p>
<p>GOLDWASSER, R. et al. Desmame e interrupção da ventilação mecânica. <i>III Consenso Brasileiro de Ventilação Mecânica</i>. São Paulo: Associação de Medicina Intensiva Brasileira, 2007.</p>
<p>SILVA, M. G. B. et al. Application of mechanical ventilation weaning predictors after elective cardiac surgery. <i>Brazilian Journal of Cardiovascular Surgery</i>, v. 30, n. 6, p. 605–609, 2015.</p>
<p>HESS, D. R.; KACMAREK, R. M. <i>Essentials of mechanical ventilation</i>. 4. ed. New York: McGraw-Hill, 2019. Tabela 16-2.</p>
<p>DRÄGER. <i>Did you know? The negative inspiratory force index (NIF)</i>. Disponível em: https://www.draeger.com/Library/Content/didyouknow_1_NIF_en.pdf. Acesso em: 20 out. 2026.</p>
<p>PENZ, C. A. <i>Base de conhecimento: weaning from mechanical ventilation</i>. Material didático da disciplina Sistemas Probabilísticos e Difusos, IFSC, Florianópolis, 2024.</p>
<p>NORSYS SOFTWARE CORPORATION. <i>Netica application user's guide</i>. Vancouver: Norsys, 2023. Disponível em: https://www.norsys.com. Acesso em: 20 out. 2026.</p>
<p>RADA-VILELA, J. <i>fuzzylite: a fuzzy logic control library (version 4.0)</i>. Disponível em: https://www.fuzzylite.com. Acesso em: 20 out. 2026.</p>
<p>ZADEH, L. A. Fuzzy sets. <i>Information and Control</i>, v. 8, n. 3, p. 338–353, 1965. DOI: 10.1016/S0019-9958(65)90241-X.</p>
<p>MAMDANI, E. H.; ASSILIAN, S. An experiment in linguistic synthesis with a fuzzy logic controller. <i>International Journal of Man-Machine Studies</i>, v. 7, n. 1, p. 1–13, 1975. DOI: 10.1016/0020-7373(75)90002-4.</p>
<p>PEARL, J. <i>Probabilistic reasoning in intelligent systems: networks of plausible inference</i>. San Mateo: Morgan Kaufmann, 1988.</p>
<p>SCIKIT-FUZZY. <i>Documentation</i>. Disponível em: https://scikit-fuzzy.readthedocs.io. Acesso em: 20 out. 2026.</p>
<p>PEDREGOSA, F. et al. Scikit-learn: machine learning in Python. <i>Journal of Machine Learning Research</i>, v. 12, p. 2825–2830, 2011.</p>
</div>
""")

# ---------------------------------------------------------------- Apêndice A
rule_items = []
for _, r in rules.iterrows():
    sentence = str(r["SE...ENTÃO"])
    rule_items.append(f'<div class="rule"><b>R{int(r["Regra"])}.</b> {e(sentence)}</div>')
parts.append(f"""
<h1 id="apendice-a" class="newpage">Apêndice A — Base de regras completa (12 regras compactas)</h1>
<p>A base abaixo é enxuta, completa e consistente: as 12 regras (4 de <b>Sucesso</b> e 8 de <b>Fracasso</b>) utilizam cláusulas parciais que particionam integralmente todas as 3<sup>4</sup> = 81 combinações do espaço de entrada NIF × RR × VT × RSBI. As mesmas regras estão exportadas nos arquivos <span class="mono">fuzzy/weaning_mamdani.fll</span>, <span class="mono">fuzzy/weaning_mamdani.fcl</span> e <span class="mono">fuzzy/weaning_mamdani.fis</span>, além de <span class="mono">tables/12_fuzzy_regras.csv</span>.</p>
<div style="font-size:8.4pt; line-height:1.5; margin-top:6pt;">
{''.join(rule_items)}
</div>
""")

# ---------------------------------------------------------------- Apêndice B
cpt_rows = []
for _, r in cpt.iterrows():
    cpt_rows.append([e(r["Rede"]), e(r["Variável"]), e(r["Classe"]), e(r["Estado"]), n(r["P(estado | classe)"], 4)])
t32 = simple_table("cpt", "Probabilidades condicionais completas das duas redes Naive Bayes, estimadas no treino com suavização de Laplace (α = 1).", ["Rede", "Variável", "Classe", "Estado", "P(estado | classe)"], cpt_rows, source="Fonte: elaboração própria. As priors estão na Tabela [[tab:priors]].", cls="tiny", numeric_from=4)
parts.append(f"""
<h1 id="apendice-b" class="newpage">Apêndice B — Probabilidades condicionais das redes</h1>
<p>As 40 linhas abaixo correspondem a todas as combinações de rede, variável, classe e estado. A soma das probabilidades de cada variável dentro de cada classe é 1. A suavização de Laplace foi aplicada com α = 1; as contagens brutas não são mostradas, mas podem ser recuperadas a partir das probabilidades e do tamanho de cada classe (12 fracassos e 9 sucessos).</p>
{t32}
""")

# ---------------------------------------------------------------- Apêndice C
val_rows = [
    ["Conferência linha a linha da base", "4 arquivos da disciplina", "0 divergências", "Respira_DB_{Treinamento,Teste}[_Process_Cases].txt idênticos aos CSVs."],
    ["Naive Bayes manual × sklearn.CategoricalNB", "Rede 1 e Rede 2", "3,3 × 10<sup>−16</sup>", "Mesmas priors suavizadas e α = 1."],
    ["Netica 7.01 GUI (Process Cases) × Python", "Redes 1 e 2 (60 casos)", "≤ 5,46 × 10<sup>−7</sup>", "Executado com CTRL_Respira.txt; 60/60 classes iguais."],
    ["Netica C API 5.04 × Python", "Redes 1 e 2 (60 casos)", "≤ 9,87 × 10<sup>−8</sup>", "Inferência cega sem alvo como evidência."],
    ["Mamdani manual × scikit-fuzzy × fuzzylite 4.0", "Base completa e casos", "&lt; 10<sup>−5</sup>", "12 regras; .fcl, .fis e .fll validados com exit code 0."],
]
t33 = simple_table("validacao", "Testes de validação numérica, gráfica e de integridade implementados no projeto.", ["Teste", "Escopo", "Erro/diferença", "Observação"], val_rows, source="Fonte: elaboração própria. Reproduzido por src/generate_artifacts.py e scripts/preparar_entrega.py.", cls="tiny keep")

tree = """weaning-ifsc/
├── data/                     # 21 treino + 9 teste + 30 completos (CSV/TSV)
├── notebooks/
│   └── Weaning_Bayes_Fuzzy.ipynb
├── src/
│   ├── weaning_lib.py        # dados, NB, Mamdani (12 regras), métricas, validações
│   ├── plotting.py           # 20 figuras analíticas
│   ├── generate_artifacts.py # tabelas, figuras, resultados.json
│   ├── export_models.py      # .dne (Netica), .fis (MATLAB/Octave), .fcl e .fll
│   ├── build_notebook.py     # constrói e executa o notebook
│   └── build_report.py       # gera este relatório HTML/PDF
├── netica/                   # redes .dne (nó Resultado), casos e evidências Netica 7.01
├── fuzzy/                    # weaning_mamdani.{fll,fcl,fis} e evidências QtFuzzyLite 4.0
├── figures/                  # 20 PNG em alta resolução
├── tables/                   # 23 CSV + Markdown
└── report/
    ├── Relatorio_Weaning_Bayes_Fuzzy.pdf
    ├── Relatorio_Weaning_Bayes_Fuzzy.html
    └── resultados.json"""
code_snip = """# Núcleo do Naive Bayes (src/weaning_lib.py)
for cls in CLASSES:
    logp[cls] = math.log(self.priors[cls]) + sum(
        math.log(self.likelihoods[var][cls][rec.loc[i, var]])
        for var in VARIABLES
    )
m = max(logp.values())
p = {c: math.exp(logp[c] - m) for c in CLASSES}
z = sum(p.values())
p = {c: p[c] / z for c in CLASSES}

# Núcleo Mamdani — 12 regras compactas (src/weaning_lib.py)
for rule in FUZZY_RULES:
    active_vars = [v for v in VARIABLES if rule["terms"].get(v) is not None]
    alpha = min(mus[v][rule["terms"][v]] for v in active_vars)
    if alpha > cutoff:
        clipped = np.minimum(alpha, mf_out)
        agg = np.maximum(agg, clipped)
area = trapezoid(agg, OUT)
momento = trapezoid(OUT * agg, OUT)
centroide = momento / area"""
parts.append(f"""
<h1 id="apendice-c" class="newpage">Apêndice C — Reprodutibilidade, validações e evidências gráficas</h1>

<h2>C.1 Estrutura de arquivos</h2>
<div class="code">{e(tree)}</div>

<h2>C.2 Como reproduzir</h2>
<p>O ambiente usado foi Python 3.12 (em ambiente virtual <span class="mono">.venv</span>), com NumPy, pandas, Matplotlib, SciPy, scikit-learn, scikit-fuzzy, statsmodels, nbformat, nbclient, nbconvert, reportlab e pypdf, além do Chromium para impressão do PDF e dos programas gráficos <b>Netica Application 7.01</b> e <b>QtFuzzyLite 4.0</b>. Os comandos abaixo regeneram tabelas, figuras, modelos exportados, notebook, relatório e ZIP:</p>
<div class="code">python3 -m venv .venv &amp;&amp; .venv/bin/pip install -r requirements.txt
./run_all.sh</div>
<p>O notebook é executado de ponta a ponta por <span class="mono">nbclient</span> e salvo com as saídas. O relatório é impresso a partir do HTML com o Chromium headless e recebe numeração de páginas por sobreposição em PDF.</p>

<h2>C.3 Validações</h2>
{t33}
<p>As verificações confrontam implementações independentes: <span class="mono">sklearn.CategoricalNB</span> usa as mesmas priors suavizadas, <span class="mono">scikit-fuzzy</span> e <span class="mono">fuzzylite 4.0</span> usam exatamente a mesma base de 12 regras compactas, e o <b>Netica 7.01</b> processa ambas as redes via <i>Cases → Process Cases</i> usando <span class="mono">CTRL_Respira.txt</span>.</p>

<div class="pagebreak"></div>
<h2>C.4 Trechos centrais do código</h2>
<div class="code">{e(code_snip)}</div>
<p>O código completo, comentado, está em <span class="mono">src/weaning_lib.py</span> e nos demais arquivos de <span class="mono">src/</span>. As redes Netica estão em <span class="mono">netica/Weaning_Rede1_Clinica.dne</span> e <span class="mono">netica/Weaning_Rede2_Netica.dne</span>; o FIS está em <span class="mono">fuzzy/weaning_mamdani.fll</span>, <span class="mono">fuzzy/weaning_mamdani.fcl</span> e <span class="mono">fuzzy/weaning_mamdani.fis</span>.</p>

<div class="box neutral">
<h4>Declaração de escopo</h4>
<p>Este é um trabalho acadêmico de modelagem sobre uma base histórica e pequena. Os resultados não constituem recomendação clínica individual e não substituem protocolos institucionais, avaliação médica ou testes de respiração espontânea. Qualquer uso em pacientes exige validação prospectiva, aprovação ética e governança clínica.</p>
</div>
""")

import json as _json
_exec_meta = _json.loads((ROOT / "netica/evidencias/execucao.json").read_text(encoding="utf-8"))
native_validation = w.validate_against_netica()
_pc_lookup = {(r["rede"], r["conjunto"]): r["erro_maximo_process_cases_python"] for r in _exec_meta.get("process_cases_gui", {}).get("validation", [])}
native_rows = [
    [
        str(int(r["Rede"])),
        e(r["Conjunto"]),
        str(int(r["Casos"])),
        f'{r["Erro máximo |P(S) Netica − Python|"]:.2e}',
        f'{_pc_lookup.get((int(r["Rede"]), str(r["Conjunto"])), 0.0):.2e}',
        str(int(r["Classes concordantes"])),
    ]
    for _, r in native_validation.iterrows()
]
native_table = simple_table(
    "netica_nativo", "Concordância entre o Netica (API nativa 5.04 e interface gráfica 7.01 via Process Cases com CTRL_Respira.txt) e a implementação Python.",
    ["Rede", "Conjunto", "Casos", "Erro máx. API C", "Erro máx. Process Cases", "Classes iguais"], native_rows,
    source="Fonte: execução real do Netica Application 7.01 (process_cases_rede*.txt) e Netica C API 5.04.",
    cls="tiny keep", note="Tolerância 10⁻⁶; o gabarito nunca é inserido como evidência durante a inferência."
)
netica_auto_fig = fig(
    "netica_auto", str(ROOT / "netica/evidencias/04_cortes_gerados.png"),
    "Etapa de discretização automática no Netica Application 7.01 (Modify → Auto-Discretize Nodes, 2 estados, 20% de arredondamento, 21 casos de treino): cortes NIF −25, RR 30, VT 350 e RSBI 90.",
    source="Fonte: captura real da execução no Netica Application 7.01 (Limited Mode).", width="82%"
)
qtfl_setup_fig = fig(
    "qtfl_setup", str(ROOT / "fuzzy/evidencias/01_qtfuzzylite_setup.png"),
    "Execução real no QtFuzzyLite 4.0: aba Setup com weaning_mamdani.fll carregado, quatro entradas (NIF, RR, VT, RSBI), saída Resultado e as 12 regras Mamdani validadas sem erros (Good Rules: 12, Bad Rules: 0).",
    source="Fonte: captura real da execução do QtFuzzyLite 4.0 (qtfuzzylite.exe).", width="88%"
)
qtfl_surf1_fig = fig(
    "qtfl_surf1", str(ROOT / "fuzzy/evidencias/02_qtfuzzylite_surface_nif_rr.png"),
    "Mapa de superfície 2D no QtFuzzyLite 4.0 (Tools → Surface 2D): NIF × RR → Resultado (VT = 450 mL, RSBI = 90 resp/min/L).",
    source="Fonte: captura real da janela Surface 2D no QtFuzzyLite 4.0.", width="48%"
)
qtfl_surf2_fig = fig(
    "qtfl_surf2", str(ROOT / "fuzzy/evidencias/03_qtfuzzylite_surface_vt_rsbi.png"),
    "Mapa de superfície 2D no QtFuzzyLite 4.0 (Tools → Surface 2D): VT × RSBI → Resultado (NIF = −30 cm.H₂O, RR = 25 resp/min).",
    source="Fonte: captura real da janela Surface 2D no QtFuzzyLite 4.0.", width="48%"
)
qtfl_case1_fig = fig(
    "qtfl_te5", str(ROOT / "fuzzy/evidencias/06_qtfuzzylite_caso1_te5_sucesso.png"),
    "Caso 1 (TE5 — Sucesso) na aba Control do QtFuzzyLite 4.0: regras R1 (0,750) e R2 (0,250) ativadas em verde e centroide Resultado = 72,732 (1,000/Sucesso).",
    source="Fonte: captura real da aba Control no QtFuzzyLite 4.0.", width="86%"
)
qtfl_case2_fig = fig(
    "qtfl_tr10", str(ROOT / "fuzzy/evidencias/07_qtfuzzylite_caso2_tr10_fracasso.png"),
    "Caso 2 (TR10 — Fracasso) na aba Control do QtFuzzyLite 4.0: regras R5, R6, R7 (1,000), R8 e R10 ativadas em verde e centroide Resultado = 25,750 (1,000/Fracasso).",
    source="Fonte: captura real da aba Control no QtFuzzyLite 4.0.", width="86%"
)
qtfl_case3_fig = fig(
    "qtfl_te3", str(ROOT / "fuzzy/evidencias/08_qtfuzzylite_caso3_te3_duvidoso.png"),
    "Caso 3 (TE3 — Crítico/duvidoso) na aba Control do QtFuzzyLite 4.0: competição entre R8 (Fracasso, 0,700) e R1 (Sucesso, 0,300), agregação bimodal e centroide Resultado = 40,048 (0,832/Fracasso + 0,168/Sucesso).",
    source="Fonte: captura real da aba Control no QtFuzzyLite 4.0.", width="86%"
)
parts.append(f"""
<div class="pagebreak"></div>
<h2>C.5 Evidência da discretização e da inferência no Netica 7.01</h2>
<p>O Netica Application 7.01 foi executado em ambiente gráfico isolado. Para a Rede 2, aplicou-se <i>Modify → Auto-Discretize Nodes</i> com dois estados e 20% de arredondamento exclusivamente sobre <span class="mono">netica/treino_discretizacao.cas</span> (21 casos de treino, sem desfecho), produzindo os cortes NIF −25, RR 30, VT 350 e RSBI 90. Em seguida, ambas as redes (<span class="mono">Weaning_Rede1_Clinica.dne</span> e <span class="mono">Weaning_Rede2_Netica.dne</span>, ambas com nó pai <b>Resultado</b>) foram processadas por <i>Cases → Process Cases</i> com <span class="mono">CTRL_Respira.txt</span> sobre <span class="mono">Respira_DB_Treinamento_Process_Cases.txt</span> e <span class="mono">Respira_DB_Teste_Process_Cases.txt</span>. A Tabela [[tab:netica_nativo]] confirma a concordância integral entre o motor Netica e o Python.</p>
{netica_auto_fig}
{native_table}
<p>Para repetir: abra <span class="mono">netica/Weaning_Rede1_Clinica.dne</span> ou <span class="mono">netica/Weaning_Rede2_Netica.dne</span> no Netica 7.01, execute <i>Network → Compile</i> e <i>Network → Automatic Updating</i>, e use <i>Cases → Process Cases</i> com <span class="mono">references/originais/CTRL_Respira.txt</span> e os arquivos de caso contínuos da disciplina. Todos os hashes e saídas estão em <span class="mono">netica/evidencias/execucao.json</span>.</p>

<div class="pagebreak"></div>
<h2>C.6 Evidência da execução gráfica e dos mapas de superfície no QtFuzzyLite 4.0</h2>
<p>Os arquivos <span class="mono">fuzzy/weaning_mamdani.fll</span>, <span class="mono">fuzzy/weaning_mamdani.fcl</span> e <span class="mono">fuzzy/weaning_mamdani.fis</span> foram validados no motor oficial <b>fuzzylite 4.0</b> (<span class="mono">fuzzylite.exe</span>) e executados na interface gráfica <b>QtFuzzyLite 4.0</b> (<span class="mono">qtfuzzylite.exe</span>), a partir da distribuição preservada em <span class="mono">references/originais/fuzzylite-4.0.zip</span>. As capturas reais abaixo (preservadas em <span class="mono">fuzzy/evidencias/</span>) documentam a validação das 12 regras na aba <i>Setup</i>, os mapas bidimensionais (<i>Tools → Surface 2D</i>) e as simulações interativas de fuzzificação, regras ativadas e defuzzificação na aba <i>Control</i> para TE5, TR10 e TE3.</p>
{qtfl_setup_fig}
{qtfl_surf1_fig}
{qtfl_surf2_fig}
{qtfl_case1_fig}
{qtfl_case2_fig}
{qtfl_case3_fig}
<p>O pacote <span class="mono">Entrega_Weaning_IFSC_Luiz_Ciarallo.zip</span> reúne o relatório, o notebook, os modelos, os dados, o código e todas as evidências gráficas do Netica 7.01 e do QtFuzzyLite 4.0. A geração desse pacote não comprova o envio por e-mail, que permanece uma etapa de encaminhamento.</p>
""")

# ---------------------------------------------------------------------------
# Montagem do HTML e exportação
# ---------------------------------------------------------------------------
def resolve_placeholders(text: str) -> str:
    import re

    # Numera tabelas e figuras pela ordem em que aparecem no documento, não
    # pela ordem em que foram construídas no script.
    for prefix, label, cls, numbers in (("tab", "Tabela", "tablabel", TABNUM), ("fig", "Figura", "figlabel", FIGNUM)):
        order = re.findall(rf'id="{prefix}-(\w+)"', text)
        numbers.clear()
        numbers.update({key: i for i, key in enumerate(order, start=1)})

        def renumber(m, numbers=numbers, label=label, cls=cls):
            return f'{m.group(1)}<span class="{cls}">{label} {numbers[m.group(2)]}</span>'

        text = re.sub(rf'(id="{prefix}-(\w+)".*?)<span class="{cls}">{label} \d+</span>', renumber, text, flags=re.S)

    # Referências plurais: "Tabelas 20–22", "Tabelas 8 e 9", "Figuras 13–14".
    def plural_range(m):
        kind, a, b = m.group(1), m.group(2), m.group(3)
        d = TABNUM if kind == "Tabelas" else FIGNUM
        return f"{kind} {d[a]}–{d[b]}"

    text = re.sub(r"(Tabelas)\s+\[\[tab:(\w+)\]\]\s*[–-]\s*\[\[tab:(\w+)\]\]", plural_range, text)
    text = re.sub(r"(Figuras)\s+\[\[fig:(\w+)\]\]\s*[–-]\s*\[\[fig:(\w+)\]\]", plural_range, text)

    def plural_and(m):
        kind, a, b = m.group(1), m.group(2), m.group(3)
        d = TABNUM if kind == "Tabelas" else FIGNUM
        return f"{kind} {d[a]} e {d[b]}"

    text = re.sub(r"(Tabelas)\s+\[\[tab:(\w+)\]\]\s+e\s+\[\[tab:(\w+)\]\]", plural_and, text)
    text = re.sub(r"(Figuras)\s+\[\[fig:(\w+)\]\]\s+e\s+\[\[fig:(\w+)\]\]", plural_and, text)

    # Referências singulares já prefixadas: "Tabela [[...]]" -> "Tabela N".
    def singular_tab(m):
        return f"Tabela {TABNUM[m.group(1)]}"

    def singular_fig(m):
        return f"Figura {FIGNUM[m.group(1)]}"

    text = re.sub(r"\bTabela\s+\[\[tab:(\w+)\]\]", singular_tab, text)
    text = re.sub(r"\bFigura\s+\[\[fig:(\w+)\]\]", singular_fig, text)

    # Placeholders restantes (sem prefixo).
    for key, num in FIGNUM.items():
        text = text.replace(f"[[fig:{key}]]", f"Figura {num}")
    for key, num in TABNUM.items():
        text = text.replace(f"[[tab:{key}]]", f"Tabela {num}")
    leftovers = [m for m in ("[[fig:", "[[tab:") if m in text]
    if leftovers:
        raise RuntimeError(f"Placeholders não resolvidos: {leftovers}")
    return text

def build_html() -> str:
    body = resolve_placeholders("\n".join(parts))
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Apoio à decisão na interrupção da ventilação mecânica — Naive Bayes e Fuzzy</title>
<meta name="author" content="Luiz Eduardo do Nascimento Ciarallo; Renato Deodoro">
<meta name="description" content="Relatório técnico-científico: redes Naive Bayes e sistema de inferência fuzzy Mamdani para desmame ventilatório.">
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""


def chrome_print(html_path: Path, pdf_path: Path):
    chrome = os.environ.get("CHROME_BIN") or shutil.which("google-chrome") or shutil.which("chromium-browser") or shutil.which("chromium")
    if chrome is None:
        candidates = sorted((Path.home() / ".cache/ms-playwright").glob("chromium-*/chrome-linux64/chrome"))
        chrome = str(candidates[-1]) if candidates else None
    if chrome is None:
        raise RuntimeError("Chromium/Google Chrome não encontrado para gerar o PDF.")
    cmd = [
        chrome,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=12000",
        "--print-to-pdf=" + str(pdf_path),
        html_path.as_uri(),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def add_page_decorations(pdf_path: Path):
    from pypdf import PdfReader, PdfWriter
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4

    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    writer = PdfWriter()
    for i, page in enumerate(reader.pages):
        wpage = writer.add_page(page)
        if i > 0:
            packet = io.BytesIO()
            c = canvas.Canvas(packet, pagesize=A4)
            width, height = A4
            # cabeçalho
            c.setFillColorRGB(0.07, 0.21, 0.36)
            c.setFont("Helvetica", 7.2)
            c.drawString(42, height - 26, "Weaning ventilatório · Naive Bayes e Mamdani")
            c.setFillColorRGB(0.32, 0.38, 0.43)
            c.setFont("Helvetica", 6.8)
            c.drawRightString(width - 42, height - 26, "IFSC · Sistemas Probabilísticos e Difusos · 2026")
            c.setStrokeColorRGB(0.82, 0.86, 0.89)
            c.setLineWidth(0.4)
            c.line(42, height - 32, width - 42, height - 32)
            # rodapé
            c.setFillColorRGB(0.32, 0.38, 0.43)
            c.setFont("Helvetica", 7.2)
            c.drawCentredString(width / 2, 26, f"p. {i + 1} de {total}")
            c.setStrokeColorRGB(0.82, 0.86, 0.89)
            c.line(42, 34, width - 42, 34)
            c.save()
            packet.seek(0)
            overlay = PdfReader(packet).pages[0]
            wpage.merge_page(overlay)
    writer.add_metadata({
        "/Title": "Apoio à decisão na interrupção da ventilação mecânica — Naive Bayes e Fuzzy",
        "/Author": "Luiz Eduardo do Nascimento Ciarallo; Renato Deodoro",
        "/Subject": "Redes Naive Bayes e sistema de inferência fuzzy Mamdani para desmame ventilatório",
        "/Keywords": "desmame ventilatório; Naive Bayes; lógica fuzzy; Mamdani; Netica; QtFuzzyLite; IFSC",
    })
    with open(pdf_path, "wb") as fh:
        writer.write(fh)


def main():
    html = build_html()
    html_path = REPORT / "Relatorio_Weaning_Bayes_Fuzzy.html"
    pdf_path = REPORT / "Relatorio_Weaning_Bayes_Fuzzy.pdf"
    html_path.write_text(html, encoding="utf-8")
    print("HTML:", html_path, f"({html_path.stat().st_size/1e6:.1f} MB)")
    chrome_print(html_path, pdf_path)
    add_page_decorations(pdf_path)
    print("PDF:", pdf_path, f"({pdf_path.stat().st_size/1e6:.1f} MB)")
    print("Figuras:", len(FIGNUM), "Tabelas:", len(TABNUM))


if __name__ == "__main__":
    main()
