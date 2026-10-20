# -*- coding: utf-8 -*-
"""Exporta as redes para Netica (.dne) e o FIS para MATLAB/Octave (.fis) e FCL."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import weaning_lib as w

ASCII_NET1 = {
    "NIF": {"Forte": "Forte", "Limítrofe": "Limitrofe", "Fraco": "Fraco"},
    "RR": {"Normal": "Normal", "Limítrofe": "Limitrofe", "Elevada": "Elevada"},
    "VT": {"Baixo": "Baixo", "Adequado": "Adequado", "Elevado": "Elevado"},
    "RSBI": {"Favorável": "Favoravel", "Limítrofe": "Limitrofe", "Desfavorável": "Desfavoravel"},
}
NET2_ASCII = {
    "NIF": {"≤ −25": "ate_-25", "> −25": "maior_-25"},
    "RR": {"< 30": "ate_30", "≥ 30": "maior_30"},
    "VT": {"< 350": "ate_350", "≥ 350": "maior_350"},
    "RSBI": {"< 90": "ate_90", "≥ 90": "maior_90"},
}


def _states_for(model: w.NaiveBayesModel, var: str) -> list[str]:
    if model.method == "clinical":
        return [ASCII_NET1[var][s] for s in model.states[var]]
    return [NET2_ASCII[var][s] for s in model.states[var]]


def _dne(model: w.NaiveBayesModel, title: str) -> str:
    lines = ["// ~->[DNET-1]->~", f"// Rede Naive Bayes — {title}", "// Gerada para inspeção/uso no Netica; CPTs estimadas SOMENTE com os 21 casos de treino.", ""]
    lines += [f"bnet {title} {{", ""]
    # pai
    lines += [
        "node Desmame {",
        "  kind = NATURE;",
        "  discrete = TRUE;",
        "  chance = CHANCE;",
        "  states = (Fracasso, Sucesso);",
        "  parents = ();",
        f"  probs = ({model.priors['Fracasso']:.8f}, {model.priors['Sucesso']:.8f});",
        "};",
        "",
    ]
    for var in w.VARIABLES:
        states = _states_for(model, var)
        lines += [
            f"node {var} {{",
            "  kind = NATURE;",
            "  discrete = TRUE;",
            "  chance = CHANCE;",
            f"  states = ({', '.join(states)});",
            "  parents = (Desmame);",
            "  probs =",
        ]
        p_f = [model.likelihoods[var]["Fracasso"][s] for s in model.states[var]]
        p_s = [model.likelihoods[var]["Sucesso"][s] for s in model.states[var]]
        lines.append("    ((" + ", ".join(f"{x:.8f}" for x in p_f) + "),   // Desmame = Fracasso")
        lines.append("     (" + ", ".join(f"{x:.8f}" for x in p_s) + "));  // Desmame = Sucesso")
        lines += ["};", ""]
    lines += ["}", ""]
    return "\n".join(lines)


def export_netica(train: pd.DataFrame):
    models = {
        "Weaning_Rede1_Clinica": w.fit_naive_bayes(train, "clinical"),
        "Weaning_Rede2_Netica": w.fit_naive_bayes(train, "netica"),
    }
    for name, model in models.items():
        (w.NETICA_DIR / f"{name}.dne").write_text(_dne(model, name), encoding="utf-8")
    # Casos discretizados (ASCII) para "Process Cases".
    for method, name in [("clinical", "Rede1"), ("netica", "Rede2")]:
        model = models["Weaning_Rede1_Clinica" if method == "clinical" else "Weaning_Rede2_Netica"]
        rec = w.discretize(train, method).copy()
        rec.insert(0, "IDnum", train["IDnum"].to_numpy())
        rec["Desmame"] = train["Resultado"].astype(str).to_numpy()
        for var in w.VARIABLES:
            rec[var] = rec[var].map(lambda s, v=var: ASCII_NET1[v][s] if method == "clinical" else NET2_ASCII[v][s])
        rec.to_csv(w.NETICA_DIR / f"casos_treino_{name}.csv", index=False)
    # Casos de teste, sem o desfecho (para inferência), e com desfecho (auditoria).
    _, test, _ = w.load_data()
    for method, name in [("clinical", "Rede1"), ("netica", "Rede2")]:
        rec = w.discretize(test, method).copy()
        rec.insert(0, "IDnum", test["IDnum"].to_numpy())
        for var in w.VARIABLES:
            rec[var] = rec[var].map(lambda s, v=var: ASCII_NET1[v][s] if method == "clinical" else NET2_ASCII[v][s])
        rec.to_csv(w.NETICA_DIR / f"casos_teste_{name}_sem_desfecho.csv", index=False)
        rec["Desmame"] = test["Resultado"].astype(str).to_numpy()
        rec.to_csv(w.NETICA_DIR / f"casos_teste_{name}_com_desfecho.csv", index=False)
    (w.NETICA_DIR / "README_NETICA.md").write_text(
        """# Uso dos modelos no Netica

Arquivos:

- `Weaning_Rede1_Clinica.dne`: rede com discretização por conhecimento clínico (3 faixas por variável).
- `Weaning_Rede2_Netica.dne`: rede com dois intervalos automáticos por variável, reproduzindo a execução de referência do Netica.
- `casos_treino_Rede1.csv` e `casos_treino_Rede2.csv`: os 21 casos já discretizados (rótulos ASCII), com a coluna `Desmame` (gabarito).
- `casos_teste_*_sem_desfecho.csv`: 9 casos para `Cases > Process Cases` (inferência cega).
- `casos_teste_*_com_desfecho.csv`: 9 casos para `Cases > Test With Cases` (auditoria).

As CPTs e as priors já estão gravadas nos arquivos `.dne`, estimadas exclusivamente
com os 21 casos de treino e suavização de Laplace (alpha = 1). Para reproduzir:

1. Abra o `.dne` no Netica (File > Open).
2. Compile a rede (Network > Compile).
3. Para inferência: Cases > Process Cases > escolha o CSV sem desfecho.
4. Para métricas: Cases > Test With Cases > escolha o CSV com desfecho.
5. Confira as crenças em `Desmame` no painel de nós.

Os cortes da Rede 2 são `[−60,−25) | [−25,−10]`, `[13,30) | [30,46]`,
`[120,350) | [350,950]` e `[10,90) | [90,270]`. Eles foram obtidos apenas
com o treino (medianas/medianas arredondadas; partições 10/11 ou 11/10).
Nenhum valor da base de teste foi usado para defini-los.

A inspeção do `.dne` é opcional: toda a inferência do relatório foi replicada
em Python (`src/weaning_lib.py`) e validada contra `sklearn.CategoricalNB`.
""",
        encoding="utf-8",
    )


def export_fis():
    # .fis para MATLAB/Octave
    lines = [
        "[System]",
        "Name='Weaning_Mamdani'",
        "Type='mamdani'",
        "Version=2.0",
        "NumInputs=4",
        "NumOutputs=1",
        "NumRules=81",
        "AndMethod='min'",
        "OrMethod='max'",
        "ImpMethod='min'",
        "AggMethod='max'",
        "DefuzzMethod='centroid'",
        "",
    ]
    for k, var in enumerate(w.VARIABLES, start=1):
        terms = w.FUZZY_TERMS[var]
        lines += [f"[Input{k}]", f"Name='{var}'", f"Range=[{w.FUZZY_RANGES[var][0]:g} {w.FUZZY_RANGES[var][1]:g}]", f"NumMFs={len(terms)}"]
        for j, term in enumerate(terms, start=1):
            kind, params, label = w.FUZZY_MFS[var][term]
            mf = "trapmf" if kind == "trap" else "trimf"
            lines.append(f"MF{j}='{label.replace('í','i').replace('á','a').replace('é','e').replace('ó','o').replace('ú','u').replace('â','a').replace('ê','e').replace('ô','o').replace('ç','c')}':'{mf}',[{ ' '.join(f'{p:g}' for p in params)}]")
        lines.append("")
    terms_out = ["MB", "B", "M", "A", "MA"]
    lines += ["[Output1]", "Name='Potencial'", "Range=[0 100]", f"NumMFs={len(terms_out)}"]
    for j, term in enumerate(terms_out, start=1):
        kind, params, label = w.FUZZY_MFS["OUT"][term]
        mf = "trapmf" if kind == "trap" else "trimf"
        lines.append(f"MF{j}='{label}':'{mf}',[{' '.join(f'{p:g}' for p in params)}]")
    lines += ["", "[Rules]"]
    idx = {
        "NIF": {t: i + 1 for i, t in enumerate(w.FUZZY_TERMS["NIF"])},
        "RR": {t: i + 1 for i, t in enumerate(w.FUZZY_TERMS["RR"])},
        "VT": {t: i + 1 for i, t in enumerate(w.FUZZY_TERMS["VT"])},
        "RSBI": {t: i + 1 for i, t in enumerate(w.FUZZY_TERMS["RSBI"])},
        "OUT": {t: i + 1 for i, t in enumerate(terms_out)},
    }
    for rule in w.FUZZY_RULES:
        ant = [idx["NIF"][rule["terms"]["NIF"]], idx["RR"][rule["terms"]["RR"]], idx["VT"][rule["terms"]["VT"]], idx["RSBI"][rule["terms"]["RSBI"]]]
        lines.append(" ".join(str(x) for x in ant) + f", {idx['OUT'][rule['consequent']]} (1) : 1")
    (w.FUZZY_DIR / "weaning_mamdani.fis").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # .fcl para fuzzylite 4.0
    fcl = ["FUNCTION_BLOCK weaning_mamdani", "", "VAR_INPUT"]
    for var in w.VARIABLES:
        fcl.append(f"  {var} : REAL;")
    fcl += ["END_VAR", "", "VAR_OUTPUT", "  Potencial : REAL;", "END_VAR", ""]
    ascii_term = lambda s: (s.replace("í", "i").replace("á", "a").replace("é", "e").replace("ó", "o").replace("ú", "u").replace("â", "a").replace("ê", "e").replace("ô", "o").replace("ç", "c").replace(" ", "_"))
    for var in w.VARIABLES:
        fcl.append(f"FUZZIFY {var}")
        for term in w.FUZZY_TERMS[var]:
            kind, params, label = w.FUZZY_MFS[var][term]
            if kind == "trap":
                fcl.append(f"  TERM {ascii_term(label)} := trapmf ({', '.join(f'{p:g}' for p in params)});")
            else:
                fcl.append(f"  TERM {ascii_term(label)} := trimf ({', '.join(f'{p:g}' for p in params)});")
        fcl += ["END_FUZZIFY", ""]
    fcl += ["DEFUZZIFY Potencial"]
    for term in terms_out:
        kind, params, label = w.FUZZY_MFS["OUT"][term]
        if kind == "trap":
            fcl.append(f"  TERM {ascii_term(label)} := trapmf ({', '.join(f'{p:g}' for p in params)});")
        else:
            fcl.append(f"  TERM {ascii_term(label)} := trimf ({', '.join(f'{p:g}' for p in params)});")
    fcl += ["  METHOD : COG;", "  DEFAULT := 0;", "END_DEFUZZIFY", "", "RULEBLOCK regras", "  AND : MIN;", "  ACT : MIN;", "  ACCU : MAX;"]
    for rule in w.FUZZY_RULES:
        fcl.append(
            f"  RULE {rule['id']} : IF NIF IS {ascii_term(w.TERM_DISPLAY[rule['terms']['NIF']])} "
            f"AND RR IS {ascii_term(w.TERM_DISPLAY[rule['terms']['RR']])} "
            f"AND VT IS {ascii_term(w.TERM_DISPLAY[rule['terms']['VT']])} "
            f"AND RSBI IS {ascii_term(w.TERM_DISPLAY[rule['terms']['RSBI']])} "
            f"THEN Potencial IS {ascii_term(w.TERM_DISPLAY[rule['consequent']])};"
        )
    fcl += ["END_RULEBLOCK", "", "END_FUNCTION_BLOCK", ""]
    (w.FUZZY_DIR / "weaning_mamdani.fcl").write_text("\n".join(fcl), encoding="utf-8")


if __name__ == "__main__":
    train, test, full = w.load_data()
    export_netica(train)
    export_fis()
    print("Modelos exportados para", w.NETICA_DIR, "e", w.FUZZY_DIR)
