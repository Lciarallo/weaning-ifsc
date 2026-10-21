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

def _states_for(model: w.NaiveBayesModel, var: str) -> list[str]:
    if model.method == "clinical":
        return [ASCII_NET1[var][s] for s in model.states[var]]
    raise ValueError("A Rede 2 deve ser salva pelo Netica nativo")


def _dne(model: w.NaiveBayesModel, title: str) -> str:
    lines = ["// ~->[DNET-1]->~", f"// Rede Naive Bayes — {title}", "// Gerada para inspeção/uso no Netica; CPTs estimadas SOMENTE com os 21 casos de treino.", ""]
    lines += [f"bnet {title} {{", "visual V1 {defdispform = BELIEFBARS;};", ""]
    # pai
    lines += [
        "node Desmame {",
        "  kind = NATURE;",
        "  discrete = TRUE;",
        "  chance = CHANCE;",
        "  states = (Fracasso, Sucesso);",
        "  parents = ();",
        f"  probs = ({model.priors['Fracasso']:.8f}, {model.priors['Sucesso']:.8f});",
        "  visual V1 {center = (300, 500);};",
        "};",
        "",
    ]
    for i, var in enumerate(w.VARIABLES):
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
        lines += [f"  visual V1 {{center = ({150 + 300 * (i % 2)}, {100 + 200 * (i // 2)});}};", "};", ""]
    lines += ["};", ""]
    return "\n".join(lines)


def export_netica(train: pd.DataFrame):
    models = {
        "Weaning_Rede1_Clinica": w.fit_naive_bayes(train, "clinical"),
    }
    # A Rede 2 é um arquivo nativo aprendido pelo Netica. Nunca sobrescrevê-lo
    # com uma exportação Python apresentada como execução do programa.
    w.verify_netica_evidence()
    for name, model in models.items():
        (w.NETICA_DIR / f"{name}.dne").write_text(_dne(model, name), encoding="utf-8")
    # Casos discretizados (ASCII) para "Process Cases".
    for method, name in [("clinical", "Rede1"), ("netica", "Rede2")]:
        rec = w.discretize(train, method).copy() if method == "clinical" else train[list(w.VARIABLES)].copy()
        rec.insert(0, "IDnum", train["IDnum"].to_numpy())
        rec["Desmame"] = train["Resultado"].astype(str).to_numpy()
        if method == "clinical":
            for var in w.VARIABLES:
                rec[var] = rec[var].map(ASCII_NET1[var])
        rec.to_csv(w.NETICA_DIR / f"casos_treino_{name}.csv", index=False)
    # Casos de teste, sem o desfecho (para inferência), e com desfecho (auditoria).
    _, test, _ = w.load_data()
    for method, name in [("clinical", "Rede1"), ("netica", "Rede2")]:
        rec = w.discretize(test, method).copy() if method == "clinical" else test[list(w.VARIABLES)].copy()
        rec.insert(0, "IDnum", test["IDnum"].to_numpy())
        if method == "clinical":
            for var in w.VARIABLES:
                rec[var] = rec[var].map(ASCII_NET1[var])
        rec.to_csv(w.NETICA_DIR / f"casos_teste_{name}_sem_desfecho.csv", index=False)
        rec["Desmame"] = test["Resultado"].astype(str).to_numpy()
        rec.to_csv(w.NETICA_DIR / f"casos_teste_{name}_com_desfecho.csv", index=False)


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
        fcl.append(f"  RANGE := ({w.FUZZY_RANGES[var][0]:g} .. {w.FUZZY_RANGES[var][1]:g});")
        for term in w.FUZZY_TERMS[var]:
            kind, params, label = w.FUZZY_MFS[var][term]
            mf = "Trapezoid" if kind == "trap" else "Triangle"
            fcl.append(f"  TERM {ascii_term(label)} := {mf} {' '.join(f'{p:g}' for p in params)};")
        fcl += ["END_FUZZIFY", ""]
    fcl += ["DEFUZZIFY Potencial", "  RANGE := (0 .. 100);"]
    for term in terms_out:
        kind, params, label = w.FUZZY_MFS["OUT"][term]
        mf = "Trapezoid" if kind == "trap" else "Triangle"
        fcl.append(f"  TERM {ascii_term(label)} := {mf} {' '.join(f'{p:g}' for p in params)};")
    fcl += ["  METHOD : COG;", "  ACCU : MAX;", "  DEFAULT := 0;", "END_DEFUZZIFY", "", "RULEBLOCK regras", "  AND : MIN;", "  ACT : MIN;"]
    for rule in w.FUZZY_RULES:
        fcl.append(
            f"  RULE {rule['id']} : if NIF is {ascii_term(w.TERM_DISPLAY[rule['terms']['NIF']])} "
            f"and RR is {ascii_term(w.TERM_DISPLAY[rule['terms']['RR']])} "
            f"and VT is {ascii_term(w.TERM_DISPLAY[rule['terms']['VT']])} "
            f"and RSBI is {ascii_term(w.TERM_DISPLAY[rule['terms']['RSBI']])} "
            f"then Potencial is {ascii_term(w.TERM_DISPLAY[rule['consequent']])};"
        )
    fcl += ["END_RULEBLOCK", "", "END_FUNCTION_BLOCK", ""]
    (w.FUZZY_DIR / "weaning_mamdani.fcl").write_text("\n".join(fcl), encoding="utf-8")


if __name__ == "__main__":
    train, test, full = w.load_data()
    export_netica(train)
    export_fis()
    print("Modelos exportados para", w.NETICA_DIR, "e", w.FUZZY_DIR)
