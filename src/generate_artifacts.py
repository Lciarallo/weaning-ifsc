# -*- coding: utf-8 -*-
"""Gera tabelas, figuras, métricas e artefatos reprodutíveis do projeto."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import plotting
import weaning_lib as w

w.FIG_DIR.mkdir(exist_ok=True)
w.TAB_DIR.mkdir(exist_ok=True)


def pct(x):
    return 100.0 * float(x)


def build_case_table(df, pred1, pred2):
    out = df[["Base", "ID", "IDnum", "NIF", "RR", "VT", "RSBI", "Resultado"]].copy().reset_index(drop=True)
    out["Resultado"] = out["Resultado"].astype(str)
    out["Rede 1: P(Sucesso)"] = pred1["p_sucesso"].to_numpy()
    out["Rede 1: P(Fracasso)"] = pred1["p_fracasso"].to_numpy()
    out["Rede 1: predito"] = pred1["predito"].to_numpy()
    out["Rede 2: P(Sucesso)"] = pred2["p_sucesso"].to_numpy()
    out["Rede 2: P(Fracasso)"] = pred2["p_fracasso"].to_numpy()
    out["Rede 2: predito"] = pred2["predito"].to_numpy()
    out["Rede 1: acerto"] = np.where(out["Resultado"] == out["Rede 1: predito"], "Acerto", "Erro")
    out["Rede 2: acerto"] = np.where(out["Resultado"] == out["Rede 2: predito"], "Acerto", "Erro")
    return out


def main():
    train, test, full = w.load_data()
    native_validation = w.validate_against_netica()
    w.save_table(native_validation, "22_validacao_netica")

    # ------------------------------------------------------------------ NB
    models = {
        "Rede 1 (clínica)": w.fit_naive_bayes(train, "clinical"),
        "Rede 2 (Netica)": w.fit_naive_bayes(train, "netica"),
    }
    predictions = {
        "Rede 1 (clínica)": {split: models["Rede 1 (clínica)"].predict(df) for split, df in [("Treino", train), ("Teste", test), ("Completa", full)]},
        "Rede 2 (Netica)": {split: models["Rede 2 (Netica)"].predict(df) for split, df in [("Treino", train), ("Teste", test), ("Completa", full)]},
    }
    nb_rows = []
    cm_store = {}
    for model_name, model in models.items():
        for split, df in [("Treino", train), ("Teste", test), ("Completa", full)]:
            pr = predictions[model_name][split]
            m = w.confusion_and_metrics(pr["Resultado"], pr["predito"], pr["p_sucesso"])
            nb_rows.append(w.metrics_row(model_name, split, m))
            cm_store[f"cm_nb{1 if 'Rede 1' in model_name else 2}_{ {'Treino':'train','Teste':'test','Completa':'full'}[split] }"] = w.confusion_matrix(pr["Resultado"], pr["predito"])
    nb_metrics = pd.DataFrame(nb_rows)
    w.save_table(nb_metrics, "07_nb_metricas")

    nb_train = build_case_table(train, predictions["Rede 1 (clínica)"]["Treino"], predictions["Rede 2 (Netica)"]["Treino"])
    nb_test = build_case_table(test, predictions["Rede 1 (clínica)"]["Teste"], predictions["Rede 2 (Netica)"]["Teste"])
    nb_full = build_case_table(full, predictions["Rede 1 (clínica)"]["Completa"], predictions["Rede 2 (Netica)"]["Completa"])
    w.save_table(nb_train, "05_nb_treino")
    w.save_table(nb_test, "06_nb_teste")
    w.save_table(nb_full, "05b_nb_completa")

    w.save_table(w.descriptive_table(full), "01_dados_descritivos")
    w.save_table(w.audit_netica_thresholds(train), "04_auditoria_netica")

    clinical_table = []
    just = {
        ("NIF", "Forte"): "PImax ≤ −30 cm.H₂O (pressão mais negativa) é critério clássico de força muscular adequada; NIF < −30 indica maior probabilidade de sucesso.",
        ("NIF", "Limítrofe"): "Faixa de transição entre o corte de sucesso (< −30) e o de fraqueza (> −20), explicitada na base de conhecimento (Dräger).",
        ("NIF", "Fraco"): "NIF > −20 cm.H₂O (pressão menos negativa) associa-se a provável fracasso; o III Consenso aponta PImax > −30 como preditor de falha.",
        ("RR", "Normal"): "Frequência respiratória ≤ 20 resp/min situa-se na faixa de repouso; RR < 30 é critério de sucesso.",
        ("RR", "Limítrofe"): "RR entre 20 e 30 resp/min ainda é aceitável, mas próxima do limite; RR ≥ 30–35 é preditor de fracasso.",
        ("RR", "Elevada"): "RR > 30 resp/min (sobretudo ≥ 35) sugere desconforto, aumento do trabalho respiratório e maior risco de falha.",
        ("VT", "Baixo"): "VT < 350 mL aproxima-se de < 5 mL/kg (paciente de 70 kg); o consenso associa VT < 5 mL/kg a fracasso.",
        ("VT", "Adequado"): "VT entre 350 e 600 mL (≈ 5–8,5 mL/kg) situa-se na faixa fisiológica; > 315 mL é valor de referência de sucesso.",
        ("VT", "Elevado"): "VT > 600 mL é supra fisiológico em repouso; não é critério de falha, mas exige avaliação de esforço e risco de lesão.",
        ("RSBI", "Favorável"): "RSBI < 105 resp/min/L é o limiar clássico (Yang & Tobin, 1991) de maior probabilidade de sucesso.",
        ("RSBI", "Limítrofe"): "Faixa 105–150: valores pouco acima do corte têm baixo valor discriminativo e alta incerteza clínica.",
        ("RSBI", "Desfavorável"): "RSBI > 150 indica respiração rápida e superficial acentuada, com maior risco de falha e necessidade de suporte.",
    }
    for var in w.VARIABLES:
        for state in w.CLINICAL_STATE_ORDER[var]:
            lim = {
                ("NIF", "Forte"): "≤ −30 cm.H₂O",
                ("NIF", "Limítrofe"): "−30 a −20 cm.H₂O",
                ("NIF", "Fraco"): "> −20 cm.H₂O",
                ("RR", "Normal"): "≤ 20 resp/min",
                ("RR", "Limítrofe"): "20–30 resp/min",
                ("RR", "Elevada"): "> 30 resp/min",
                ("VT", "Baixo"): "< 350 mL",
                ("VT", "Adequado"): "350–600 mL",
                ("VT", "Elevado"): "> 600 mL",
                ("RSBI", "Favorável"): "< 105 resp/min/L",
                ("RSBI", "Limítrofe"): "105–150 resp/min/L",
                ("RSBI", "Desfavorável"): "> 150 resp/min/L",
            }[(var, state)]
            clinical_table.append({"Variável": var, "Estado": state, "Faixa": lim, "Justificativa clínica": just[(var, state)]})
    w.save_table(pd.DataFrame(clinical_table), "02_discretizacao_clinica")

    compare = pd.DataFrame(
        [
            {"Variável": "NIF", "Rede 1 (clínica)": "−30 e −20 cm.H₂O (3 faixas)", "Rede 2 (Netica)": "−25 cm.H₂O (2 faixas)", "Efeito": "O corte automático cai no meio da zona de incerteza clínica; perde a distinção entre força forte, limítrofe e fraca."},
            {"Variável": "RR", "Rede 1 (clínica)": "20 e 30 resp/min (3 faixas)", "Rede 2 (Netica)": "30 resp/min (2 faixas)", "Efeito": "A Rede 2 preserva o limiar superior, mas não distingue RR normal de RR limítrofe."},
            {"Variável": "VT", "Rede 1 (clínica)": "350 e 600 mL (3 faixas)", "Rede 2 (Netica)": "350 mL (2 faixas)", "Efeito": "A Rede 2 separa apenas baixo volume; perde a faixa de VT elevado."},
            {"Variável": "RSBI", "Rede 1 (clínica)": "105 e 150 resp/min/L (3 faixas)", "Rede 2 (Netica)": "90 resp/min/L (2 faixas)", "Efeito": "O corte automático é mais baixo que o clássico 105; reclassifica RSBI 90–104 como desfavorável."},
        ]
    )
    w.save_table(compare, "03_comparacao_limiares")

    # CPTs
    cpt_rows = []
    for model_name, model in models.items():
        cpt = w.cpt_table(model)
        cpt["Rede"] = model_name
        cpt_rows.append(cpt)
    cpt_all = pd.concat(cpt_rows, ignore_index=True)
    w.save_table(cpt_all, "08_nb_cpt")

    # Thresholds
    threshold_rows = []
    for model_name, model in models.items():
        for split, df in [("Teste", test), ("Completa", full)]:
            pr = predictions[model_name][split]
            t = w.threshold_table(pr["p_sucesso"].to_numpy(), pr["Resultado"].to_numpy(), np.round(np.arange(0.50, 0.91, 0.05), 2))
            t.insert(0, "Rede", model_name)
            t.insert(1, "Conjunto", split)
            threshold_rows.append(t)
    w.save_table(pd.concat(threshold_rows, ignore_index=True), "09_nb_limiares")

    # LOO / redundância
    loo_rows = []
    for method, method_name in [("clinical", "Rede 1 (clínica)"), ("netica", "Rede 2 (Netica)")]:
        for feats in [w.VARIABLES, ("NIF", "RR", "VT"), ("NIF", "RSBI"), ("NIF",)]:
            m = w.cross_validate_naive_bayes(train, feats, method=method)
            loo_rows.append({"Rede": method_name, "Variáveis": " + ".join(feats), "Acurácia LOO": m["acuracia"], "Sensibilidade LOO": m["sensibilidade"], "Especificidade LOO": m["especificidade"]})
    w.save_table(pd.DataFrame(loo_rows), "10_loo_variaveis")

    # --------------------------------------------------------------- Fuzzy
    fuzzy_pred = w.predict_fuzzy(full)
    w.save_table(fuzzy_pred, "13_fuzzy_30casos")
    w.save_table(w.membership_table(), "11_fuzzy_mfs")
    w.save_table(w.fuzzy_rule_dataframe(), "12_fuzzy_regras")

    fuzzy_metrics_rows = []
    for split, df in [("Treino", train), ("Teste", test), ("Completa", full)]:
        p = fuzzy_pred.loc[fuzzy_pred["Base"] == split] if split != "Completa" else fuzzy_pred
        m60 = w.fuzzy_metrics(p, 60)
        m50 = w.fuzzy_metrics(p, 50)
        fuzzy_metrics_rows.append({"Modelo": "Fuzzy Mamdani", "Conjunto": split, "Limiar": 60, **m60})
        fuzzy_metrics_rows.append({"Modelo": "Fuzzy Mamdani", "Conjunto": split, "Limiar": 50, **m50})
    fuzzy_metrics = pd.DataFrame(fuzzy_metrics_rows)
    w.save_table(fuzzy_metrics, "14_fuzzy_metricas")

    fsw = []
    for split, df in [("Treino", train), ("Teste", test), ("Completa", full)]:
        p = fuzzy_pred.loc[fuzzy_pred["Base"] == split] if split != "Completa" else fuzzy_pred
        for th in [40, 45, 50, 55, 60, 65, 70, 75, 80]:
            m = w.fuzzy_metrics(p, th)
            fsw.append({"Conjunto": split, "Limiar": th, **m})
    w.save_table(pd.DataFrame(fsw), "15_fuzzy_limiares")

    cm_fuzzy = w.confusion_matrix(fuzzy_pred["Resultado"], np.where(fuzzy_pred["crisp"] >= 60, "Sucesso", "Fracasso"))
    cm_fuzzy50 = w.confusion_matrix(fuzzy_pred["Resultado"], np.where(fuzzy_pred["crisp"] >= 50, "Sucesso", "Fracasso"))
    cm_store["cm_fuzzy_train"] = w.confusion_matrix(fuzzy_pred.loc[fuzzy_pred.Base == "Treino", "Resultado"], np.where(fuzzy_pred.loc[fuzzy_pred.Base == "Treino", "crisp"] >= 60, "Sucesso", "Fracasso"))
    cm_store["cm_fuzzy_test"] = w.confusion_matrix(fuzzy_pred.loc[fuzzy_pred.Base == "Teste", "Resultado"], np.where(fuzzy_pred.loc[fuzzy_pred.Base == "Teste", "crisp"] >= 60, "Sucesso", "Fracasso"))
    cm_store["cm_fuzzy_full"] = cm_fuzzy
    cm_store["cm_fuzzy_full_50"] = cm_fuzzy50

    # Estudos de caso
    case_ids = ["TE5", "TR10", "TE3"]
    fuzz_rows, rule_rows, def_rows, meta_rows = [], [], [], []
    for cid in case_ids:
        r = full.loc[full.ID == cid].iloc[0]
        x = {v: float(r[v]) for v in w.VARIABLES}
        res = w.mamdani_evaluate(x)
        fdf = w.fuzzify_dataframe(res)
        fdf.insert(0, "Caso", cid)
        fdf.insert(1, "Desfecho real", str(r["Resultado"]))
        fuzz_rows.append(fdf)
        rdf = pd.DataFrame(res.activations)
        rdf.insert(0, "Caso", cid)
        rule_rows.append(rdf)
        ds = w.defuzz_summary(res)
        ds["Caso"] = cid
        ds["Desfecho real"] = str(r["Resultado"])
        ds["Decisão (3 níveis)"] = w.fuzzy_decision(res.centroid)
        ds["NB1 P(Sucesso)"] = float(predictions["Rede 1 (clínica)"]["Completa"].loc[full.index[full.ID == cid][0], "p_sucesso"])
        ds["NB2 P(Sucesso)"] = float(predictions["Rede 2 (Netica)"]["Completa"].loc[full.index[full.ID == cid][0], "p_sucesso"])
        def_rows.append(ds)
        meta_rows.append({"Caso": cid, **x, "Desfecho real": str(r["Resultado"]), "crisp": res.centroid, "decisão": w.fuzzy_decision(res.centroid), "binário": w.fuzzy_binary(res.centroid), **ds})
    fuzz_all = pd.concat(fuzz_rows, ignore_index=True)
    rules_all = pd.concat(rule_rows, ignore_index=True)
    def_all = pd.DataFrame(def_rows)
    meta_all = pd.DataFrame(meta_rows)
    w.save_table(fuzz_all, "16_caso_fuzzificacao")
    w.save_table(rules_all, "17_caso_regras")
    w.save_table(def_all, "18_caso_defuzz")
    w.save_table(meta_all, "19_caso_meta")

    # Comparação final
    def row_from_metrics(name, split, m):
        return {"Modelo": name, "Conjunto": split, "Acurácia": m["acuracia"], "Sensibilidade": m["sensibilidade"], "Especificidade": m["especificidade"], "PPV": m["ppv"], "NPV": m["npv"], "F1": m["f1"], "TN": m["tn"], "FP": m["fp"], "FN": m["fn"], "TP": m["tp"]}

    final_metrics = pd.DataFrame(
        [
            row_from_metrics("Naive Bayes 1 (clínica)", "Completa", w.confusion_and_metrics(nb_full["Resultado"], nb_full["Rede 1: predito"], nb_full["Rede 1: P(Sucesso)"])),
            row_from_metrics("Naive Bayes 2 (Netica)", "Completa", w.confusion_and_metrics(nb_full["Resultado"], nb_full["Rede 2: predito"], nb_full["Rede 2: P(Sucesso)"])),
            row_from_metrics("Fuzzy Mamdani (≥60)", "Completa", w.fuzzy_metrics(fuzzy_pred, 60)),
        ]
    )
    w.save_table(final_metrics, "20_comparacao_final")

    # Intervalos de confiança Wilson (95%) para acurácia no teste
    from statsmodels.stats.proportion import proportion_confint
    ci_rows = []
    for name, correct in [("Rede 1 (teste)", 9), ("Rede 2 (teste)", 7), ("Fuzzy (teste, ≥60)", 9)]:
        lo, hi = proportion_confint(correct, 9, alpha=0.05, method="wilson")
        ci_rows.append({"Modelo": name, "Acertos": correct, "n": 9, "IC95% inferior": lo, "IC95% superior": hi})
    w.save_table(pd.DataFrame(ci_rows), "21_intervalos_confianca")

    # Figuras
    plotting.fig_distribuicoes(full)
    plotting.fig_topologia()
    plotting.fig_matrizes_nb({"cm_nb1_train": cm_store["cm_nb1_train"], "cm_nb1_test": cm_store["cm_nb1_test"], "cm_nb2_train": cm_store["cm_nb2_train"], "cm_nb2_test": cm_store["cm_nb2_test"]})
    plotting.fig_metricas_nb(nb_metrics[nb_metrics["Conjunto"] != "Completa"])
    plotting.fig_dumbbell_nb(predictions["Rede 1 (clínica)"]["Completa"], predictions["Rede 2 (Netica)"]["Completa"])
    plotting.fig_limiares(train, test)
    plotting.fig_memberships()
    plotting.fig_rule_consequents()
    plotting.fig_surface("nif_rr")
    plotting.fig_surface("vt_rsbi")
    plotting.fig_fuzzy_cases(fuzzy_pred)
    plotting.fig_fuzzy_confusion(fuzzy_pred)
    plotting.fig_case_studies(full)
    plotting.fig_comparacao_final(final_metrics.to_dict("records"))

    # JSON com resultados-chave para relatório/notebook
    def cm_list(cm):
        return cm.astype(int).tolist()

    summary = {
        "nb_metrics": nb_metrics.to_dict("records"),
        "fuzzy_metrics": fuzzy_metrics.to_dict("records"),
        "final_metrics": final_metrics.to_dict("records"),
        "cm": {k: cm_list(v) for k, v in cm_store.items()},
        "loo": pd.read_csv(w.TAB_DIR / "10_loo_variaveis.csv").to_dict("records"),
        "validacao_sklearn": {
            "rede1_max_abs_erro": w.nb_validation_against_sklearn(train, "clinical"),
            "rede2_max_abs_erro": w.nb_validation_against_sklearn(train, "netica"),
        },
        "validacao_netica": native_validation.to_dict("records"),
        "fuzzy_sklearn": w.validate_fuzzy_against_skfuzzy(full, ["TE1", "TE3", "TR10"]).to_dict("records"),
        "fuzzy_pred": fuzzy_pred.to_dict("records"),
        "nb_train": nb_train.to_dict("records"),
        "nb_test": nb_test.to_dict("records"),
        "casos": meta_all.to_dict("records"),
    }
    (w.ROOT / "report").mkdir(exist_ok=True)
    (w.ROOT / "report" / "resultados.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    print("Artefatos gerados.")
    print(nb_metrics.round(3).to_string(index=False))
    print("\nFuzzy:\n", fuzzy_metrics.round(3).to_string(index=False))
    print("\nFinal:\n", final_metrics.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
