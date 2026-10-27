# -*- coding: utf-8 -*-
"""
Biblioteca central do projeto de desmame ventilatório.

Contém:
  * carga e discretização das bases de treino/teste;
  * implementação auditável de Naive Bayes discreto com suavização de Laplace;
  * métricas de classificação e análise de limiar;
  * Sistema de Inferência Fuzzy Mamdani (4 entradas, 5 termos de saída,
    81 regras), com inferência manual (para inspeção) e validação via
    scikit-fuzzy.

Autor: projeto acadêmico IFSC — Sistemas Probabilísticos e Difusos.
"""
from __future__ import annotations

import itertools
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"
NETICA_DIR = ROOT / "netica"
FUZZY_DIR = ROOT / "fuzzy"

for _d in (FIG_DIR, TAB_DIR, NETICA_DIR, FUZZY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CLASSES = ("Fracasso", "Sucesso")
POSITIVE = "Sucesso"
VARIABLES = ("NIF", "RR", "VT", "RSBI")
VAR_LABELS = {
    "NIF": "Pico negativo da pressão inspiratória — NIF/PImax (cm.H₂O)",
    "RR": "Taxa respiratória — RR (resp/min)",
    "VT": "Volume tidal — VT (mL)",
    "RSBI": "Índice de respiração rápida e superficial — RSBI (resp/min/L)",
}
SHORT_LABELS = {"NIF": "NIF", "RR": "RR", "VT": "VT", "RSBI": "RSBI"}

# ---------------------------------------------------------------------------
# 1. Dados
# ---------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega treino (21) e teste (9); devolve também a base completa (30)."""
    order = ["Base", "IDnum", "NIF", "RR", "VT", "RSBI", "Resultado"]
    train = pd.read_csv(DATA_DIR / "weaning_train.csv")[order]
    test = pd.read_csv(DATA_DIR / "weaning_test.csv")[order]
    full = pd.concat([train, test], ignore_index=True)
    for df in (train, test, full):
        df["Resultado"] = pd.Categorical(df["Resultado"], CLASSES, ordered=True)
        df["ID"] = df["Base"].map({"Treino": "TR", "Teste": "TE"}) + df["IDnum"].astype(str)
    return train, test, full


def descriptive_table(full: pd.DataFrame) -> pd.DataFrame:
    """Média ± desvio-padrão e faixa por desfecho (base completa)."""
    records = []
    for var in VARIABLES:
        for cls in CLASSES:
            x = full.loc[full["Resultado"] == cls, var]
            records.append(
                {
                    "Variável": var,
                    "Desfecho": cls,
                    "n": int(x.size),
                    "Média": x.mean(),
                    "DP": x.std(ddof=1),
                    "Mínimo": x.min(),
                    "Máximo": x.max(),
                    "Média ± DP": f"{x.mean():.1f} ± {x.std(ddof=1):.1f}",
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. Discretizações
# ---------------------------------------------------------------------------
# Rede 1 — conhecimento especialista. Limites fechados/abertos explicitados.
# NIF: mais negativo = mais forte. Forte <= -30; limítrofe (-30,-20]; fraco > -20.
# RR: normal <= 20; limítrofe (20,30]; elevada > 30.
# VT: baixo < 350; adequado [350,600]; elevado > 600.
# RSBI: favorável < 105; limítrofe [105,150]; desfavorável > 150.
CLINICAL_BINS = {
    "NIF": [(-np.inf, -30, "Forte"), (-30, -20, "Limítrofe"), (-20, np.inf, "Fraco")],
    "RR": [(-np.inf, 20, "Normal"), (20, 30, "Limítrofe"), (30, np.inf, "Elevada")],
    "VT": [(-np.inf, 350, "Baixo"), (350, 600, "Adequado"), (600, np.inf, "Elevado")],
    "RSBI": [(-np.inf, 105, "Favorável"), (105, 150, "Limítrofe"), (150, np.inf, "Desfavorável")],
}

CLINICAL_STATE_ORDER = {
    "NIF": ["Forte", "Limítrofe", "Fraco"],
    "RR": ["Normal", "Limítrofe", "Elevada"],
    "VT": ["Baixo", "Adequado", "Elevado"],
    "RSBI": ["Favorável", "Limítrofe", "Desfavorável"],
}

CLINICAL_DISPLAY = {
    "NIF": {"Forte": "Forte (≤ −30)", "Limítrofe": "Limítrofe (−30 a −20)", "Fraco": "Fraco (> −20)"},
    "RR": {"Normal": "Normal (≤ 20)", "Limítrofe": "Limítrofe (20–30)", "Elevada": "Elevada (> 30)"},
    "VT": {"Baixo": "Baixo (< 350)", "Adequado": "Adequado (350–600)", "Elevado": "Elevado (> 600)"},
    "RSBI": {"Favorável": "Favorável (< 105)", "Limítrofe": "Limítrofe (105–150)", "Desfavorável": "Desfavorável (> 150)"},
}

def load_netica_levels() -> dict[str, tuple[float, float, float]]:
    """Lê os níveis salvos pelo Netica Application após Auto-Discretize Nodes.

    Não estima medianas nem contém cortes alternativos. Se a evidência não
    existir, a execução falha em vez de simular uma execução do Netica.
    """
    path = NETICA_DIR / "evidencias" / "02_discretizacao_automatica.dne"
    content = path.read_text(encoding="utf-8")
    levels = {}
    for var in VARIABLES:
        match = re.search(rf"node\s+{var}\s*\{{.*?levels\s*=\s*\(([^)]+)\)", content, re.S)
        if match is None:
            raise ValueError(f"Netica não forneceu níveis para {var}")
        values = tuple(float(x.strip()) for x in match[1].split(","))
        if len(values) != 3 or not values[0] < values[1] < values[2]:
            raise ValueError(f"Esperados dois intervalos crescentes para {var}: {values}")
        levels[var] = values
    return levels


def verify_netica_evidence() -> dict:
    """Confere hashes, 21 entradas de treino e ausência do teste no arquivo de cortes."""
    evidence = json.loads((NETICA_DIR / "evidencias" / "execucao.json").read_text())
    for filename, expected in evidence["sha256"].items():
        actual = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Evidência Netica alterada: {filename}; refaça a execução nativa")
    train = pd.read_csv(DATA_DIR / "weaning_train.csv")
    cases = pd.read_csv(NETICA_DIR / "treino_discretizacao.cas", sep="\t")
    if len(cases) != 21 or list(cases.columns) != list(VARIABLES):
        raise ValueError("Discretização deve receber apenas as quatro entradas dos 21 casos de treino")
    if not np.array_equal(cases.to_numpy(), train[list(VARIABLES)].to_numpy()):
        raise ValueError("Casos de discretização divergem do treino original")
    return evidence


NETICA_LEVELS = load_netica_levels()
NETICA_CUTS = {v: values[1] for v, values in NETICA_LEVELS.items()}


def _netica_number(value: float) -> str:
    return f"{value:g}".replace("-", "−")


NETICA_STATE_ORDER = {
    var: [f"< {_netica_number(cut)}", f"≥ {_netica_number(cut)}"]
    for var, cut in NETICA_CUTS.items()
}
NETICA_INTERVALS = {
    var: (f"[{_netica_number(low)}, {_netica_number(cut)})", f"[{_netica_number(cut)}, {_netica_number(high)}]")
    for var, (low, cut, high) in NETICA_LEVELS.items()
}


def clinical_state(var: str, x: float) -> str:
    """Mapeia valor contínuo para o estado clínico da Rede 1."""
    if var == "NIF":
        if x <= -30:
            return "Forte"
        if x <= -20:
            return "Limítrofe"
        return "Fraco"
    if var == "RR":
        if x <= 20:
            return "Normal"
        if x <= 30:
            return "Limítrofe"
        return "Elevada"
    if var == "VT":
        if x < 350:
            return "Baixo"
        if x <= 600:
            return "Adequado"
        return "Elevado"
    if var == "RSBI":
        if x < 105:
            return "Favorável"
        if x <= 150:
            return "Limítrofe"
        return "Desfavorável"
    raise KeyError(var)


def netica_state(var: str, x: float) -> str:
    """Estado da Rede 2, com intervalos semiabertos [a,b)."""
    return NETICA_STATE_ORDER[var][0 if x < NETICA_CUTS[var] else 1]


def discretize(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Aplica a discretização clínica ('clinical') ou automática ('netica')."""
    fn = clinical_state if method == "clinical" else netica_state
    out = pd.DataFrame(index=df.index)
    for var in VARIABLES:
        out[var] = df[var].map(lambda x, v=var: fn(v, x))
    return out


def audit_netica_thresholds(train: pd.DataFrame) -> pd.DataFrame:
    """Descreve no treino os cortes já produzidos pelo Netica; não os estima."""
    verify_netica_evidence()
    rows = []
    for var in VARIABLES:
        values = train[var]
        low = int((values < NETICA_CUTS[var]).sum()) if var in ("NIF", "RR", "VT", "RSBI") else 0
        high = int(len(values) - low)
        med = float(values.median())
        rows.append(
            {
                "Variável": var,
                "Corte adotado": NETICA_CUTS[var],
                "Mediana treino": med,
                "n faixa inferior": low,
                "n faixa superior": high,
                "Diferença": abs(low - high),
                "Intervalos": f"{NETICA_INTERVALS[var][0]} | {NETICA_INTERVALS[var][1]}",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Naive Bayes discreto
# ---------------------------------------------------------------------------


@dataclass
class NaiveBayesModel:
    method: str
    alpha: float
    states: dict[str, list[str]]
    priors: dict[str, float]
    likelihoods: dict[str, dict[str, dict[str, float]]]
    class_counts: dict[str, int]
    train_n: int
    n_features: int

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        rec = discretize(df, self.method)
        rows = []
        for i in df.index:
            logp = {}
            for cls in CLASSES:
                logp[cls] = math.log(self.priors[cls]) + sum(
                    math.log(self.likelihoods[var][cls][rec.loc[i, var]]) for var in VARIABLES
                )
            m = max(logp.values())
            p = {c: math.exp(logp[c] - m) for c in CLASSES}
            z = sum(p.values())
            p = {c: p[c] / z for c in CLASSES}
            pred = POSITIVE if p[POSITIVE] >= 0.5 else "Fracasso"
            rows.append(
                {
                    "p_fracasso": p["Fracasso"],
                    "p_sucesso": p["Sucesso"],
                    "predito": pred,
                    "certeza": max(p.values()),
                }
            )
        out = pd.DataFrame(rows, index=df.index)
        out["ID"] = df["ID"]
        out["Base"] = df["Base"]
        out["IDnum"] = df["IDnum"]
        out["Resultado"] = df["Resultado"].astype(str)
        return out


def fit_naive_bayes(df: pd.DataFrame, method: str, alpha: float = 1.0) -> NaiveBayesModel:
    """Ajusta Naive Bayes discreto com suavização de Laplace (alpha=1)."""
    rec = discretize(df, method)
    state_order = CLINICAL_STATE_ORDER if method == "clinical" else NETICA_STATE_ORDER
    states = {var: list(state_order[var]) for var in VARIABLES}
    # Garante que nenhum valor observado ficou fora da grade declarada.
    for var in VARIABLES:
        observed = set(rec[var].unique().tolist())
        if not observed.issubset(set(states[var])):
            raise ValueError(f"Estado não previsto para {var}: {observed - set(states[var])}")
    counts = {cls: int((df["Resultado"].astype(str) == cls).sum()) for cls in CLASSES}
    priors = {cls: (counts[cls] + alpha) / (len(df) + alpha * len(CLASSES)) for cls in CLASSES}
    likelihoods: dict[str, dict[str, dict[str, float]]] = {}
    for var in VARIABLES:
        likelihoods[var] = {}
        for cls in CLASSES:
            mask = df["Resultado"].astype(str) == cls
            n_cls = int(mask.sum())
            likelihoods[var][cls] = {
                s: (int(((rec.loc[mask, var] == s)).sum()) + alpha) / (n_cls + alpha * len(states[var]))
                for s in states[var]
            }
    return NaiveBayesModel(
        method=method,
        alpha=alpha,
        states=states,
        priors=priors,
        likelihoods=likelihoods,
        class_counts=counts,
        train_n=len(df),
        n_features=len(VARIABLES),
    )


def cpt_table(model: NaiveBayesModel) -> pd.DataFrame:
    """Tabela longa das probabilidades condicionais (para auditoria/relatório)."""
    rows = []
    for var in VARIABLES:
        for cls in CLASSES:
            for state in model.states[var]:
                rows.append(
                    {
                        "Rede": "Rede 1 (clínica)" if model.method == "clinical" else "Rede 2 (Netica)",
                        "Variável": var,
                        "Classe": cls,
                        "Estado": state,
                        "P(estado | classe)": model.likelihoods[var][cls][state],
                    }
                )
    return pd.DataFrame(rows)


def cross_validate_naive_bayes(df: pd.DataFrame, features: Iterable[str], method: str = "clinical", alpha: float = 1.0) -> dict:
    """Validação leave-one-out no TREINO, refazendo priors e CPTs em cada dobra.

    Usada apenas como análise de estabilidade/seleção de variáveis; não toca
    nos 9 casos de teste. Os cortes ficam fixos: não é uma validação aninhada
    da discretização automática.
    """
    features = tuple(features)
    y = df["Resultado"].astype(str).to_numpy()
    rec = discretize(df, method)
    preds, probs = [], []
    for i in range(len(df)):
        tr_idx = [j for j in range(len(df)) if j != i]
        prior = {c: (sum(y[tr_idx] == c) + alpha) / (len(tr_idx) + alpha * 2) for c in CLASSES}
        states = {v: sorted(rec[v].unique()) for v in features}
        lik = {
            v: {
                c: {
                    s: (sum((rec.loc[tr_idx, v].to_numpy() == s) & (y[tr_idx] == c)) + alpha)
                    / (sum(y[tr_idx] == c) + alpha * len(states[v]))
                    for s in states[v]
                }
                for c in CLASSES
            }
            for v in features
        }
        lp = {
            c: math.log(prior[c]) + sum(math.log(lik[v][c][rec.loc[i, v]]) for v in features)
            for c in CLASSES
        }
        m = max(lp.values())
        q = {c: math.exp(lp[c] - m) for c in CLASSES}
        z = sum(q.values())
        ps = q["Sucesso"] / z
        preds.append("Sucesso" if ps >= 0.5 else "Fracasso")
        probs.append(ps)
    preds = np.asarray(preds)
    probs = np.asarray(probs)
    return confusion_and_metrics(y, preds, probs)


def confusion_matrix(y_true: Iterable[str], y_pred: Iterable[str]) -> np.ndarray:
    """Matriz com linhas (real: Fracasso, Sucesso) e colunas (predito)."""
    y_true = np.asarray(list(y_true), dtype=object)
    y_pred = np.asarray(list(y_pred), dtype=object)
    return np.array(
        [
            [int(((y_true == "Fracasso") & (y_pred == "Fracasso")).sum()), int(((y_true == "Fracasso") & (y_pred == "Sucesso")).sum())],
            [int(((y_true == "Sucesso") & (y_pred == "Fracasso")).sum()), int(((y_true == "Sucesso") & (y_pred == "Sucesso")).sum())],
        ],
        dtype=int,
    )


def confusion_and_metrics(y_true: Iterable[str], y_pred: Iterable[str], p_success: Iterable[float] | None = None) -> dict:
    """Métricas com Sucesso como classe positiva (TP = sucesso previsto e real)."""
    y_true = np.asarray(list(y_true), dtype=object)
    y_pred = np.asarray(list(y_pred), dtype=object)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    eps = 1e-12
    sens = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    ppv = tp / (tp + fp + eps)
    npv = tn / (tn + fn + eps)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    out = {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "acuracia": float(acc), "sensibilidade": float(sens), "especificidade": float(spec),
        "ppv": float(ppv), "npv": float(npv), "f1": float(f1),
    }
    if p_success is not None:
        p = np.asarray(list(p_success), dtype=float)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        ybin = (y_true == "Sucesso").astype(int)
        out["brier"] = float(np.mean((p - ybin) ** 2))
        out["log_loss"] = float(-np.mean(ybin * np.log(p) + (1 - ybin) * np.log(1 - p)))
        out["certeza_media"] = float(np.mean(np.maximum(p, 1 - p)))
    return out


def metrics_row(name: str, split: str, m: dict) -> dict:
    return {
        "Modelo": name,
        "Conjunto": split,
        "Acurácia": m["acuracia"],
        "Sensibilidade": m["sensibilidade"],
        "Especificidade": m["especificidade"],
        "PPV": m["ppv"],
        "NPV": m["npv"],
        "F1": m["f1"],
        "Brier": m.get("brier", np.nan),
        "Certeza média": m.get("certeza_media", np.nan),
        "TN": m["tn"], "FP": m["fp"], "FN": m["fn"], "TP": m["tp"],
    }


def threshold_table(p_success: np.ndarray, y_true: np.ndarray, thresholds: Iterable[float]) -> pd.DataFrame:
    rows = []
    for th in thresholds:
        pred = np.where(p_success >= th, "Sucesso", "Fracasso")
        m = confusion_and_metrics(y_true, pred)
        rows.append({"Limiar": th, **m})
    return pd.DataFrame(rows)


def nb_validation_against_sklearn(train: pd.DataFrame, method: str) -> float:
    """Confere P(Sucesso|x) do código manual contra CategoricalNB (alpha=1)."""
    from sklearn.naive_bayes import CategoricalNB
    from sklearn.preprocessing import OrdinalEncoder

    rec = discretize(train, method)
    enc = OrdinalEncoder()
    X = enc.fit_transform(rec[list(VARIABLES)])
    y = (train["Resultado"].astype(str) == "Sucesso").astype(int)
    # Usa as MESMAS priors suavizadas do modelo manual; sem isso, o sklearn
    # aplicaria MLE às priors e a comparação ficaria contaminada.
    model = fit_naive_bayes(train, method)
    clf = CategoricalNB(
        alpha=1.0,
        fit_prior=False,
        class_prior=[model.priors["Fracasso"], model.priors["Sucesso"]],
    ).fit(X, y)
    p_sk = clf.predict_proba(X)[:, 1]
    p_manual = model.predict(train)["p_sucesso"].to_numpy()
    return float(np.max(np.abs(p_sk - p_manual)))


def validate_against_netica(tolerance: float = 1e-6) -> pd.DataFrame:
    """Compara probabilidades reais do motor Netica com a implementação Python."""
    verify_netica_evidence()
    train, test, _ = load_data()
    rows = []
    for number, method in ((1, "clinical"), (2, "netica")):
        model = fit_naive_bayes(train, method)
        for split, df in (("treino", train), ("teste", test)):
            native = pd.read_csv(NETICA_DIR / "evidencias" / f"rede{number}_{split}_probabilidades.csv")
            if native["IDnum"].tolist() != df["IDnum"].tolist():
                raise ValueError(f"Casos Netica fora de ordem: Rede {number}, {split}")
            python = model.predict(df)
            delta = float(np.max(np.abs(native["p_sucesso"].to_numpy() - python["p_sucesso"].to_numpy())))
            if delta > tolerance:
                raise ValueError(f"Netica/Python divergem: Rede {number}, {split}, diferença {delta}")
            native_classes = np.where(native["p_sucesso"] >= 0.5, "Sucesso", "Fracasso")
            if not np.array_equal(native_classes, python["predito"].to_numpy()):
                raise ValueError(f"Classes Netica/Python divergem: Rede {number}, {split}")
            rows.append({"Rede": number, "Conjunto": split, "Casos": len(df),
                         "Erro máximo |P(S) Netica − Python|": delta,
                         "Classes concordantes": len(df), "Tolerância": tolerance})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Sistema Fuzzy Mamdani
# ---------------------------------------------------------------------------
# Universos e funções de pertinência (parâmetros exatos no relatório).
FUZZY_RANGES = {"NIF": (-60.0, 0.0), "RR": (5.0, 60.0), "VT": (100.0, 1000.0), "RSBI": (0.0, 300.0), "OUT": (0.0, 100.0)}

# linspace garante extremos exatos; arange acumulava erro e deixava o último
# ponto do NIF em +8,5e-13 (fora do universo) e o da RR em 59,9999.
FUZZY_UNIVERSES = {
    "NIF": np.linspace(-60.0, 0.0, 601),
    "RR": np.linspace(5.0, 60.0, 551),
    "VT": np.linspace(100.0, 1000.0, 901),
    "RSBI": np.linspace(0.0, 300.0, 3001),
    "OUT": np.linspace(0.0, 100.0, 10001),
}

# (tipo, parâmetros, rótulo exibido)
FUZZY_MFS = {
    "NIF": {
        "NIF_Fraco": ("trap", [-25, -15, 0, 0], "Fraco"),
        "NIF_Limitrofe": ("tri", [-35, -25, -15], "Limítrofe"),
        "NIF_Forte": ("trap", [-60, -60, -35, -25], "Forte"),
    },
    "RR": {
        "RR_Adequada": ("trap", [5, 5, 22, 30], "Adequada"),
        "RR_Limitrofe": ("tri", [22, 30, 38], "Limítrofe"),
        "RR_Elevada": ("trap", [30, 38, 60, 60], "Elevada"),
    },
    "VT": {
        "VT_Baixo": ("trap", [100, 100, 300, 400], "Baixo"),
        "VT_Adequado": ("tri", [300, 475, 650], "Adequado"),
        "VT_Elevado": ("trap", [600, 750, 1000, 1000], "Elevado"),
    },
    "RSBI": {
        "RSBI_Favoravel": ("trap", [0, 0, 90, 120], "Favorável"),
        "RSBI_Limitrofe": ("tri", [90, 120, 150], "Limítrofe"),
        "RSBI_Desfavoravel": ("trap", [120, 150, 300, 300], "Desfavorável"),
    },
    "OUT": {
        "MB": ("trap", [0, 0, 15, 30], "Muito Baixo"),
        "B": ("tri", [15, 30, 45], "Baixo"),
        "M": ("tri", [35, 50, 65], "Moderado"),
        "A": ("tri", [55, 70, 85], "Alto"),
        "MA": ("trap", [70, 85, 100, 100], "Muito Alto"),
    },
}

FUZZY_TERMS = {
    "NIF": ["NIF_Fraco", "NIF_Limitrofe", "NIF_Forte"],
    "RR": ["RR_Adequada", "RR_Limitrofe", "RR_Elevada"],
    "VT": ["VT_Baixo", "VT_Adequado", "VT_Elevado"],
    "RSBI": ["RSBI_Favoravel", "RSBI_Limitrofe", "RSBI_Desfavoravel"],
}
FUZZY_WEIGHTS = {
    "NIF": {"NIF_Fraco": -2, "NIF_Limitrofe": 0, "NIF_Forte": 2},
    "RR": {"RR_Adequada": 2, "RR_Limitrofe": 0, "RR_Elevada": -2},
    "VT": {"VT_Baixo": -1, "VT_Adequado": 1, "VT_Elevado": 0},
    "RSBI": {"RSBI_Favoravel": 2, "RSBI_Limitrofe": 0, "RSBI_Desfavoravel": -2},
}
TERM_DISPLAY = {
    **{t: meta[2] for var in ("NIF", "RR", "VT", "RSBI") for t, meta in FUZZY_MFS[var].items()},
    **{t: meta[2] for t, meta in FUZZY_MFS["OUT"].items()},
}
TERM_DISPLAY["NIF_Limitrofe"] = "Limítrofe"
TERM_DISPLAY["RR_Limitrofe"] = "Limítrofe"
TERM_DISPLAY["RSBI_Limitrofe"] = "Limítrofe"

# Limiar operacional conservador: só recomenda liberação com potencial >= 60.
FUZZY_CUT = 60.0
FUZZY_GRAY_LOW = 40.0

# 81 regras: todas as combinações 3 x 3 x 3 x 3.
# A pontuação soma os pesos dos termos; a função abaixo converte a pontuação
# em um dos cinco consequentes. A base é funcional (uma única saída por
# combinação de antecedentes), portanto consistente.
FUZZY_RULES: list[dict] = []


def _score_to_consequent(score: int) -> str:
    if score >= 6:
        return "MA"
    if score >= 4:
        return "A"
    if score >= 2:
        return "M"
    if score >= -2:
        return "B"
    return "MB"


for _i, _combo in enumerate(itertools.product(*(FUZZY_TERMS[v] for v in VARIABLES)), start=1):
    _score = int(sum(FUZZY_WEIGHTS[v][t] for v, t in zip(VARIABLES, _combo)))
    FUZZY_RULES.append(
        {
            "id": _i,
            "terms": dict(zip(VARIABLES, _combo)),
            "score": _score,
            "consequent": _score_to_consequent(_score),
        }
    )


def fuzzy_mu(var: str, term: str, x: float) -> float:
    """Pertinência exata (avaliada na malha do universo, passo 0,01)."""
    kind, params, _ = FUZZY_MFS[var][term]
    from skfuzzy import membership as mf

    if kind == "tri":
        return float(mf.trimf(np.array([x], dtype=float), params)[0])
    return float(mf.trapmf(np.array([x], dtype=float), params)[0])


def fuzzy_memberships(x: dict[str, float]) -> dict[str, dict[str, float]]:
    return {var: {term: fuzzy_mu(var, term, x[var]) for term in FUZZY_TERMS[var]} for var in VARIABLES}


@dataclass
class FuzzyResult:
    x: dict[str, float]
    memberships: dict[str, dict[str, float]]
    activations: list[dict] = field(default_factory=list)
    aggregate: np.ndarray | None = None
    centroid: float = float("nan")
    area: float = float("nan")
    moment: float = float("nan")


def mamdani_evaluate(x: dict[str, float], cutoff: float = 1e-9) -> FuzzyResult:
    """Inferência Mamdani manual: min (E), min (implicação), max (agregação)."""
    mus = fuzzy_memberships(x)
    agg = np.zeros_like(FUZZY_UNIVERSES["OUT"], dtype=float)
    activations = []
    for rule in FUZZY_RULES:
        alpha = min(mus[v][rule["terms"][v]] for v in VARIABLES)
        if alpha > cutoff:
            out_kind, out_params, _ = FUZZY_MFS["OUT"][rule["consequent"]]
            from skfuzzy import membership as mf

            mf_out = mf.trimf(FUZZY_UNIVERSES["OUT"], out_params) if out_kind == "tri" else mf.trapmf(FUZZY_UNIVERSES["OUT"], out_params)
            clipped = np.minimum(alpha, mf_out)
            agg = np.maximum(agg, clipped)
            activations.append(
                {
                    "Regra": rule["id"],
                    "NIF": rule["terms"]["NIF"],
                    "RR": rule["terms"]["RR"],
                    "VT": rule["terms"]["VT"],
                    "RSBI": rule["terms"]["RSBI"],
                    "mu_NIF": mus["NIF"][rule["terms"]["NIF"]],
                    "mu_RR": mus["RR"][rule["terms"]["RR"]],
                    "mu_VT": mus["VT"][rule["terms"]["VT"]],
                    "mu_RSBI": mus["RSBI"][rule["terms"]["RSBI"]],
                    "alpha": alpha,
                    "Consequente": rule["consequent"],
                    "Score": rule["score"],
                }
            )
    if agg.max() <= 0:
        raise ValueError(f"Nenhuma regra ativada para {x}; base incompleta.")
    area = float(trapezoid(agg, FUZZY_UNIVERSES["OUT"]))
    moment = float(trapezoid(FUZZY_UNIVERSES["OUT"] * agg, FUZZY_UNIVERSES["OUT"]))
    return FuzzyResult(x=x, memberships=mus, activations=activations, aggregate=agg, centroid=moment / area, area=area, moment=moment)


def fuzzy_decision(value: float) -> str:
    if value >= FUZZY_CUT:
        return "Sucesso"
    if value >= FUZZY_GRAY_LOW:
        return "Cautela"
    return "Fracasso"


def fuzzy_binary(value: float) -> str:
    """Decisão binária conservadora: só libera com potencial >= 60."""
    return "Sucesso" if value >= FUZZY_CUT else "Fracasso"


def predict_fuzzy(df: pd.DataFrame) -> pd.DataFrame:
    """Avalia todos os casos na base fornecida."""
    rows = []
    for _, r in df.iterrows():
        x = {var: float(r[var]) for var in VARIABLES}
        res = mamdani_evaluate(x)
        rows.append(
            {
                "ID": r["ID"],
                "Base": r["Base"],
                "IDnum": int(r["IDnum"]),
                "NIF": r["NIF"],
                "RR": r["RR"],
                "VT": r["VT"],
                "RSBI": r["RSBI"],
                "Resultado": str(r["Resultado"]),
                "crisp": res.centroid,
                "decisao": fuzzy_decision(res.centroid),
                "binario": fuzzy_binary(res.centroid),
                "n_regras_ativadas": len(res.activations),
                "area": res.area,
                "momento": res.moment,
            }
        )
    out = pd.DataFrame(rows)
    out["acerto_conservador"] = out["binario"] == out["Resultado"]
    out["faixa"] = np.where(out["crisp"] >= FUZZY_CUT, "Sucesso (≥60)", np.where(out["crisp"] >= FUZZY_GRAY_LOW, "Cautela (40–60)", "Fracasso (<40)"))
    return out


def fuzzy_metrics(pred: pd.DataFrame, cutoff: float | None = None) -> dict:
    """Métricas binárias conservadoras; cutoff opcional para análise de limiar."""
    if cutoff is None:
        binary = pred["binario"]
    else:
        binary = np.where(pred["crisp"] >= cutoff, "Sucesso", "Fracasso")
    return confusion_and_metrics(pred["Resultado"], binary)


def fuzzy_rule_dataframe() -> pd.DataFrame:
    rows = []
    for rule in FUZZY_RULES:
        rows.append(
            {
                "Regra": rule["id"],
                **{v: TERM_DISPLAY[rule["terms"][v]] for v in VARIABLES},
                "Pontuação": rule["score"],
                "Consequente": TERM_DISPLAY[rule["consequent"]],
                "SE...ENTÃO": (
                    "SE NIF é "
                    + TERM_DISPLAY[rule["terms"]["NIF"]]
                    + " E RR é "
                    + TERM_DISPLAY[rule["terms"]["RR"]]
                    + " E VT é "
                    + TERM_DISPLAY[rule["terms"]["VT"]]
                    + " E RSBI é "
                    + TERM_DISPLAY[rule["terms"]["RSBI"]]
                    + " ENTÃO Potencial é "
                    + TERM_DISPLAY[rule["consequent"]]
                ),
            }
        )
    return pd.DataFrame(rows)


def membership_table() -> pd.DataFrame:
    rows = []
    for var in ("NIF", "RR", "VT", "RSBI", "OUT"):
        for term, (kind, params, label) in FUZZY_MFS[var].items():
            rows.append(
                {
                    "Variável": "Potencial de desmame" if var == "OUT" else var,
                    "Termo": label,
                    "Tipo": "Trapezoidal" if kind == "trap" else "Triangular",
                    "Parâmetros [a,b,c(,d)]": ", ".join(f"{p:g}" for p in params),
                    "Universo": f"[{FUZZY_RANGES[var][0]:g}, {FUZZY_RANGES[var][1]:g}]",
                }
            )
    return pd.DataFrame(rows)


def fuzzify_dataframe(res: FuzzyResult) -> pd.DataFrame:
    rows = []
    for var in VARIABLES:
        for term in FUZZY_TERMS[var]:
            rows.append(
                {
                    "Variável": var,
                    "Valor": res.x[var],
                    "Termo": TERM_DISPLAY[term],
                    "μ(x)": res.memberships[var][term],
                }
            )
    return pd.DataFrame(rows)


def defuzz_summary(res: FuzzyResult) -> dict:
    return {
        "Área agregada (∫μ dy)": res.area,
        "Momento (∫y·μ dy)": res.moment,
        "Centroide y* = momento/área": res.centroid,
        "Regras com α>0": len(res.activations),
        "Decisão conservadora": fuzzy_binary(res.centroid),
    }


def build_skfuzzy_system():
    """Constrói o mesmo FIS no scikit-fuzzy para validação independente."""
    from skfuzzy import control as ctrl
    from skfuzzy import membership as mf

    antecedents = {v: ctrl.Antecedent(FUZZY_UNIVERSES[v], v) for v in VARIABLES}
    for var in VARIABLES:
        for term, (kind, params, _) in FUZZY_MFS[var].items():
            antecedents[var][term] = mf.trimf(FUZZY_UNIVERSES[var], params) if kind == "tri" else mf.trapmf(FUZZY_UNIVERSES[var], params)
    consequent = ctrl.Consequent(FUZZY_UNIVERSES["OUT"], "Potencial")
    for term, (kind, params, _) in FUZZY_MFS["OUT"].items():
        consequent[term] = mf.trimf(FUZZY_UNIVERSES["OUT"], params) if kind == "tri" else mf.trapmf(FUZZY_UNIVERSES["OUT"], params)

    rules = []
    for rule in FUZZY_RULES:
        ant = (
            antecedents["NIF"][rule["terms"]["NIF"]]
            & antecedents["RR"][rule["terms"]["RR"]]
            & antecedents["VT"][rule["terms"]["VT"]]
            & antecedents["RSBI"][rule["terms"]["RSBI"]]
        )
        rules.append(ctrl.Rule(ant, consequent[rule["consequent"]]))
    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system, flush_after_run=200)
    return system, sim


def skfuzzy_predict(x: dict[str, float]) -> float:
    _, sim = build_skfuzzy_system()
    for var in VARIABLES:
        sim.input[var] = x[var]
    sim.compute()
    return float(sim.output["Potencial"])


def validate_fuzzy_against_skfuzzy(df: pd.DataFrame, sample_ids: Iterable[str] | None = None) -> pd.DataFrame:
    if sample_ids is None:
        sample_ids = ["TE1", "TE3", "TR10"]
    rows = []
    for id_ in sample_ids:
        r = df.loc[df["ID"] == id_].iloc[0]
        x = {v: float(r[v]) for v in VARIABLES}
        manual = mamdani_evaluate(x).centroid
        sk = skfuzzy_predict(x)
        rows.append({"Caso": id_, "Centroide manual": manual, "Centroide scikit-fuzzy": sk, "Diferença absoluta": abs(manual - sk)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Utilidades
# ---------------------------------------------------------------------------


def fmt_pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}".replace(".", ",")


def fmt_num(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def confusion_dataframe(cm: np.ndarray, model: str, split: str) -> pd.DataFrame:
    return pd.DataFrame(
        cm,
        index=pd.Index(["Real Fracasso", "Real Sucesso"], name=f"{model} — {split}"),
        columns=["Predito Fracasso", "Predito Sucesso"],
    )


def save_table(df: pd.DataFrame, name: str, also_md: bool = True) -> Path:
    path = TAB_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    if also_md:
        (TAB_DIR / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")
    return path
