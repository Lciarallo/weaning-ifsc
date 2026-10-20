# -*- coding: utf-8 -*-
"""Figuras do projeto, em estilo consistente e prontas para o relatório."""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

import weaning_lib as w

mpl.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 210,
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.edgecolor": "#39434D",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    }
)

C = {
    "Fracasso": "#B23A48",
    "Sucesso": "#1F6F8B",
    "Rede 1": "#12355B",
    "Rede 2": "#D98E04",
    "Fuzzy": "#2A9D8F",
    "gray": "#6C757D",
    "light": "#E9EEF3",
    "dark": "#172A3A",
}
TERM_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
OUT_COLORS = {"MB": "#8E2C3B", "B": "#C65D4B", "M": "#D9A441", "A": "#6A9E75", "MA": "#2F6E64"}
VAR_TITLES = {
    "NIF": "NIF/PImax (cm.H₂O)",
    "RR": "RR (resp/min)",
    "VT": "VT (mL)",
    "RSBI": "RSBI (resp/min/L)",
}
SRC = "Fonte: elaboração própria com os dados do trabalho (Ashutosh et al., 1992)."


def _save(fig, name: str):
    path = w.FIG_DIR / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def fig_distribuicoes(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.6))
    cuts = {
        "NIF": [(-30, "−30"), (-20, "−20")],
        "RR": [(30, "30"), (20, "20")],
        "VT": [(350, "350"), (600, "600")],
        "RSBI": [(105, "105"), (150, "150")],
    }
    rng = np.random.default_rng(7)
    for ax, var in zip(axes.ravel(), w.VARIABLES):
        data = [df.loc[df.Resultado == cls, var].to_numpy() for cls in w.CLASSES]
        bp = ax.boxplot(
            data,
            positions=[1, 2],
            widths=0.38,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="white", lw=2),
            whiskerprops=dict(color="#39434D"),
            capprops=dict(color="#39434D"),
        )
        for patch, cls in zip(bp["boxes"], w.CLASSES):
            patch.set_facecolor(C[cls])
            patch.set_alpha(0.78)
        for i, (vals, cls) in enumerate(zip(data, w.CLASSES), start=1):
            jitter = rng.uniform(-0.11, 0.11, size=len(vals))
            ax.scatter(i + jitter, vals, s=22, color=C[cls], edgecolor="white", linewidth=0.5, zorder=3)
        for k, (cut, lab) in enumerate(cuts[var]):
            ax.axhline(cut, color="#495057", lw=1.0, ls="--" if k == 0 else ":")
            ax.text(0.985, cut, lab, transform=ax.get_yaxis_transform(), va="bottom", ha="right", fontsize=7.5, color="#495057", clip_on=True)
        ax.set_xticks([1, 2], ["Fracasso", "Sucesso"])
        ax.set_title(VAR_TITLES[var])
        ax.set_xlim(0.55, 2.45)
        ax.margins(y=0.10)
    fig.suptitle("Distribuição das variáveis fisiológicas por desfecho (n = 30)", fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    return _save(fig, "fig01_distribuicoes.png")


def _node(ax, x, y, wdt, hgt, title, lines, face="#FFFFFF", edge="#12355B", title_color="#12355B", fs=8.2):
    box = FancyBboxPatch(
        (x - wdt / 2, y - hgt / 2), wdt, hgt,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.1, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x, y + hgt * 0.24, title, ha="center", va="center", fontsize=fs + 1.0, fontweight="bold", color=title_color, zorder=3)
    if lines:
        ax.text(x, y - hgt * 0.13, "\n".join(lines), ha="center", va="center", fontsize=fs - 0.7, color="#263238", zorder=3, linespacing=1.25)


def fig_topologia():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
    specs = [
        (
            "Rede 1 — discretização por conhecimento clínico",
            {
                "NIF": ["Forte: ≤ −30", "Limítrofe: −30 a −20", "Fraco: > −20"],
                "RR": ["Normal: ≤ 20", "Limítrofe: 20–30", "Elevada: > 30"],
                "VT": ["Baixo: < 350", "Adequado: 350–600", "Elevado: > 600"],
                "RSBI": ["Favorável: < 105", "Limítrofe: 105–150", "Desfavorável: > 150"],
            },
            C["Rede 1"],
        ),
        (
            "Rede 2 — discretização automática (Netica, 2 faixas)",
            {
                "NIF": ["[−60, −25)", "[−25, −10]"],
                "RR": ["[13, 30)", "[30, 46]"],
                "VT": ["[120, 350)", "[350, 950]"],
                "RSBI": ["[10, 90)", "[90, 270]"],
            },
            C["Rede 2"],
        ),
    ]
    for ax, (title, states, color) in zip(axes, specs):
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-0.45, 1.25)
        ax.axis("off")
        ax.set_title(title, pad=12, color=color)
        parent = (0, 1.03)
        _node(ax, *parent, 0.92, 0.18, "Desmame", ["P(Sucesso) = 43,5%", "P(Fracasso) = 56,5%"], face="#F7FAFC", edge=color, title_color=color, fs=8.8)
        xs = [-1.28, -0.43, 0.43, 1.28]
        ychild = 0.10
        heights = 0.29 if len(next(iter(states.values()))) == 3 else 0.25
        for var, x in zip(w.VARIABLES, xs):
            ax.annotate(
                "",
                xy=(x, ychild + 0.12),
                xytext=(parent[0], parent[1] - 0.10),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0.0"),
            )
            _node(ax, x, ychild, 0.80, heights, var, states[var], face="#FFFFFF", edge=color, title_color=color, fs=7.2)
        ax.text(0.0, 0.62, "P(Desmame | evidências) ∝ P(Desmame) · Π P(filho | Desmame)", ha="center", fontsize=7.7,
                color="#37474F", bbox=dict(boxstyle="round,pad=0.25", fc="#F4F6F8", ec="#CFD8DC", lw=0.6))
        ax.text(0.0, -0.37, "Suavização de Laplace (α = 1) aplicada a todas as CPTs", ha="center", fontsize=7.4, color="#546E7A", style="italic")
    fig.suptitle("Topologia Naive Bayes e grades de discretização", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    return _save(fig, "fig02_topologia_nb.png")


def fig_matrizes_nb(metrics):
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2))
    configs = [
        ("Rede 1 — Treino", metrics["cm_nb1_train"], C["Rede 1"]),
        ("Rede 1 — Teste", metrics["cm_nb1_test"], C["Rede 1"]),
        ("Rede 2 — Treino", metrics["cm_nb2_train"], C["Rede 2"]),
        ("Rede 2 — Teste", metrics["cm_nb2_test"], C["Rede 2"]),
    ]
    for ax, (title, cm, color) in zip(axes.ravel(), configs):
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(6, cm.max()))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() * 0.55 else color)
        ax.set_xticks([0, 1], ["Pred. Fracasso", "Pred. Sucesso"])
        ax.set_yticks([0, 1], ["Real Fracasso", "Real Sucesso"])
        ax.set_title(title, color=color)
        ax.grid(False)
        ax.set_xlabel("Classe predita")
        ax.set_ylabel("Classe real")
    fig.suptitle("Matrizes de confusão das redes Naive Bayes (limiar 50%)", fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    return _save(fig, "fig03_matrizes_confusao_nb.png")


def fig_metricas_nb(metrics_df):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), gridspec_kw={"width_ratios": [1.45, 1]})
    df = metrics_df.copy()
    labels = [("Rede 1" if "Rede 1" in m else "Rede 2") + "\n" + s for m, s in zip(df["Modelo"], df["Conjunto"])]
    x = np.arange(len(df))
    width = 0.25
    for k, (col, color) in enumerate(zip(["Acurácia", "Sensibilidade", "Especificidade"], ["#12355B", "#2A9D8F", "#D98E04"])):
        bars = axes[0].bar(x + (k - 1) * width, df[col].to_numpy(), width, label=col, color=color, alpha=0.9, edgecolor="white")
        for b in bars:
            axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.018, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=6.6, color="#37474F", rotation=90)
    axes[0].set_xticks(x, labels, fontsize=8)
    axes[0].set_ylim(0, 1.20)
    axes[0].set_ylabel("Valor (0–1)")
    axes[0].axhline(1.0, color="#B0BEC5", lw=0.7)
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    axes[0].set_title("Métricas de classificação")
    x2 = np.arange(len(df))
    axes[1].bar(x2 - 0.18, df["Brier"], 0.34, color="#B23A48", label="Brier", edgecolor="white")
    axes[1].bar(x2 + 0.18, df["Certeza média"], 0.34, color="#6C757D", label="Certeza média", edgecolor="white")
    axes[1].set_xticks(x2, labels, fontsize=8)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="upper right")
    axes[1].set_title("Calibração e confiança")
    axes[1].set_ylabel("Valor")
    fig.suptitle("Desempenho das redes Naive Bayes", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    return _save(fig, "fig04_metricas_nb.png")


def fig_dumbbell_nb(pred1: pd.DataFrame, pred2: pd.DataFrame, top_n: int = 16):
    d = pred1[["ID", "Resultado", "p_sucesso"]].rename(columns={"p_sucesso": "Rede 1"}).merge(
        pred2[["ID", "p_sucesso"]].rename(columns={"p_sucesso": "Rede 2"}), on="ID"
    )
    d["delta"] = (d["Rede 1"] - d["Rede 2"]).abs()
    d = d.sort_values("delta", ascending=False).head(top_n).sort_values("Rede 1")
    fig, ax = plt.subplots(figsize=(8.6, 0.37 * len(d) + 1.7))
    y = np.arange(len(d))
    for i, (_, row) in enumerate(d.iterrows()):
        ax.plot([row["Rede 1"], row["Rede 2"]], [i, i], color="#B0BEC5", lw=2.2, zorder=1)
    ax.scatter(d["Rede 1"], y, s=55, color=C["Rede 1"], label="Rede 1 (clínica)", zorder=3, edgecolor="white", linewidth=0.7)
    ax.scatter(d["Rede 2"], y, s=55, color=C["Rede 2"], label="Rede 2 (Netica)", zorder=3, edgecolor="white", linewidth=0.7)
    ax.axvline(0.5, color="#455A64", ls="--", lw=1)
    ax.axvline(0.6, color="#455A64", ls=":", lw=1)
    ax.set_yticks(y, [f"{r.ID}  ({r.Resultado})" for r in d.itertuples()], fontsize=7.6)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(1.01, i, f"Δ={r['delta']:.2f}", va="center", fontsize=6.7, color="#546E7A")
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("P(Sucesso | evidências)")
    ax.set_title(f"Casos em que as redes mais divergem (top {top_n} por |Δ|)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False)
    ax.text(0.5, -0.16, "Linha tracejada: limiar 50%; linha pontilhada: limiar conservador 60%.", transform=ax.transAxes, ha="center", fontsize=7, color="#555")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, "fig05_divergencia_probabilidades.png")


def fig_limiares(train, test):
    fig, axes = plt.subplots(4, 1, figsize=(11.3, 6.0), sharex=False)
    rng = np.random.default_rng(2)
    clinical = {
        "NIF": [-30, -20],
        "RR": [20, 30],
        "VT": [350, 600],
        "RSBI": [105, 150],
    }
    for ax, var in zip(axes, w.VARIABLES):
        lo = min(train[var].min(), test[var].min())
        hi = max(train[var].max(), test[var].max())
        pad = 0.05 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.55, 0.55)
        ax.scatter(train[var], rng.uniform(-0.13, 0.13, len(train)), s=24, color="#263238", alpha=0.75, label="Treino (n=21)", zorder=3)
        ax.scatter(test[var], rng.uniform(-0.38, -0.23, len(test)), s=28, marker="^", color="#78909C", alpha=0.9, label="Teste (n=9)", zorder=3)
        for cut in clinical[var]:
            ax.axvline(cut, color=C["Rede 1"], lw=1.5, ls="--")
        ax.axvline(w.NETICA_CUTS[var], color=C["Rede 2"], lw=1.8, ls="-")
        ax.set_yticks([])
        ax.set_ylabel(var, rotation=0, ha="right", va="center", fontweight="bold")
        ax.grid(axis="y", visible=False)
        ax.set_title("", loc="left")
        # rótulos dos cortes
        ytext = 0.39
        for cut in clinical[var]:
            ax.text(cut, ytext, f"clínico {cut:g}", color=C["Rede 1"], fontsize=7.2, ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
        ax.text(w.NETICA_CUTS[var], -0.49, f"Netica {w.NETICA_CUTS[var]:g}", color=C["Rede 2"], fontsize=7.2, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
        ax.set_xlabel(VAR_TITLES[var], fontsize=8.2)
    handles = [
        Line2D([], [], color="#263238", marker="o", ls="", ms=5, label="Treino (n=21)"),
        Line2D([], [], color="#78909C", marker="^", ls="", ms=5, label="Teste (n=9)"),
        Line2D([], [], color=C["Rede 1"], ls="--", lw=1.5, label="Cortes clínicos (Rede 1)"),
        Line2D([], [], color=C["Rede 2"], ls="-", lw=1.8, label="Corte automático (Rede 2)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Cortes de discretização: conhecimento clínico versus Netica", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    return _save(fig, "fig06_limiares.png")


def _mf_array(var: str, term: str):
    kind, params, _ = w.FUZZY_MFS[var][term]
    from skfuzzy import membership as mf

    x = w.FUZZY_UNIVERSES[var]
    return mf.trimf(x, params) if kind == "tri" else mf.trapmf(x, params)


def fig_memberships():
    fig = plt.figure(figsize=(11.8, 8.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1], hspace=0.53, wspace=0.22)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax, var in zip(axes, w.VARIABLES):
        x = w.FUZZY_UNIVERSES[var]
        for k, term in enumerate(w.FUZZY_TERMS[var]):
            y = _mf_array(var, term)
            ax.plot(x, y, color=TERM_COLORS[k], lw=2.0, label=w.TERM_DISPLAY[term])
            ax.fill_between(x, 0, y, color=TERM_COLORS[k], alpha=0.12)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(-0.03, 1.08)
        ax.set_title(VAR_TITLES[var], loc="left")
        ax.set_ylabel("μ(x)")
        ax.legend(loc="upper right", ncol=3, fontsize=7.2)
        # marcas clínicas
        marks = {"NIF": [-30, -20], "RR": [30], "VT": [350], "RSBI": [105, 150]}[var]
        for m in marks:
            ax.axvline(m, color="#546E7A", lw=0.9, ls=":")
    axo = fig.add_subplot(gs[2, :])
    x = w.FUZZY_UNIVERSES["OUT"]
    for term in ["MB", "B", "M", "A", "MA"]:
        y = _mf_array("OUT", term)
        axo.plot(x, y, color=OUT_COLORS[term], lw=2.3, label=w.TERM_DISPLAY[term])
        axo.fill_between(x, 0, y, color=OUT_COLORS[term], alpha=0.12)
    axo.axvline(w.FUZZY_CUT, color="#263238", lw=1.2, ls="--")
    axo.annotate("limiar conservador = 60", xy=(w.FUZZY_CUT, 0.45), xytext=(w.FUZZY_CUT + 8, 0.76),
                 fontsize=8, color="#263238", arrowprops=dict(arrowstyle="-", color="#263238", lw=0.8))
    axo.set_xlim(0, 100)
    axo.set_ylim(-0.03, 1.10)
    axo.set_title("Saída — Potencial de desmame (0–100)", loc="left")
    axo.set_xlabel("Potencial de desmame")
    axo.set_ylabel("μ(y)")
    axo.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=8.0)
    fig.suptitle("Funções de pertinência do sistema Mamdani", fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005, "Trapezoides nas extremidades (saturação fisiológica) e triângulos nas transições; sobreposições representam a incerteza clínica.", ha="center", fontsize=7.8, color="#555")
    return _save(fig, "fig07_funcoes_pertinencia.png")


def fig_rule_consequents():
    rules = w.fuzzy_rule_dataframe()
    counts = rules["Consequente"].value_counts().reindex(["Muito Baixo", "Baixo", "Moderado", "Alto", "Muito Alto"]).fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    colors = [OUT_COLORS[k] for k in ["MB", "B", "M", "A", "MA"]]
    axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 0.5, f"{int(v)}", ha="center", fontweight="bold", color="#263238")
    axes[0].set_ylim(0, max(counts.values) + 5)
    axes[0].set_title("Distribuição dos 81 consequentes")
    axes[0].set_ylabel("Número de regras")
    axes[0].tick_params(axis="x", rotation=15)
    # heatmap NIF x RR para VT adequado e RSBI favorável
    sub = rules[(rules["VT"] == "Adequado") & (rules["RSBI"] == "Favorável")]
    pivot = sub.pivot_table(index="NIF", columns="RR", values="Pontuação", aggfunc="first")
    order_nif = ["Fraco", "Limítrofe", "Forte"]
    order_rr = ["Adequada", "Limítrofe", "Elevada"]
    pivot = pivot.reindex(index=order_nif, columns=order_rr)
    im = axes[1].imshow(pivot.values, cmap="RdYlGn", vmin=-3, vmax=7)
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{pivot.values[i, j]:+.0f}", ha="center", va="center", fontsize=10, fontweight="bold", color="#263238")
    axes[1].set_xticks(range(3), order_rr)
    axes[1].set_yticks(range(3), order_nif)
    axes[1].set_xlabel("RR")
    axes[1].set_ylabel("NIF")
    axes[1].set_title("Pontuação das regras — VT adequado e RSBI favorável")
    axes[1].grid(False)
    fig.colorbar(im, ax=axes[1], shrink=0.8, label="Pontuação (quanto maior, mais favorável)")
    fig.tight_layout()
    return _save(fig, "fig08_regras_consequentes.png")


def fig_surface(which: str = "nif_rr"):
    from skfuzzy import control as ctrl
    # Usa o motor manual em uma grade moderada (resolução de 51 x 51).
    if which == "nif_rr":
        xs = np.linspace(-60, -10, 61)
        ys = np.linspace(10, 46, 55)
        Z = np.zeros((len(ys), len(xs)))
        for i, rr in enumerate(ys):
            for j, nif in enumerate(xs):
                Z[i, j] = w.mamdani_evaluate({"NIF": nif, "RR": rr, "VT": 450, "RSBI": 90}).centroid
        X, Y = np.meshgrid(xs, ys)
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        cs = ax.contourf(X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 16), cmap="RdYlGn")
        ax.contour(X, Y, Z, levels=[40, 60], colors="#263238", linewidths=1.2)
        ax.set_xlabel(VAR_TITLES["NIF"])
        ax.set_ylabel(VAR_TITLES["RR"])
        ax.set_title("Superfície de decisão — VT = 450 mL e RSBI = 90\n(linhas: 40 e 60)")
        fig.colorbar(cs, ax=ax, label="Potencial de desmame")
        fig.tight_layout()
        return _save(fig, "fig09_superficie_nif_rr.png")
    else:
        xs = np.linspace(120, 900, 60)
        ys = np.linspace(20, 240, 60)
        Z = np.zeros((len(ys), len(xs)))
        for i, rsbi in enumerate(ys):
            for j, vt in enumerate(xs):
                Z[i, j] = w.mamdani_evaluate({"NIF": -30, "RR": 25, "VT": vt, "RSBI": rsbi}).centroid
        X, Y = np.meshgrid(xs, ys)
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        cs = ax.contourf(X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 16), cmap="RdYlGn")
        ax.contour(X, Y, Z, levels=[40, 60], colors="#263238", linewidths=1.2)
        ax.set_xlabel(VAR_TITLES["VT"])
        ax.set_ylabel(VAR_TITLES["RSBI"])
        ax.set_title("Superfície de decisão — NIF = −30 cm.H₂O e RR = 25\n(linhas: 40 e 60)")
        fig.colorbar(cs, ax=ax, label="Potencial de desmame")
        fig.tight_layout()
        return _save(fig, "fig10_superficie_vt_rsbi.png")


def fig_fuzzy_cases(pred: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12.4, 4.6))
    d = pred.copy()
    d["ord"] = np.arange(len(d))
    colors = [C[r] for r in d["Resultado"]]
    ax.scatter(d["ord"], d["crisp"], c=colors, s=55, edgecolor="white", linewidth=0.7, zorder=3)
    for _, r in d.iterrows():
        ax.plot([r["ord"], r["ord"]], [0, r["crisp"]], color="#CFD8DC", lw=0.8, zorder=1)
    ax.axhline(60, color="#263238", ls="--", lw=1.2, label="Limiar conservador 60")
    ax.axhline(40, color="#6C757D", ls=":", lw=1.1, label="Início da cautela 40")
    ax.axhline(50, color="#B0BEC5", ls="-", lw=0.8, alpha=0.7, label="Referência 50")
    ax.set_xticks(d["ord"], d["ID"], rotation=90, fontsize=6.7)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Potencial de desmame (crisp)")
    ax.set_title("Saída do sistema Fuzzy nos 30 casos", loc="left")
    handles = [Line2D([], [], marker="o", color="w", markerfacecolor=C["Fracasso"], markersize=7, label="Fracasso real"),
               Line2D([], [], marker="o", color="w", markerfacecolor=C["Sucesso"], markersize=7, label="Sucesso real")]
    handles += ax.get_legend_handles_labels()[0]
    ax.legend(handles=handles, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    return _save(fig, "fig11_fuzzy_30casos.png")


def fig_fuzzy_confusion(pred: pd.DataFrame):
    # matriz conservadora >=60 e matriz com limiar 50 para comparação
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.8))
    for ax, cutoff, title in [(axes[0], 50, "Limiar 50"), (axes[1], 60, "Limiar conservador 60")]:
        cm = w.confusion_matrix(pred["Resultado"], np.where(pred["crisp"] >= cutoff, "Sucesso", "Fracasso"))
        im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=15)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15, fontweight="bold", color="white" if cm[i, j] > 8 else "#1B5E20")
        ax.set_xticks([0, 1], ["Pred. Fracasso", "Pred. Sucesso"])
        ax.set_yticks([0, 1], ["Real Fracasso", "Real Sucesso"])
        ax.set_title(title)
        ax.grid(False)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
    fig.suptitle("Sistema Fuzzy — matrizes de confusão na base completa (n=30)", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "fig12_fuzzy_confusao.png")


def _plot_input_case(res: w.FuzzyResult, case_label: str, filename: str):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.0))
    for ax, var in zip(axes.ravel(), w.VARIABLES):
        x = w.FUZZY_UNIVERSES[var]
        for k, term in enumerate(w.FUZZY_TERMS[var]):
            y = _mf_array(var, term)
            ax.plot(x, y, color=TERM_COLORS[k], lw=1.8, label=w.TERM_DISPLAY[term])
            ax.fill_between(x, 0, y, color=TERM_COLORS[k], alpha=0.10)
        xv = res.x[var]
        ax.axvline(xv, color="#111111", lw=1.8)
        ax.set_title(f"{VAR_TITLES[var]}  |  x = {xv:g}", loc="left", fontsize=10)
        active = [(term, res.memberships[var][term]) for term in w.FUZZY_TERMS[var] if res.memberships[var][term] > 1e-9]
        txt = "\n".join([f"μ({w.TERM_DISPLAY[t]}) = {m:.3f}".replace(".", ",") for t, m in active])
        ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=7.6, color="#263238",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CFD8DC", lw=0.5, alpha=0.92))
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(-0.03, 1.16)
        ax.set_ylabel("μ(x)")
        ax.set_xlabel(VAR_TITLES[var], fontsize=8)
        ax.legend(loc="lower right", fontsize=6.8, ncol=3)
    fig.suptitle(f"{case_label} — fuzzificação das quatro entradas", fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    return _save(fig, filename)


def _plot_output_case(res: w.FuzzyResult, case_label: str, filename: str):
    fig = plt.figure(figsize=(11.2, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1])
    ax = fig.add_subplot(gs[0, 0])
    out_x = w.FUZZY_UNIVERSES["OUT"]
    # regiões dos consequentes, mais transparentes
    for term in ["MB", "B", "M", "A", "MA"]:
        y = _mf_array("OUT", term)
        ax.plot(out_x, y, color=OUT_COLORS[term], lw=1.0, alpha=0.55, ls="--")
    ax.fill_between(out_x, 0, res.aggregate, color=C["Fuzzy"], alpha=0.45, edgecolor="#1B5E20", linewidth=1.0, label="Região agregada (max)")
    ax.plot(out_x, res.aggregate, color="#1B5E20", lw=2.2)
    ax.axvline(res.centroid, color="#111111", lw=2.0, label=f"Centroide = {res.centroid:.2f}".replace(".", ","))
    ax.scatter([res.centroid], [0], color="#111111", zorder=5, s=30)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Potencial de desmame (0–100)")
    ax.set_ylabel("μ agregado(y)")
    ax.set_title("Agregação Mamdani e defuzzificação (COA)", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.98, 0.05, f"Área = {res.area:.3f}\nMomento = {res.moment:.2f}\ny* = {res.centroid:.3f}".replace(".", ","),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CFD8DC"))

    ax2 = fig.add_subplot(gs[0, 1])
    acts = pd.DataFrame(res.activations).sort_values(["alpha", "Regra"])
    labels = [f"R{r.Regra}: {w.TERM_DISPLAY[r.Consequente]}" for r in acts.itertuples()]
    colors = [OUT_COLORS[r.Consequente] for r in acts.itertuples()]
    ax2.barh(np.arange(len(acts)), acts["alpha"], color=colors, edgecolor="white")
    for i, v in enumerate(acts["alpha"]):
        ax2.text(v + 0.012, i, f"{v:.3f}".replace(".", ","), va="center", fontsize=7.0)
    ax2.set_yticks(np.arange(len(acts)), labels, fontsize=7.1)
    ax2.set_xlim(0, max(0.8, acts["alpha"].max() * 1.35))
    ax2.set_xlabel("Força de disparo α = min(μ antecedentes)")
    ax2.set_title(f"Regras ativadas (α > 0): {len(acts)}", loc="left")
    ax2.grid(axis="y", visible=False)
    fig.suptitle(f"{case_label} — inferência e saída defuzzificada", fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    return _save(fig, filename)


def fig_case_studies(full: pd.DataFrame):
    cases = [
        ("TE5", "Caso 1 — Sucesso claro (teste, TE5)", "fig13_caso1_entradas.png", "fig14_caso1_saida.png"),
        ("TR10", "Caso 2 — Fracasso claro (treino, TR10)", "fig15_caso2_entradas.png", "fig16_caso2_saida.png"),
        ("TE3", "Caso 3 — Crítico/duvidoso (teste, TE3)", "fig17_caso3_entradas.png", "fig18_caso3_saida.png"),
    ]
    out = []
    for cid, label, f1, f2 in cases:
        r = full.loc[full.ID == cid].iloc[0]
        res = w.mamdani_evaluate({v: float(r[v]) for v in w.VARIABLES})
        out.append((cid, _plot_input_case(res, label, f1), _plot_output_case(res, label, f2)))
    return out


def fig_comparacao_final(metric_rows):
    df = pd.DataFrame(metric_rows)
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    x = np.arange(len(df))
    wdt = 0.25
    for k, (col, color) in enumerate(zip(["Acurácia", "Sensibilidade", "Especificidade"], ["#12355B", "#2A9D8F", "#D98E04"])):
        ax.bar(x + (k - 1) * wdt, df[col], wdt, color=color, label=col, edgecolor="white")
        for i, v in enumerate(df[col]):
            ax.text(i + (k - 1) * wdt, v + 0.015, f"{v:.2f}", ha="center", fontsize=6.7, color="#37474F")
    ax.set_xticks(x, df["Modelo"], fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Valor (0–1)")
    ax.legend(ncol=3, loc="upper center")
    ax.set_title("Comparação final na base completa (n = 30)", loc="left")
    fig.tight_layout()
    return _save(fig, "fig19_comparacao_final.png")
