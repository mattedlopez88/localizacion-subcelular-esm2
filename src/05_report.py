"""Etapa 4: tablas, figuras y summary.json a partir de results/ y data/processed/*.json (idempotente)."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES, PROC = ROOT / "results", ROOT / "data/processed"
MET, PRED, TAB, FIG = RES / "metrics", RES / "predictions", RES / "tables", RES / "figures"
REPS = ["aac", "dpc", "esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m"]
NAMES = {"aac": "AAC", "dpc": "DPC", "esm2_8m": "ESM-2 8M", "esm2_35m": "ESM-2 35M",
         "esm2_150m": "ESM-2 150M", "esm2_650m": "ESM-2 650M"}
SHORT = {k: v.replace("ESM-2 ", "") for k, v in NAMES.items()}
CLASES_ES = {"Nucleus": "Núcleo", "Cytoplasm": "Citoplasma", "Extracellular": "Extracelular",
             "Mitochondrion": "Mitocondria", "Cell.membrane": "Membrana", "Endoplasmic.reticulum": "RE",
             "Plastid": "Plastidio", "Golgi.apparatus": "Golgi", "Lysosome/Vacuole": "Lisosoma/Vacuola",
             "Peroxisome": "Peroxisoma"}
SETS = {"test": "test", "sethard": "setHARD"}
# Okabe-Ito (apta para daltónicos)
C_TEST, C_HARD, C_CLASSIC, C_ESM = "#0072B2", "#D55E00", "#E69F00", "#009E73"
SEED, N_BOOT = 42, 1000

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300})
W = 3.35  # ancho de una columna IEEE (~8,5 cm) en pulgadas


def load(rep, clf, ts):
    return json.loads((MET / f"{rep}__{clf}__{ts}.json").read_text())


def f3(x):
    return f"{x:.3f}"


def with_ci(m, k):
    lo, hi = m["ci95"][k]
    return f"{m[k]:.3f} [{lo:.3f}, {hi:.3f}]"


def save_table(df, name):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.tex").write_text(df.to_latex(index=False, escape=True, column_format="l" + "r" * (df.shape[1] - 1)))


def save_fig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


def results_table(reps, clf, cost, compact):
    rows = []
    for r in reps:
        row = {"Representación": NAMES[r], "Dim.": load(r, clf, "test")["dim"],
               "Parámetros (M)": f"{cost[r]['params'] / 1e6:.1f}" if r in cost else "-"}
        for ts, tn in SETS.items():
            m = load(r, clf, ts)
            keys = [("f1_macro", "F1-macro")] if compact else [("f1_macro", "F1-macro"), ("mcc", "MCC"), ("accuracy", "Acc.")]
            for k, kn in keys:
                row[f"{kn} {tn}"] = with_ci(m, k) if not compact else f3(m[k])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_diff(y, pa, pb, n_classes):
    """Diferencia F1-macro (b - a) con IC 95 % percentil; mismos remuestreos para ambos modelos."""
    rng = np.random.default_rng(SEED)
    labels = list(range(n_classes))
    d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        d.append(f1_score(y[i], pb[i], labels=labels, average="macro", zero_division=0)
                 - f1_score(y[i], pa[i], labels=labels, average="macro", zero_division=0))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.mean(d)), float(lo), float(hi)


def mcnemar_row(a, b, ts):
    y, pa, pb = (np.load(PRED / f"{n}.npy") for n in (f"y_{ts}", f"{a}__logreg__{ts}", f"{b}__logreg__{ts}"))
    ca, cb = pa == y, pb == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())  # b: A acierta y B falla; c: al revés
    exact = n01 + n10 < 25
    res = mcnemar([[int((ca & cb).sum()), n01], [n10, int((~ca & ~cb).sum())]], exact=exact, correction=True)
    return {"Comparación": f"{NAMES[a]} vs {NAMES[b]}", "Conjunto": SETS[ts], "b (A acierta, B falla)": n01,
            "c (A falla, B acierta)": n10, "Estadístico": f3(res.statistic), "p": f"{res.pvalue:.4g}",
            "Prueba": "exacta" if exact else "chi2 con corrección"}


def fig_f1_vs_params(esm, cls_best, cost):
    fig, ax = plt.subplots(figsize=(W, 2.6))
    x = [cost[r]["params"] for r in esm]
    for ts, col, off in (("test", C_TEST, 0.97), ("sethard", C_HARD, 1.03)):
        m = [load(r, "logreg", ts) for r in esm]
        y = np.array([k["f1_macro"] for k in m])
        err = np.array([[k["f1_macro"] - k["ci95"]["f1_macro"][0], k["ci95"]["f1_macro"][1] - k["f1_macro"]] for k in m]).T
        ax.errorbar(np.array(x) * off, y, yerr=err, fmt="o-", color=col, capsize=2, ms=4, lw=1, label=f"ESM-2, {SETS[ts]}")
        for r, ls in zip(("aac", "dpc"), (":", "--")):
            ax.axhline(load(r, "logreg", ts)["f1_macro"], color=col, ls=ls, lw=1, alpha=.8,
                       label=f"{NAMES[r]}, {SETS[ts]}")
    ax.set_xscale("log")
    ax.set_xticks(x, [SHORT[r] for r in esm])
    ax.minorticks_off()
    ax.set_xlabel("Parámetros del modelo (escala log)")
    ax.set_ylabel("F1-macro")
    ax.set_ylim(0, 0.8)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left", handlelength=1.6, columnspacing=0.8)
    save_fig(fig, "fig_f1_vs_params")


def fig_f1_vs_tiempo(esm, cost):
    fig, ax = plt.subplots(figsize=(W, 2.3))
    t = [cost[r]["seconds_per_1000_seqs"] for r in esm]
    f = [load(r, "logreg", "test")["f1_macro"] for r in esm]
    ax.plot(t, f, "o-", color=C_ESM, ms=4, lw=1)
    for r, a, b in zip(esm, t, f):
        ax.annotate(SHORT[r], (a, b), textcoords="offset points", xytext=(4, -9))
    ax.set_xlabel("Tiempo de inferencia (s / 1000 secuencias)")
    ax.set_ylabel("F1-macro (test)")
    ax.margins(x=0.12, y=0.15)
    save_fig(fig, "fig_f1_vs_tiempo")


def fig_confusion(best, classes):
    cm = np.array(load(best, "logreg", "test")["confusion_matrix"], dtype=float)
    cm = cm / cm.sum(1, keepdims=True)
    names = [CLASES_ES[c] for c in classes]
    fig, ax = plt.subplots(figsize=(W, W))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > .5 else "black")
    ax.set_xticks(range(len(names)), names, rotation=90)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    for s in ax.spines.values():
        s.set_visible(True)
    save_fig(fig, "fig_confusion")


def fig_f1_por_clase(classic, esm, classes, n_train):
    order = sorted(classes, key=lambda c: -n_train[c])
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(W, 3.4))
    for k, (r, col, dy) in enumerate(((classic, C_CLASSIC, -0.2), (esm, C_ESM, 0.2))):
        pc = load(r, "logreg", "test")["per_class"]
        ax.barh(y + dy, [pc[c]["f1"] for c in order], height=0.4, color=col, label=NAMES[r])
    ax.set_yticks(y, [CLASES_ES[c] for c in order])
    ax.invert_yaxis()
    ax.set_xlabel("F1 (test)")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, loc="lower right")
    save_fig(fig, "fig_f1_por_clase")


def fig_clases(summary_json, classes):
    order = sorted(classes, key=lambda c: -summary_json["train"][c])
    fig, ax = plt.subplots(figsize=(W, 3.0))
    y = np.arange(len(order))
    for k, (s, col) in enumerate((("train", C_TEST), ("test", C_ESM), ("sethard", C_HARD))):
        tot = sum(summary_json[s].values())
        ax.barh(y + (k - 1) * 0.27, [100 * summary_json[s][c] / tot for c in order], height=0.27, color=col,
                label=SETS.get(s, s) if s != "train" else "train")
    ax.set_yticks(y, [CLASES_ES[c] for c in order])
    ax.invert_yaxis()
    ax.set_xlabel("Proporción de secuencias (%)")
    ax.legend(frameon=False, loc="lower right")
    save_fig(fig, "fig_clases")


def main():
    reps = [r for r in REPS if (MET / f"{r}__logreg__test.json").exists()]
    esm = [r for r in reps if r.startswith("esm2")]
    classic = [r for r in reps if not r.startswith("esm2")]
    cost = json.loads((MET / "embedding_cost.json").read_text())
    hom = json.loads((PROC / "homology_report.json").read_text())
    dsum = json.loads((PROC / "data_summary.json").read_text())
    classes = load(reps[0], "logreg", "test")["classes"]
    n_train = dsum["n_per_class_and_split"]["train"]
    f1 = {r: load(r, "logreg", "test")["f1_macro"] for r in reps}
    best_classic, best_esm = max(classic, key=f1.get), max(esm, key=f1.get)

    # tablas
    main_df = results_table(reps, "logreg", cost, compact=False)
    save_table(main_df, "main_results")
    save_table(results_table(reps, "mlp", cost, compact=True), "secondary_mlp")

    pc = {r: load(r, "logreg", "test")["per_class"] for r in reps}
    save_table(pd.DataFrame([{"Clase": CLASES_ES[c], **{NAMES[r]: f3(pc[r][c]["f1"]) for r in reps},
                              "n train": n_train[c]} for c in classes]), "per_class_f1")

    pairs = [(best_classic, best_esm)] + list(zip(esm[:-1], esm[1:]))
    mc = pd.DataFrame([mcnemar_row(a, b, ts) for a, b in pairs for ts in SETS])
    save_table(mc, "mcnemar")

    eff = []
    for i, r in enumerate(esm):
        eff.append({"Modelo": NAMES[r], "Parámetros (M)": f"{cost[r]['params'] / 1e6:.1f}",
                    "s / 1000 seq.": f"{cost[r]['seconds_per_1000_seqs']:.1f}",
                    "Memoria pico (GB)": f"{cost[r]['peak_mps_memory_bytes'] / 1e9:.2f}",
                    "F1-macro test": f3(f1[r]),
                    "Ganancia vs anterior": f3(f1[r] - f1[esm[i - 1]]) if i else "-"})
    save_table(pd.DataFrame(eff), "efficiency")

    # figuras
    fig_f1_vs_params(esm, best_classic, cost)
    fig_f1_vs_tiempo(esm, cost)
    fig_confusion(max(reps, key=f1.get), classes)
    fig_f1_por_clase(best_classic, best_esm, classes, n_train)
    fig_clases(dsum["n_per_class_and_split"], classes)

    # summary.json
    diffs = {}
    for ts in SETS:
        y, pa, pb = (np.load(PRED / f"{n}.npy") for n in (f"y_{ts}", f"{best_classic}__logreg__{ts}", f"{best_esm}__logreg__{ts}"))
        d, lo, hi = paired_bootstrap_diff(y, pa, pb, len(classes))
        diffs[SETS[ts]] = {"diff_f1_macro_observed": load(best_esm, "logreg", ts)["f1_macro"] - load(best_classic, "logreg", ts)["f1_macro"],
                           "boot_mean": d, "ci95": [lo, hi]}
    summary = {
        "n_per_split": dsum["n_per_split"], "n_per_class_and_split": dsum["n_per_class_and_split"],
        "best_classic": best_classic, "best_esm": best_esm,
        "chosen_C_logreg": {r: load(r, "logreg", "test")["config"]["C"] for r in reps},
        "convergence_warnings": {r: load(r, "logreg", "test")["convergence_warnings"] for r in reps},
        "logreg": {r: {SETS[ts]: {k: load(r, "logreg", ts)[k] for k in ("f1_macro", "mcc", "accuracy", "ci95")} for ts in SETS} for r in reps},
        "mlp_f1_macro": {r: {SETS[ts]: load(r, "mlp", ts)["f1_macro"] for ts in SETS} for r in reps},
        "diff_best_esm_minus_best_classic": diffs,
        "mcnemar": mc.to_dict("records"),
        "esm_gain_vs_previous": {r["Modelo"]: r["Ganancia vs anterior"] for r in eff},
        "homology": {k: hom[k] for k in ("test_vs_train", "sethard_vs_train", "clusters_total",
                                          "clusters_with_train_and_test", "test_seqs_in_mixed_clusters")},
        "homology_params": hom["params"],
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(main_df.to_string(index=False))


if __name__ == "__main__":
    main()
