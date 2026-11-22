"""Etapa 3: entrena regresión logística (C por CV agrupada) y un MLP de chequeo; evalúa en test y setHARD."""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROC, EMB = ROOT / "data/processed", ROOT / "data/embeddings"
METRICS, PRED = ROOT / "results/metrics", ROOT / "results/predictions"
REPS = ["aac", "dpc", "esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m"]
GRID_C = [0.0001, 0.001, 0.01, 0.1, 1, 10]
SEED, N_BOOT, MAX_ITER = 42, 1000, 3000


def emb_name(rep, prefix=""):
    return f"{prefix}{rep}" + ("_mean" if rep.startswith("esm2") else "") + ".npy"


def load_split(rep):
    ds, sh = pd.read_csv(PROC / "dataset.csv"), pd.read_csv(PROC / "sethard.csv")
    X, Xs = np.load(EMB / emb_name(rep)), np.load(EMB / emb_name(rep, "sethard_"))
    assert len(X) == len(ds) and len(Xs) == len(sh), f"{rep}: filas CSV != matriz"
    return ds, sh, X, Xs


def bootstrap_ci(y, pred, labels):
    """IC 95 % percentil de F1-macro, MCC y accuracy; mismos remuestreos para cada llamada (seed 42)."""
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        vals.append((f1_score(y[i], pred[i], labels=labels, average="macro", zero_division=0),
                     matthews_corrcoef(y[i], pred[i]), accuracy_score(y[i], pred[i])))
    lo, hi = np.percentile(vals, [2.5, 97.5], axis=0)
    return {m: [float(a), float(b)] for m, a, b in zip(["f1_macro", "mcc", "accuracy"], lo, hi)}


def evaluate(y, pred, classes):
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    return {"f1_macro": f1_score(y, pred, labels=labels, average="macro", zero_division=0),
            "mcc": matthews_corrcoef(y, pred), "accuracy": accuracy_score(y, pred),
            "ci95": bootstrap_ci(y, pred, labels), "n_test": len(y),
            "per_class": {c: {"f1": f[k], "precision": p[k], "recall": r[k], "support": int(s[k])}
                          for k, c in enumerate(classes)},
            "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist()}


def run(rep):
    t0 = time.time()
    ds, sh, X, Xs = load_split(rep)
    classes = sorted(ds.label.unique())
    idx = {c: i for i, c in enumerate(classes)}
    tr = (ds.split == "train").to_numpy()
    Xtr, ytr, groups = X[tr], ds.label[tr].map(idx).to_numpy(), ds.cluster[tr].to_numpy()
    tests = {"test": (X[~tr], ds.label[~tr].map(idx).to_numpy()), "sethard": (Xs, sh.label.map(idx).to_numpy())}

    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr, groups))
    for a, b in folds:
        assert not set(groups[a]) & set(groups[b]), "clúster repartido entre entrenamiento y validación"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=MAX_ITER))
        gs = GridSearchCV(pipe, {"logisticregression__C": GRID_C}, scoring="f1_macro", cv=folds, refit=True, n_jobs=1)
        gs.fit(Xtr, ytr)
        mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True,
                                                            random_state=SEED))
        mlp.fit(Xtr, ytr)
    n_warn = len(w)  # aplica a CV + reajuste + MLP
    if n_warn:
        print(f"AVISO {rep}: {n_warn} ConvergenceWarning; sube MAX_ITER o repórtalo")

    cv = {str(c): {"f1_macro_mean": float(gs.cv_results_["mean_test_score"][k]),
                   "f1_macro_std": float(gs.cv_results_["std_test_score"][k]),
                   "f1_macro_folds": [float(gs.cv_results_[f"split{j}_test_score"][k]) for j in range(5)]}
          for k, c in enumerate(GRID_C)}
    best_c = gs.best_params_["logisticregression__C"]
    models = {"logreg": (gs.best_estimator_, {"C": best_c, "class_weight": "balanced", "max_iter": MAX_ITER,
                                              "n_iter_final": int(gs.best_estimator_[-1].n_iter_[0]), "cv": cv}),
              "mlp": (mlp, {"hidden_layer_sizes": [256], "class_weight": None, "early_stopping": True,
                            "n_iter": int(mlp[-1].n_iter_)})}
    PRED.mkdir(parents=True, exist_ok=True)
    for name, (_, yt) in tests.items():  # verdad de terreno para McNemar y bootstrap pareado en 05
        np.save(PRED / f"y_{name}.npy", yt.astype(np.int64))
    for clf, (model, cfg) in models.items():
        for name, (Xt, yt) in tests.items():
            pred = model.predict(Xt)
            out = {"rep": rep, "clf": clf, "testset": name, "classes": classes, "dim": X.shape[1],
                   "n_train": int(tr.sum()), "config": cfg, "convergence_warnings": n_warn,
                   "seconds_total_rep": round(time.time() - t0, 1), **evaluate(yt, pred, classes)}
            (METRICS / f"{rep}__{clf}__{name}.json").write_text(json.dumps(out, indent=1))
            np.save(PRED / f"{rep}__{clf}__{name}.npy", pred.astype(np.int64))
            print(f"{rep} {clf} {name}: F1-macro={out['f1_macro']:.3f}")
    print(f"{rep}: C={best_c}, {time.time() - t0:.0f} s")


def main():
    explicit = sys.argv[1:]
    reps = explicit or [r for r in REPS if (EMB / emb_name(r)).exists()]
    METRICS.mkdir(parents=True, exist_ok=True)
    for rep in reps:
        # sin argumentos no se recalcula lo ya hecho; con argumentos se fuerza
        if not explicit and (METRICS / f"{rep}__mlp__sethard.json").exists():
            print(f"{rep}: ya calculado, se omite")
            continue
        run(rep)


if __name__ == "__main__":
    main()
