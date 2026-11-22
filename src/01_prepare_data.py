"""Etapa 1: descarga DeepLoc 1.0 y setHARD, clustering al 30 %, chequeo de fuga y CSVs finales."""
import json
import shutil
import subprocess
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"
WORK = RAW / "mmseqs_work"
DEEPLOC_URL = "https://services.healthtech.dtu.dk/services/DeepLoc-1.0/deeploc_data.fasta"
# Archivo oficial del repo de Light Attention (Stärk et al. 2021)
SETHARD_URL = "https://raw.githubusercontent.com/HannesStark/protein-localization/master/data_files/setHARD.fasta"
CLASSES = {"Cell.membrane", "Cytoplasm", "Endoplasmic.reticulum", "Extracellular", "Golgi.apparatus",
           "Lysosome/Vacuole", "Mitochondrion", "Nucleus", "Peroxisome", "Plastid"}
STANDARD, EXTRA = set("ACDEFGHIKLMNPQRSTVWY"), set("XUBZO")
THREADS = "8"


def download(url, dest):
    if not dest.exists():
        print(f"descargando {url}")
        urllib.request.urlretrieve(url, dest)


def read_fasta(path):
    """Devuelve (id, etiqueta cruda sin -M/-S/-U, es_test, secuencia)."""
    recs, hdr, seq = [], None, []
    for line in path.read_text().splitlines() + [">"]:
        if line.startswith(">"):
            if hdr:
                f = hdr.split()
                recs.append((f[0], f[1].rsplit("-", 1)[0], "test" in f[2:], "".join(seq).upper()))
            hdr, seq = line[1:], []
        elif line.strip():
            seq.append(line.strip())
    return recs


def write_fasta(df, path):
    path.write_text("".join(f">{i}\n{s}\n" for i, s in zip(df["id"], df["sequence"])))


def mmseqs(*args):
    subprocess.run(["mmseqs", *args, "--threads", THREADS, "-v", "1"], check=True)


def search_hits(query, target, tag):
    """Mejor identidad por secuencia de consulta, con e-value <= 1e-3 y cobertura >= 0.8 en ambas."""
    write_fasta(query, WORK / f"{tag}_q.fasta")
    write_fasta(target, WORK / "train.fasta")
    res = WORK / f"{tag}.m8"
    mmseqs("easy-search", str(WORK / f"{tag}_q.fasta"), str(WORK / "train.fasta"), str(res), str(WORK / f"tmp_{tag}"),
           "-s", "7.5", "-e", "1e-3", "--max-seqs", "1000",
           "--format-output", "query,target,pident,qcov,tcov,evalue")
    h = pd.read_csv(res, sep="\t", names=["q", "t", "pident", "qcov", "tcov", "evalue"])
    h = h[(h.evalue <= 1e-3) & (h.qcov >= 0.8) & (h.tcov >= 0.8)]
    best = h.groupby("q").pident.max()
    n = len(query)
    # pident sale en porcentaje (0-100) en esta versión de mmseqs
    return {f"{t}%": {"n": int((best >= t).sum()), "pct": round(100 * (best >= t).sum() / n, 2)}
            for t in (30, 40, 50)} | {"n_query": n}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    download(DEEPLOC_URL, RAW / "deeploc_data.fasta")
    sethard_ok = True
    try:
        download(SETHARD_URL, RAW / "setHARD.fasta")
    except Exception as e:  # no se sustituye por otra fuente
        sethard_ok = False
        print(f"AVISO: no se obtuvo setHARD ({e}); se omite sethard.csv")

    # a. DeepLoc
    recs = read_fasta(RAW / "deeploc_data.fasta")
    discarded = Counter(r[1] for r in recs if r[1] not in CLASSES)
    df = pd.DataFrame([r for r in recs if r[1] in CLASSES], columns=["id", "label", "test", "sequence"])
    df["split"] = np.where(df.pop("test"), "test", "train")

    # c. clustering de todo DeepLoc
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    write_fasta(df, WORK / "all.fasta")
    mmseqs("easy-cluster", str(WORK / "all.fasta"), str(WORK / "clu"), str(WORK / "tmp_clu"),
           "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0", "-s", "7.5")
    clu = pd.read_csv(WORK / "clu_cluster.tsv", sep="\t", names=["rep", "id"])
    df = df.merge(clu.rename(columns={"rep": "cluster"}), on="id", how="left")
    assert df.cluster.notna().all()
    df = df[["id", "sequence", "label", "split", "cluster"]]

    # b. setHARD
    sh = None
    if sethard_ok:
        sh = pd.DataFrame(read_fasta(RAW / "setHARD.fasta"), columns=["id", "label", "test", "sequence"]).drop(columns="test")
        sh_discarded = Counter(sh.label[~sh.label.isin(CLASSES)])
        sh = sh[sh.label.isin(CLASSES)][["id", "sequence", "label"]]

    # d. fuga: test y setHARD contra train
    train, test = df[df.split == "train"], df[df.split == "test"]
    mixed = df.groupby("cluster").split.nunique()
    homology = {
        "params": "mmseqs easy-search -s 7.5 -e 1e-3 --max-seqs 1000; cobertura query y target >= 0.8; mejor hit por secuencia",
        "test_vs_train": search_hits(test, train, "test"),
        "clusters_total": int(len(mixed)),
        "clusters_with_train_and_test": int((mixed > 1).sum()),
        "test_seqs_in_mixed_clusters": int(test.cluster.isin(mixed[mixed > 1].index).sum()),
    }
    if sh is not None:
        homology["sethard_vs_train"] = search_hits(sh, train, "sethard")
    (OUT / "homology_report.json").write_text(json.dumps(homology, indent=2))

    # e. salidas
    df.to_csv(OUT / "dataset.csv", index=False)
    if sh is not None:
        sh.to_csv(OUT / "sethard.csv", index=False)

    def lengths(s):
        n = s.str.len()
        return {"min": int(n.min()), "median": float(n.median()), "p95": float(n.quantile(0.95)),
                "max": int(n.max()), "gt_1022": int((n > 1022).sum())}

    def nonstd(d):
        return int(d.sequence.map(lambda s: bool(set(s) & EXTRA)).sum())

    summary = {
        "mmseqs_version": subprocess.run(["mmseqs", "version"], capture_output=True, text=True).stdout.strip(),
        "n_per_split": df.split.value_counts().to_dict() | ({"sethard": len(sh)} if sh is not None else {}),
        "n_per_class_and_split": {s: g.label.value_counts().to_dict() for s, g in df.groupby("split")}
                                 | ({"sethard": sh.label.value_counts().to_dict()} if sh is not None else {}),
        "lengths": {"dataset": lengths(df.sequence)} | ({"sethard": lengths(sh.sequence)} if sh is not None else {}),
        "discarded_labels_deeploc": dict(discarded),
        "n_clusters": int(df.cluster.nunique()),
        "seqs_with_nonstandard_symbols_XUBZO": {"dataset": nonstd(df)} | ({"sethard": nonstd(sh)} if sh is not None else {}),
        "sethard_available": sh is not None,
    }
    if sh is not None:
        summary["discarded_labels_sethard"] = dict(sh_discarded)
    (OUT / "data_summary.json").write_text(json.dumps(summary, indent=2))

    # f. asserts
    for d in [df] + ([sh] if sh is not None else []):
        assert d.id.is_unique and (d.sequence.str.len() > 0).all()
        assert set(d.label) == CLASSES, set(d.label) ^ CLASSES
        assert all(set(s) <= STANDARD | EXTRA for s in d.sequence)
    print(json.dumps(summary, indent=2))
    print(json.dumps(homology, indent=2))


if __name__ == "__main__":
    main()
