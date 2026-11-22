"""Etapa 2a: AAC (20) y DPC (400) sobre la secuencia completa, sin truncar."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AA = "ACDEFGHIKLMNPQRSTVWY"
IDX = {a: i for i, a in enumerate(AA)}


def features(seq):
    # Los símbolos no estándar se ignoran; los dipéptidos solapados que los tocan no cuentan
    v = [IDX.get(a, -1) for a in seq]
    aac, dpc = np.zeros(20), np.zeros(400)
    for i, a in enumerate(v):
        if a >= 0:
            aac[a] += 1
            if i + 1 < len(v) and v[i + 1] >= 0:
                dpc[a * 20 + v[i + 1]] += 1
    return aac / aac.sum(), (dpc / dpc.sum() if dpc.sum() else dpc)


def main():
    out = ROOT / "data/embeddings"
    out.mkdir(parents=True, exist_ok=True)
    for prefix, csv in (("", "dataset.csv"), ("sethard_", "sethard.csv")):
        seqs = pd.read_csv(ROOT / "data/processed" / csv).sequence
        aac, dpc = (np.array(m, dtype=np.float32) for m in zip(*map(features, seqs)))
        assert aac.shape == (len(seqs), 20) and dpc.shape == (len(seqs), 400)
        assert np.allclose(aac.sum(1), 1, atol=1e-5)
        # Una secuencia de un solo residuo válido no tiene dipéptidos: su fila DPC es 0
        short = (dpc.sum(1) == 0).sum()
        assert np.allclose(dpc.sum(1)[dpc.sum(1) > 0], 1, atol=1e-5)
        print(f"{csv}: {len(seqs)} filas, {short} sin dipéptidos válidos")
        np.save(out / f"{prefix}aac.npy", aac)
        np.save(out / f"{prefix}dpc.npy", dpc)


if __name__ == "__main__":
    main()
