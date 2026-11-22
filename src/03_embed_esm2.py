"""Etapa 2b: embeddings congelados de ESM-2 (mean y <cls>) y costo medido de inferencia."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
EMB, COST = ROOT / "data/embeddings", ROOT / "results/metrics/embedding_cost.json"
MODELS = {"esm2_8m": "facebook/esm2_t6_8M_UR50D", "esm2_35m": "facebook/esm2_t12_35M_UR50D",
          "esm2_150m": "facebook/esm2_t30_150M_UR50D"}
DEVICE = "mps"
TOKEN_BUDGET = 4096  # tokens (con padding) por lote: conservador para 24 GB compartidos con el sistema
MAX_RES = 1022
PAD = 32  # pocas formas distintas: MPS compila y cachea un grafo por forma y con longitudes libres se agota la memoria


def truncate(s):
    return s if len(s) <= MAX_RES else s[:511] + s[-511:]


@torch.no_grad()
def run_batch(model, tok, seqs):
    """Un forward; si MPS se queda sin memoria, parte el lote a la mitad (nunca cae a CPU)."""
    try:
        enc = tok(seqs, return_tensors="pt", padding=True, pad_to_multiple_of=PAD).to(DEVICE)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].clone()
        m[:, 0] = 0  # <cls>
        m[torch.arange(len(seqs)), enc["attention_mask"].sum(1) - 1] = 0  # <eos>
        mean = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        return mean.cpu().numpy(), h[:, 0].cpu().numpy()
    except RuntimeError as e:
        if "out of memory" not in str(e).lower() or len(seqs) == 1:
            raise
        print(f"  MPS sin memoria con lote de {len(seqs)}; se divide")
        torch.mps.empty_cache()
        a, b = run_batch(model, tok, seqs[:len(seqs) // 2]), run_batch(model, tok, seqs[len(seqs) // 2:])
        return np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]])


def embed(model, tok, seqs, budget, peak=None):
    """Devuelve (mean, bos) en el orden de `seqs`. `peak` es una lista con la memoria pico de MPS."""
    seqs = [truncate(s) for s in seqs]
    order = np.argsort([len(s) for s in seqs], kind="stable")
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * -(-(len(seqs[i]) + 2) // PAD) * PAD > budget:  # la nueva es la más larga del lote
            batches.append(cur)
            cur = []
        cur.append(i)
    batches.append(cur)
    mean, bos = [], []
    for b in batches:
        r = run_batch(model, tok, [seqs[i] for i in b])
        mean.append(r[0]), bos.append(r[1])
        if peak is not None:
            peak[0] = max(peak[0], torch.mps.driver_allocated_memory())
        torch.mps.empty_cache()  # sin esto el caché por forma crece hasta ~24 GB y ahoga la máquina
    inv = np.argsort(np.concatenate(batches))  # deshace el orden por longitud
    return np.concatenate(mean)[inv].astype(np.float32), np.concatenate(bos)[inv].astype(np.float32)


def check(model, tok, seqs, dim):
    """Lote vs. una por una: detecta errores de máscara/padding."""
    short = sorted(seqs, key=len)[:5]
    short = [s[:k] for s, k in zip(short, (20, 35, 50, 70, 90))]  # longitudes distintas fuerzan padding
    bm, bb = embed(model, tok, short, budget=10**6)
    im = np.concatenate([embed(model, tok, [s], budget=1)[0] for s in short])
    ib = np.concatenate([embed(model, tok, [s], budget=1)[1] for s in short])
    dm, db = float(np.abs(bm - im).max()), float(np.abs(bb - ib).max())
    assert bm.shape == (5, dim) and dm < 1e-4 and db < 1e-4, (bm.shape, dm, db)
    print(f"  chequeo lote vs individual: max|dif| mean={dm:.2e} bos={db:.2e}")
    return dm, db


def main():
    names = sys.argv[1:2] or list(MODELS)
    assert all(n in MODELS for n in names), f"modelo desconocido; opciones: {list(MODELS)}"
    assert torch.backends.mps.is_available()
    seqs = pd.read_csv(ROOT / "data/processed/dataset.csv").sequence.tolist()
    sh = pd.read_csv(ROOT / "data/processed/sethard.csv").sequence.tolist()
    COST.parent.mkdir(parents=True, exist_ok=True)
    for name in names:
        files = [EMB / f"{p}{name}_{pool}.npy" for p in ("", "sethard_") for pool in ("mean", "bos")]
        if all(f.exists() for f in files):
            print(f"{name}: ya existe, se omite")
            continue
        print(f"{name}: cargando {MODELS[name]}")
        tok = AutoTokenizer.from_pretrained(MODELS[name])
        model = AutoModel.from_pretrained(MODELS[name], dtype=torch.float32, add_pooling_layer=False).to(DEVICE).eval()
        dim = model.config.hidden_size
        dm, db = check(model, tok, seqs, dim)

        torch.mps.empty_cache()
        peak = [torch.mps.driver_allocated_memory()]
        torch.mps.synchronize()
        t0 = time.perf_counter()
        mean, bos = embed(model, tok, seqs, TOKEN_BUDGET, peak)
        torch.mps.synchronize()
        secs = time.perf_counter() - t0

        smean, sbos = embed(model, tok, sh, TOKEN_BUDGET)  # fuera de la medición
        for arr, n in ((mean, len(seqs)), (bos, len(seqs)), (smean, len(sh)), (sbos, len(sh))):
            assert arr.shape == (n, dim) and arr.dtype == np.float32 and np.isfinite(arr).all()
        EMB.mkdir(parents=True, exist_ok=True)
        for f, arr in zip(files, (mean, bos, smean, sbos)):
            np.save(f, arr)

        cost = json.loads(COST.read_text()) if COST.exists() else {}
        cost[name] = {
            "hf_id": MODELS[name], "params": sum(p.numel() for p in model.parameters()), "dim": dim,
            "layers": model.config.num_hidden_layers, "n_sequences": len(seqs),
            "inference_seconds": round(secs, 2), "seconds_per_1000_seqs": round(1000 * secs / len(seqs), 3),
            "peak_mps_memory_bytes": peak[0], "peak_memory_metric": "torch.mps.driver_allocated_memory (máximo tras cada lote)",
            "token_budget_per_batch": TOKEN_BUDGET, "device": DEVICE, "dtype": "float32",
            "torch": torch.__version__, "transformers": transformers.__version__,
            "check_batch_vs_single_max_abs_diff": {"mean": dm, "bos": db},
        }
        COST.write_text(json.dumps(cost, indent=2))
        print(f"  {secs:.1f} s, {1000 * secs / len(seqs):.2f} s/1000 seqs, pico {peak[0] / 2**30:.2f} GiB")
        del model
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
