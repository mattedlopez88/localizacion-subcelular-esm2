# Localización subcelular de proteínas con ESM-2

¿Cuánto aporta el tamaño de un modelo de lenguaje de proteínas? Embeddings congelados de ESM-2
(8M, 35M, 150M, 650M) frente a descriptores clásicos (AAC, DPC) en DeepLoc 1.0 (10 clases).

Proyecto semestral de Inteligencia Artificial 2. Integrantes: Erick Granda, Joao Gavilanes, Matias Lopez.

## Resultados preliminares (Entrega 2)

Regresión logística, una semilla. `test` es la partición oficial de DeepLoc 1.0 y setHARD el
conjunto estricto de Stärk et al. Tablas completas en `results/tables/`.

| Representación | F1-macro test | Accuracy test | F1-macro setHARD | s / 1000 secuencias |
|---|---|---|---|---|
| AAC | 0.303 | 0.374 | 0.173 | – |
| DPC | 0.346 | 0.441 | 0.219 | – |
| ESM-2 8M | 0.554 | 0.664 | 0.371 | 18.7 |
| ESM-2 35M | 0.603 | 0.719 | 0.415 | 37.6 |
| ESM-2 150M | 0.650 | 0.761 | 0.436 | 68.8 |
| ESM-2 650M | 0.700 | 0.806 | 0.521 | 210.8 |

## Estructura

```
data/raw/          descargas originales (no versionadas)
data/processed/    dataset.csv, sethard.csv, clusters e informe de homología
data/embeddings/   matrices .npy por representación (no versionadas)
src/               pipeline, un script por etapa
results/           metrics/, tables/, figures/ (versionados)
```

## Contrato entre etapas

- `data/processed/dataset.csv`: columnas `id, sequence, label, split, cluster`.
  `split` ∈ {train, test} es la partición oficial de DeepLoc 1.0. `cluster` es el clúster de
  MMseqs2 al 30 % de identidad y agrupa los pliegues de la validación cruzada.
- `data/processed/sethard.csv`: mismas columnas sin `split` ni `cluster`; evaluación estricta.
- `data/embeddings/{rep}_{pool}.npy`: matriz float32 con las filas en el orden de
  `dataset.csv`; `sethard_{rep}_{pool}.npy` en el orden de `sethard.csv`.
  `rep` ∈ {aac, dpc, esm2_8m, esm2_35m, esm2_150m, esm2_650m}; `pool` ∈ {mean, bos} (solo ESM-2).
- `results/metrics/`: un JSON por (representación, clasificador, conjunto de prueba).

## Protocolo experimental (fijo)

- Clasificador principal: `StandardScaler` + regresión logística con `class_weight="balanced"`;
  `C` elegido por validación cruzada de 5 pliegues agrupada por `cluster`, optimizando F1-macro.
- Secundario: un MLP de una capa oculta, solo como chequeo.
- Métricas: F1-macro (principal), MCC, accuracy, F1 por clase y matriz de confusión.
- Significancia: IC 95 % por bootstrap (1000 remuestreos) y prueba de McNemar.
- Eficiencia: parámetros y segundos por cada 1000 secuencias.
- Secuencias de más de 1022 residuos: se conservan los primeros 511 y los últimos 511.
- Semilla: 42.

## Ejecución

Requiere [uv](https://docs.astral.sh/uv/) y MMseqs2 (`brew install mmseqs2`).

```bash
uv sync
```

Los comandos de cada etapa se documentan aquí a medida que se implementan.

```bash
# Etapa 1: descarga DeepLoc 1.0 y setHARD, clustering al 30 %, chequeo de homología (idempotente, ~1 min)
uv run python src/01_prepare_data.py

# Etapa 2: AAC y DPC (segundos)
uv run python src/02_features_classic.py

# Etapa 2: embeddings ESM-2 (8M, 35M, 150M) y costo en results/metrics/embedding_cost.json;
# idempotente, acepta un modelo (esm2_8m | esm2_35m | esm2_150m | esm2_650m). No uses la máquina mientras mide el costo.
uv run python src/03_embed_esm2.py

# Opcional: ESM-2 650M (no corre por defecto)
uv run python src/03_embed_esm2.py esm2_650m

# Etapa 3: regresión logística (C por CV agrupada) y MLP de chequeo, métricas en results/metrics (~1 min).
# Sin argumentos omite lo ya calculado; con una representación (p. ej. esm2_650m) la recalcula.
uv run python src/04_train_eval.py

# Etapa 4: tablas, figuras y results/summary.json (idempotente)
uv run python src/05_report.py
```

## Uso de IA generativa

Se usó Claude para asistir en la generación de código e investigación.
