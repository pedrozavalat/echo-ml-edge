# ECHO — clasificador de especies en cámaras trampa

Clasificador de fauna a partir de imágenes de cámaras trampa del sur de Chile. Detecta y clasifica: Chucao, Gato huiña, Hued hued del sur, Pudú, Puma, Zorzal patagónico, más una clase `Unknown` para todo lo que no es especie objetivo.

Pipeline en producción:

```
imagen cámara trampa → detector de SpeciesNet (MegaDetector v5 bundled) → clasificador SpeciesNet fine-tuneado → especie + confianza
```

**Objetivo de éxito**: igualar o superar el baseline humano. Tres evaluadores clasificaron manualmente 77 imágenes reales de campo con accuracy 0.83–0.92 (ver `camaras-trampa/resultados/reporte_metricas.md`) — esa es la barra a superar.

Para la bitácora técnica detallada de cada iteración (datasets, decisiones, números por especie), ver `memoria_proyecto.md`.

## Estado actual

Checkpoint de referencia: `data/outputs/speciesnet_species_best_21classes.pt` — backbone EfficientNetV2-M de [SpeciesNet](https://github.com/google/cameratrapai) (congelado) + cabeza lineal (`Linear(1280,21)`) entrenada sobre las **21 clases completas de Snapshot** (las 6 especies objetivo + las 15 especies no-objetivo que se ven en la zona), **sin domain adaptation sobre HPL** (decisión explícita: preferimos generalización sobre ajustar a las 77 imágenes de benchmark). Las 15 clases no-objetivo actúan como clase de rechazo real (mucho más informativa que un umbral de confianza).

| Métrica | Valor |
|---|---|
| Val accuracy (split curado Snapshot, 21 clases) | 95.72% |
| Accuracy real global (77 img HPL, incluye 11 "Unknown") | **89.6%** |
| Accuracy real restringida a las 6 especies target (66/77 img) | **93.9%** (62/66) |
| Recall en "Unknown" (fauna no-objetivo) | **64%** (7/11) |
| Baseline humano | 83–92% |

A diferencia del `vit_hpl_finetuned.pt` (ver más abajo), este número **no está contaminado**: las 77 imágenes HPL nunca se usaron para entrenar, sólo para evaluar. Limitación conocida y explicada (no es ruido): las 15 clases no-objetivo de Snapshot son todas mamíferos — no hay ninguna clase de ave no-objetivo — así que las imágenes "Unknown" que en realidad son aves no-objetivo siguen cayendo (mal) en una de las 3 aves objetivo. Ver `memoria_proyecto.md` para el detalle completo, incluida la comparación contra la versión de 6 clases (`speciesnet_species_best.pt`, aún disponible, sin clase de rechazo) y un experimento de ensemble con la cabeza original de SpeciesNet.

### Iteración anterior: ViT-Small + domain adaptation (referencia histórica)

`data/outputs/vit_hpl_finetuned.pt` — ViT-Small, 7 clases (incluye Unknown), fine-tuneado por domain adaptation directa sobre las 77 imágenes HPL reales.

| Métrica | Valor |
|---|---|
| Accuracy real (77 img) | 96.1% (contaminado — mismas imágenes para entrenar y evaluar) |
| CV accuracy honesta (5-fold) | 78.9% ± 5.7% |

## Setup

Dos entornos separados porque `speciesnet` requiere Python <3.14 y trae su propia versión de PyTorch/onnx2torch, incompatibles con el resto del proyecto:

**Entorno principal** (`.venv`, Python 3.11+): PyTorch CPU, timm, ultralytics/PytorchWildlife para MegaDetector, scikit-learn, pandas, opencv — ver `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate   # o .venv\Scripts\activate en Windows
pip install -r requirements.txt
```

**Entorno SpeciesNet** (`.venv-speciesnet`, Python 3.12 — necesario para el modelo de referencia actual):

```bash
python3.12 -m venv .venv-speciesnet
source .venv-speciesnet/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # CPU-only, evita paquetes CUDA
pip install speciesnet scikit-learn openpyxl
```

No se requiere GPU en ninguno de los dos — todo corre en CPU (más lento: ResNet50 15 epochs ≈ 2h; el linear probe de SpeciesNet es más rápido porque sólo entrena la cabeza sobre embeddings cacheados).

## Uso

**Modelo de referencia actual — SpeciesNet fine-tuneado en 21 clases** (dentro de `.venv-speciesnet`):

```bash
SPECIESNET_CLASSES=21 python pipeline/train_classifier_speciesnet.py   # entrena la cabeza sobre las 21 clases de Snapshot
python pipeline/eval_speciesnet_21classes_pipeline.py --verbose        # valida contra las 77 imágenes reales HPL
```

Versión anterior sin clase de rechazo (sólo 6 especies, `speciesnet_species_best.pt`):

```bash
python pipeline/train_classifier_speciesnet.py                     # entrena la cabeza sobre las 6 especies objetivo
python pipeline/eval_speciesnet_finetuned_pipeline.py --verbose     # valida contra las 77 imágenes reales HPL
```

**Iteración anterior — ViT/ResNet50** (dentro de `.venv`):

Entrenar desde cero sobre el dataset Snapshot (21 clases):

```bash
python pipeline/train_classifier.py                              # ResNet50 (default)
BACKBONE=vit python pipeline/train_classifier.py                  # ViT-Small
```

Domain adaptation sobre imágenes HPL reales (7 clases, 5-fold CV + checkpoint final):

```bash
python pipeline/finetune_cv.py --backbone vit
```

Evaluar un checkpoint contra el pipeline real (MegaDetector + clasificador + métricas vs. ground truth):

```bash
CLS_CONF_THRESHOLD=0.3 python pipeline/eval_real_pipeline.py --checkpoint data/outputs/vit_hpl_finetuned.pt
```

## Datos

- **Snapshot** (`data/preprocessed_images`, ~5700 imágenes, 21 clases): dataset curado usado para pre-entrenar el backbone. [Link a dataset preprocesado](https://drive.google.com/drive/folders/1afP8EIH3L9HFg9CrJJq9swak_LVUUtZG?usp=drive_link) · [bounding boxes](https://drive.google.com/file/d/1Qz9UXpp21aV_fgSJLVhimCfX5xV7nPUw/view?usp=sharing)

  |     Índice | Clase           |          Imágenes |
  | --------: | --------------- | --------------: |
  |         0 | Hued hued       |             499 |
  |         1 | Pudu            |             499 |
  |         2 | Puma            |             500 |
  |         3 | Guina           |             314 |
  |         4 | Chucao          |             500 |
  |         5 | Zorzal          |             500 |
  |         6 | Caballo         |             214 |
  |         7 | Chingue         |             201 |
  |         8 | Guanaco         |             275 |
  |         9 | Huemul          |             245 |
  |        10 | Liebre europea  |             203 |
  |        11 | Oveja           |             204 |
  |        12 | Perro domestico |             206 |
  |        13 | Vaca            |             217 |
  |        14 | Zorro culpeo    |             215 |
  |        15 | Vison americano |             203 |
  |        16 | Quique          |             207 |
  |        17 | Jabali          |             211 |
  |        18 | Zorro chilla    |             255 |
  |        19 | Conejo europeo  |             253 |
  |        20 | Gato domestico  |             226 |
  | **Total** | **21 clases**  | **6147 imágenes** |

- **HPL** (`data/downloaded_eval_images.zip`, 77 imágenes reales de campo): el benchmark real, con ground truth en `camaras-trampa/evaluaciones/Evaluaciones_HPL_Ground_truth.xlsx`.

`data/`, `outputs/` y los checkpoints `*.pt` no están versionados en git (ver `.gitignore`) — viven local o en Drive.

## Estructura del repo

```
pipeline/                              # todo el código de entrenamiento/evaluación (ver pipeline/README.md)
  speciesnet_backbone.py                 # wrapper sobre el backbone de SpeciesNet (embedding 1280-d + cabeza original)
  train_classifier_speciesnet.py         # entrena la cabeza (linear probe); SPECIESNET_CLASSES=6|21 — modelo activo con 21
  eval_speciesnet_21classes_pipeline.py  # valida el checkpoint de 21 clases contra las 77 imágenes reales — modelo activo
  eval_speciesnet_finetuned_pipeline.py  # valida el checkpoint de 6 clases (sin rechazo) contra las 77 imágenes
  eval_speciesnet_ensemble_pipeline.py   # experimento: cabeza de 6 clases + cabeza original de SpeciesNet como filtro Otro
  eval_speciesnet_pipeline.py            # compara SpeciesNet "de fábrica" (sin fine-tuning) contra las 77 imágenes
  train_classifier.py                    # entrena desde cero sobre Snapshot (ResNet50 o ViT) — iteración anterior
  finetune_cv.py                         # domain adaptation ViT/ResNet50 sobre HPL con 5-fold CV — iteración anterior
  eval_real_pipeline.py                  # pipeline MegaDetector → clasificador ViT/ResNet50 → métricas reales
data/outputs/MANIFEST.md               # qué checkpoint es el activo vs. histórico/referencia
memoria_proyecto.md                    # bitácora técnica completa, iteración por iteración
camaras-trampa/                        # ground truth HPL, planillas de evaluación humana, benchmark y script de métricas
clasificacion_imagenes_campo/          # clasificación de imágenes de campo nuevas sin etiquetar (Drive)
```
