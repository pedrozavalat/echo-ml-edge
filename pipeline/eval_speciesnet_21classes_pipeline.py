"""
Evalúa el checkpoint de SpeciesNet fine-tuneado sobre las 21 clases de Snapshot
(6 especies objetivo + 15 fauna no-objetivo de la zona) sobre las 77 imágenes
reales HPL. A diferencia de `eval_speciesnet_finetuned_pipeline.py` (sólo 6
clases, sin opción de rechazo), este checkpoint puede predecir directamente
una de las 15 especies no-objetivo -- la pregunta es si las imágenes GT
"Unknown" caen efectivamente dentro de esas 15 categorías.

Sólo corre dentro de .venv-speciesnet.

Uso:
    source .venv-speciesnet/bin/activate
    python eval_speciesnet_21classes_pipeline.py [--checkpoint path] [--verbose]
"""

import argparse
import re
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, f1_score
from speciesnet.detector import SpeciesNetDetector

from speciesnet_backbone import EMBEDDING_DIM, IMG_SIZE, SpeciesNetFeatureExtractor

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
GT_EXCEL_PATH = REPO_ROOT / "camaras-trampa" / "evaluaciones" / "Evaluaciones_HPL_Ground_truth.xlsx"
EVAL_IMAGES_ZIP = DATA_DIR / "downloaded_eval_images.zip"
EVAL_IMAGES_DIR = DATA_DIR / "eval_downloaded_images"
DEFAULT_CHECKPOINT = DATA_DIR / "outputs" / "speciesnet_species_best_21classes.pt"

DET_CONF_THRESHOLD = 0.25
PADDING_RATIO = 0.10

TARGET_SPECIES = ["Chucao", "Gato huiña", "Hued hued del sur", "Pudú", "Puma", "Zorzal patagónico"]

# Nombre de carpeta Snapshot (tal como quedó en idx_to_class_name del checkpoint)
# -> etiqueta canónica usada en el resto del proyecto para comparar contra GT.
# Los 6 objetivo se mapean a su nombre completo; las 15 no-objetivo se mapean
# a "Unknown" para el cálculo de métricas, pero se reporta el nombre crudo
# aparte para ver en qué categoría específica cae cada imagen.
SPECIES_CANONICAL_MAP = {
    "chucao": "Chucao",
    "gato huina": "Gato huiña",
    "gato huiña": "Gato huiña",
    "guina": "Gato huiña",
    "hued hued": "Hued hued del sur",
    "hued hued del sur": "Hued hued del sur",
    "pudu": "Pudú",
    "pudú": "Pudú",
    "puma": "Puma",
    "zorzal": "Zorzal patagónico",
    "zorzal patagonico": "Zorzal patagónico",
    "zorzal patagónico": "Zorzal patagónico",
    "unknown": "Unknown",
    "otro": "Unknown",
    # las 15 no-objetivo -> Unknown (no son ninguna de las especies objetivo)
    "caballo": "Unknown",
    "chingue": "Unknown",
    "guanaco": "Unknown",
    "huemul": "Unknown",
    "liebre europea": "Unknown",
    "oveja": "Unknown",
    "perro domestico": "Unknown",
    "vaca": "Unknown",
    "zorro culpeo": "Unknown",
    "vison americano": "Unknown",
    "quique": "Unknown",
    "jabali": "Unknown",
    "zorro chilla": "Unknown",
    "conejo europeo": "Unknown",
    "gato domestico": "Unknown",
}


def norm_text(x):
    if pd.isna(x) or x is None:
        return ""
    x = str(x).strip()
    x = unicodedata.normalize("NFKD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))
    return x.lower().strip()


def canonical_species(x):
    key = norm_text(x)
    return None if key == "" else SPECIES_CANONICAL_MAP.get(key, "Unknown")


def extract_excel_row_from_filename(img_path):
    match = re.search(r"row_(\d+)", img_path.stem)
    return int(match.group(1)) if match else None


def clamp(value, lo, hi):
    return max(lo, min(value, hi))


def get_best_animal_detection(detections):
    animal_dets = [d for d in detections if d["label"] == "animal"]
    if not animal_dets:
        return None
    return max(animal_dets, key=lambda d: d["conf"])


def prepare_bbox_xywhn(bbox_xywhn, img_width, img_height, padding_ratio=0.10):
    x, y, w, h = bbox_xywhn
    x1, y1 = x * img_width, y * img_height
    x2, y2 = (x + w) * img_width, (y + h) * img_height
    pw, ph = (x2 - x1) * padding_ratio, (y2 - y1) * padding_ratio
    x1 = int(clamp(x1 - pw, 0, img_width))
    y1 = int(clamp(y1 - ph, 0, img_height))
    x2 = int(clamp(x2 + pw, 0, img_width))
    y2 = int(clamp(y2 + ph, 0, img_height))
    return None if (x2 <= x1 or y2 <= y1) else (x1, y1, x2, y2)


def detect_and_crop(img_path, detector):
    with Image.open(img_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        preprocessed = detector.preprocess(img)
        result = detector.predict(str(img_path), preprocessed)
        detections = result.get("detections", [])
        best = get_best_animal_detection(detections)

        if best is None or best["conf"] < DET_CONF_THRESHOLD:
            return None, (best["conf"] if best else None)

        bbox = prepare_bbox_xywhn(best["bbox"], img.width, img.height, PADDING_RATIO)
        if bbox is None:
            return None, best["conf"]
        crop = img.crop(bbox).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        return crop, best["conf"]


@torch.no_grad()
def classify_crop(crop_img, feature_extractor, head, idx_to_class_name):
    arr = np.asarray(crop_img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).unsqueeze(0)
    embedding = feature_extractor(x)
    probs = torch.softmax(head(embedding), dim=1)[0]
    pred_idx = int(torch.argmax(probs))
    return idx_to_class_name[pred_idx], float(probs[pred_idx])


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def main(checkpoint_path, verbose):
    print_section("Cargando modelos")
    print("Cargando detector de SpeciesNet (MegaDetector v5 bundled)...")
    detector = SpeciesNetDetector("kaggle:google/speciesnet/pyTorch/v4.0.3a/1")

    print(f"Cargando clasificador fine-tuneado desde {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    idx_to_class_name = ckpt["idx_to_class_name"]
    num_classes = ckpt["num_classes"]

    feature_extractor = SpeciesNetFeatureExtractor()
    head = nn.Linear(EMBEDDING_DIM, num_classes)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    print(f"  Clases ({num_classes}): {list(idx_to_class_name.values())}")

    print_section("Extrayendo imágenes de evaluación")
    EVAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(EVAL_IMAGES_ZIP) as zf:
        zf.extractall(EVAL_IMAGES_DIR)

    image_paths = sorted(
        p for p in EVAL_IMAGES_DIR.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    print(f"Imágenes encontradas: {len(image_paths)}")

    print_section("Cargando Ground Truth")
    gt_df = pd.read_excel(GT_EXCEL_PATH, sheet_name="Ground Truth Dataset")
    gt_df["excel_row"] = gt_df.index + 2
    gt_df["gt_label"] = gt_df["Common Name"].apply(canonical_species)
    print(f"GT entries: {len(gt_df)}")
    print(gt_df["gt_label"].value_counts().to_string())

    print_section("Corriendo pipeline (Detector → Clasificador 21 clases)")
    results = []
    for img_path in image_paths:
        excel_row = extract_excel_row_from_filename(img_path)
        if excel_row is None:
            continue

        crop_img, det_conf = detect_and_crop(img_path, detector)

        if crop_img is None:
            results.append({
                "excel_row": excel_row,
                "pred_detected": False,
                "pred_species": None,
                "clase_modelo_original": None,
                "cls_conf": None,
            })
            continue

        model_class, cls_conf = classify_crop(crop_img, feature_extractor, head, idx_to_class_name)
        results.append({
            "excel_row": excel_row,
            "pred_detected": True,
            "pred_species": canonical_species(model_class),
            "clase_modelo_original": model_class,
            "cls_conf": cls_conf,
        })

    results_df = pd.DataFrame(results).sort_values("excel_row")

    if verbose:
        print("\nDetalle por imagen:")
        for _, r in results_df.iterrows():
            if not r["pred_detected"]:
                print(f"  row_{r['excel_row']:04d} | sin detección")
                continue
            print(
                f"  row_{r['excel_row']:04d} | {r['pred_species']:<10} <- "
                f"{r['clase_modelo_original']} ({r['cls_conf']:.2f})"
            )

    print_section("Métricas globales")
    eval_df = gt_df.merge(results_df, on="excel_row", how="inner")
    eval_df["pred_combined"] = eval_df.apply(
        lambda r: r["pred_species"] if r["pred_detected"] else "No detectado", axis=1
    )

    y_true = eval_df["gt_label"]
    y_pred = eval_df["pred_combined"]

    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_wei = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    det_rate = eval_df["pred_detected"].mean()

    cls_df = eval_df[eval_df["pred_detected"] & eval_df["pred_species"].notna()]
    cls_acc = accuracy_score(cls_df["gt_label"], cls_df["pred_species"]) if len(cls_df) else float("nan")

    print(f"  N evaluadas:          {len(eval_df)}")
    print(f"  Detección rate:       {det_rate:.4f}  ({det_rate:.1%})")
    print(f"  Accuracy global:      {acc:.4f}  ({acc:.1%})")
    print(f"  Acc. clasificación:   {cls_acc:.4f}  ({cls_acc:.1%})")
    print(f"  F1 macro:             {f1_mac:.4f}")
    print(f"  F1 weighted:          {f1_wei:.4f}")

    print_section("Recall por especie objetivo + Unknown (GT vs predicción)")
    for sp in TARGET_SPECIES + ["Unknown"]:
        sp_rows = eval_df[eval_df["gt_label"] == sp]
        if sp_rows.empty:
            continue
        correct = (sp_rows["pred_combined"] == sp).sum()
        n = len(sp_rows)
        print(f"  {sp:<22} {correct}/{n}  ({correct/n:.0%})")

    print_section("¿En qué categoría cruda (de las 21) cayeron las imágenes 'Unknown'?")
    unknown_rows = eval_df[eval_df["gt_label"] == "Unknown"]
    for _, r in unknown_rows.sort_values("excel_row").iterrows():
        raw = r["clase_modelo_original"] if r["pred_detected"] else "(sin detección)"
        conf = f"{r['cls_conf']:.2f}" if r["cls_conf"] else "-"
        print(f"  row_{r['excel_row']:04d} -> {raw} (conf={conf})")

    print_section("Confusiones principales")
    for sp in TARGET_SPECIES:
        sp_rows = eval_df[eval_df["gt_label"] == sp]
        if sp_rows.empty:
            continue
        wrong = sp_rows[sp_rows["pred_combined"] != sp]["pred_combined"].value_counts()
        if not wrong.empty:
            print(f"  {sp}: {dict(wrong)}")

    print("\nListo.")
    return eval_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(Path(args.checkpoint), args.verbose)
