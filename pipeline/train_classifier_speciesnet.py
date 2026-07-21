"""
Entrena la cabeza de clasificación de SpeciesNet (linear probe sobre el
backbone congelado EfficientNetV2-M) usando el dataset curado de Snapshot
(data/preprocessed_images). Por defecto usa sólo las 6 especies objetivo
(carpetas 0-5); con SPECIESNET_CLASSES=21 usa las 21 clases completas
(6 objetivo + 15 fauna no-objetivo de la zona), para ver si las imágenes
"Unknown" del benchmark HPL caen dentro de esas 15 categorías.

Como el backbone está congelado, el embedding de 1280-d de cada imagen es
determinista y no cambia entre epochs -> se precalcula una sola vez (paso
lento, ~0.5-0.65s/imagen en CPU) y se cachea en disco; entrenar la cabeza
lineal sobre esos vectores cacheados es casi instantáneo por epoch.

Sólo corre dentro de .venv-speciesnet (ver speciesnet_backbone.py).

Uso:
    source .venv-speciesnet/bin/activate
    python train_classifier_speciesnet.py                    # 6 clases (activo)
    SPECIESNET_CLASSES=21 python train_classifier_speciesnet.py   # 21 clases
"""

import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm

from speciesnet_backbone import IMG_SIZE, EMBEDDING_DIM, SpeciesNetFeatureExtractor

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
VAL_SIZE = 0.20
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", 8))
TRAIN_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
EPOCHS = int(os.environ.get("EPOCHS", 60))
LR = float(os.environ.get("LR", 1e-3))
WEIGHT_DECAY = 1e-4

# "6" (default, modelo activo) o "21" (6 objetivo + 15 fauna no-objetivo de Snapshot)
SPECIESNET_CLASSES = os.environ.get("SPECIESNET_CLASSES", "6")
_suffix = "" if SPECIESNET_CLASSES == "6" else f"_{SPECIESNET_CLASSES}classes"

EMBEDDINGS_CACHE_PATH = OUTPUT_DIR / f"speciesnet_snapshot_embeddings{_suffix}.pt"
BEST_MODEL_PATH = OUTPUT_DIR / f"speciesnet_species_best{_suffix}.pt"
HISTORY_PATH = OUTPUT_DIR / f"training_history_speciesnet{_suffix}.csv"

# folder id (dentro de data/preprocessed_images) -> nombre canónico del proyecto
# (ver README / data/detections_metadata.json)
TARGET_FOLDER_TO_CLASS = {
    "0": "Hued hued del sur",
    "1": "Pudú",
    "2": "Puma",
    "3": "Gato huiña",
    "4": "Chucao",
    "5": "Zorzal patagónico",
}

# Las 15 especies no-objetivo de Snapshot que se pueden ver en la zona de interés.
OTHER_FOLDER_TO_CLASS = {
    "6": "Caballo",
    "7": "Chingue",
    "8": "Guanaco",
    "9": "Huemul",
    "10": "Liebre europea",
    "11": "Oveja",
    "12": "Perro domestico",
    "13": "Vaca",
    "14": "Zorro culpeo",
    "15": "Vison americano",
    "16": "Quique",
    "17": "Jabali",
    "18": "Zorro chilla",
    "19": "Conejo europeo",
    "20": "Gato domestico",
}

FOLDER_TO_CLASS = (
    TARGET_FOLDER_TO_CLASS
    if SPECIESNET_CLASSES == "6"
    else {**TARGET_FOLDER_TO_CLASS, **OTHER_FOLDER_TO_CLASS}
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_imagefolder_root(base_dir: Path) -> Path:
    candidates = [base_dir, *base_dir.rglob("*")]
    for p in candidates:
        if p.is_dir() and all((p / fid).is_dir() for fid in FOLDER_TO_CLASS):
            return p
    raise RuntimeError(
        f"No se encontraron las carpetas {list(FOLDER_TO_CLASS)} bajo {base_dir}"
    )


def build_samples(root: Path):
    samples = []
    for folder_id, class_name in FOLDER_TO_CLASS.items():
        folder = root / folder_id
        for img_path in sorted(folder.glob("*.jpg")):
            samples.append((img_path, class_name))
    return samples


class ImagePathDataset(Dataset):
    """Carga imágenes crudas (resize + [0,1]) para pasarlas por el backbone."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, _ = self.samples[idx]
        img = Image.open(path).convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR
        )
        # HWC, float32 en [0,1] -- igual que el preprocesamiento propio de
        # speciesnet/classifier.py (sin normalización tipo ImageNet).
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr)


def compute_or_load_embeddings(samples, class_to_idx):
    if EMBEDDINGS_CACHE_PATH.exists():
        print(f"Cargando embeddings cacheados desde {EMBEDDINGS_CACHE_PATH}")
        cache = torch.load(EMBEDDINGS_CACHE_PATH, weights_only=False)
        if cache["num_samples"] == len(samples):
            return cache["embeddings"], cache["labels"]
        print("El cache no coincide con el dataset actual, recalculando...")

    print(f"Precalculando embeddings para {len(samples)} imágenes (esto corre una sola vez)...")
    feature_extractor = SpeciesNetFeatureExtractor()
    loader = DataLoader(
        ImagePathDataset(samples), batch_size=EMBED_BATCH_SIZE, shuffle=False, num_workers=0
    )

    all_embeddings = torch.empty((len(samples), EMBEDDING_DIM), dtype=torch.float32)
    labels = torch.tensor(
        [class_to_idx[class_name] for _, class_name in samples], dtype=torch.long
    )

    offset = 0
    start = time.time()
    for images in tqdm(loader, desc="embeddings"):
        batch_size = images.shape[0]
        emb = feature_extractor(images)
        all_embeddings[offset : offset + batch_size] = emb
        offset += batch_size
    print(f"Embeddings calculados en {time.time() - start:.1f}s")

    torch.save(
        {"embeddings": all_embeddings, "labels": labels, "num_samples": len(samples)},
        EMBEDDINGS_CACHE_PATH,
    )
    return all_embeddings, labels


def train_one_epoch(head, loader, criterion, optimizer):
    head.train()
    running_loss, running_correct, running_total = 0.0, 0, 0
    for embeddings, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = head(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        running_loss += loss.item() * labels.size(0)
        running_correct += (preds == labels).sum().item()
        running_total += labels.size(0)
    return running_loss / running_total, running_correct / running_total


@torch.no_grad()
def evaluate(head, loader, criterion):
    head.eval()
    running_loss, running_correct, running_total = 0.0, 0, 0
    for embeddings, labels in loader:
        logits = head(embeddings)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        running_loss += loss.item() * labels.size(0)
        running_correct += (preds == labels).sum().item()
        running_total += labels.size(0)
    return running_loss / running_total, running_correct / running_total


def main():
    set_seed(SEED)
    root = find_imagefolder_root(DATA_DIR / "preprocessed_images")
    print(f"ImageFolder root: {root}")

    samples = build_samples(root)
    print(f"Total imágenes ({SPECIESNET_CLASSES} clases): {len(samples)}")

    classes = sorted(set(FOLDER_TO_CLASS.values()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class_name = {i: c for c, i in class_to_idx.items()}

    embeddings, labels = compute_or_load_embeddings(samples, class_to_idx)

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(samples))
    n_val = int(len(samples) * VAL_SIZE)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

    train_ds = TensorDataset(embeddings[train_idx], labels[train_idx])
    val_ds = TensorDataset(embeddings[val_idx], labels[val_idx])
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=False)

    head = nn.Linear(EMBEDDING_DIM, len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = []
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(head, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(head, val_loader, criterion)
        elapsed = time.time() - start
        print(
            f"[{epoch}/{EPOCHS}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({elapsed:.2f}s)"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_name": "speciesnet_efficientnetv2m",
                    "num_classes": len(classes),
                    "head_state_dict": head.state_dict(),
                    "class_to_idx": class_to_idx,
                    "idx_to_class_name": idx_to_class_name,
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
                BEST_MODEL_PATH,
            )
            print(f"  -> Nuevo mejor checkpoint guardado (val_acc={best_val_acc:.4f})")

    with open(HISTORY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"\nMejor val_acc: {best_val_acc:.4f}. Checkpoint: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
