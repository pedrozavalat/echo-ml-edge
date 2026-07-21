"""
Envoltorio sobre el backbone de SpeciesNet (EfficientNetV2-M) para usarlo como
extractor de features congelado + cabeza de clasificación propia.

El .pt de SpeciesNet es un torch.fx.GraphModule (conversión ONNX de un modelo
TensorFlow), no un nn.Module con atributos limpios como `.fc`/`.head`. Cada
operación del grafo (incluida la capa de pooling final) es un submódulo real,
así que se puede enganchar un forward hook en el nodo
"SpeciesNet/efficientnetv2-m/avg_pool/Mean_Squeeze__..." para capturar el
embedding de 1280-d antes de la cabeza de clasificación original (2498 clases).

Sólo funciona dentro de un entorno con el paquete `speciesnet` instalado
(ver .venv-speciesnet), porque deserializar el checkpoint requiere que
`onnx2torch` esté importable.
"""

from pathlib import Path

import torch
import torch.nn as nn

EMBEDDING_DIM = 1280
IMG_SIZE = 480
_EMBEDDING_NODE_SUFFIX = "avg_pool_mean_squeeze"


def _find_classifier_checkpoint(model_dir: Path) -> Path:
    matches = list(model_dir.glob("always_crop_*.pt"))
    if not matches:
        raise FileNotFoundError(
            f"No se encontró el checkpoint del clasificador SpeciesNet en {model_dir}"
        )
    return matches[0]


def _default_model_dir() -> Path:
    cache_root = Path.home() / ".cache" / "kagglehub" / "models" / "google" / "speciesnet" / "pyTorch"
    versions = sorted(p for p in cache_root.glob("v4.0.3a/*") if p.is_dir())
    if not versions:
        raise FileNotFoundError(
            f"No se encontró el modelo de SpeciesNet en {cache_root}. "
            "Corré primero `python -m speciesnet.scripts.run_model --help` "
            "para que se auto-descargue desde Kaggle."
        )
    return versions[-1]


def _find_labels_file(model_dir: Path) -> Path:
    matches = list(model_dir.glob("always_crop_*.labels.*.txt"))
    if not matches:
        raise FileNotFoundError(
            f"No se encontró el archivo de labels del clasificador SpeciesNet en {model_dir}"
        )
    return matches[0]


class SpeciesNetFeatureExtractor(nn.Module):
    """Backbone congelado de SpeciesNet (EfficientNetV2-M) que devuelve embeddings de 1280-d.

    También expone, sin costo adicional, la predicción *original* de SpeciesNet
    (2498 clases) vía `forward_with_original(x)` — el grafo original nunca se
    modifica, sólo se le engancha un hook para leer un tensor intermedio, así que
    la cabeza de 2498 clases sigue funcionando en paralelo a la nuestra.
    """

    def __init__(self, model_dir: Path = None):
        super().__init__()
        model_dir = model_dir or _default_model_dir()
        checkpoint_path = _find_classifier_checkpoint(model_dir)
        self.backbone = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        with open(_find_labels_file(model_dir), encoding="utf-8") as f:
            self.original_labels = [line.strip() for line in f.readlines()]

        embedding_node_target = None
        for node in self.backbone.graph.nodes:
            if node.op == "call_module" and _EMBEDDING_NODE_SUFFIX in node.name:
                embedding_node_target = node.target
        if embedding_node_target is None:
            raise RuntimeError(
                "No se encontró el nodo de embedding (avg_pool/Mean_Squeeze) "
                "en el grafo de SpeciesNet."
            )
        self._embedding_module = self.backbone.get_submodule(embedding_node_target)
        self._captured = {}
        self._embedding_module.register_forward_hook(self._capture_hook)

    def _capture_hook(self, module, inp, out):
        self._captured["embedding"] = out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.backbone(x)
        return self._captured["embedding"]

    def forward_with_original(self, x: torch.Tensor):
        """Devuelve (embedding_1280d, logits_originales_2498_clases)."""
        with torch.no_grad():
            original_logits = self.backbone(x)
        return self._captured["embedding"], original_logits

    def original_taxonomy_string(self, class_idx: int) -> str:
        return self.original_labels[class_idx]


class SpeciesNetClassifierHead(nn.Module):
    """Extractor de features de SpeciesNet (congelado) + cabeza lineal entrenable."""

    def __init__(self, feature_extractor: SpeciesNetFeatureExtractor, num_classes: int):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = nn.Linear(EMBEDDING_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.feature_extractor(x)
        return self.head(embedding)
