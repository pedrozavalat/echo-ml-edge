# pipeline/

Todos los scripts de entrenamiento y evaluación del proyecto. Se importan entre sí por estar
en el mismo directorio (`from speciesnet_backbone import ...`), y todos resuelven `data/` en la
raíz del repo vía `Path(__file__).resolve().parent.parent`. Si movés alguno de estos archivos a
otra carpeta, hay que ajustar esa línea.

## Línea activa (checkpoint de producción)

- `speciesnet_backbone.py` — wrapper del backbone congelado de SpeciesNet (EfficientNetV2-M).
- `train_classifier_speciesnet.py` — entrena la cabeza lineal. `SPECIESNET_CLASSES=21` entrena
  el checkpoint activo actual.
- `eval_speciesnet_21classes_pipeline.py` — evalúa el checkpoint activo
  (`data/outputs/speciesnet_species_best_21classes.pt`) sobre las 77 imágenes HPL reales.

Requieren `.venv-speciesnet` (Python 3.12) — el checkpoint no es deserializable en el `.venv`
principal (Python 3.14). Setup: `pip install -r pipeline/requirements-speciesnet.txt` (ver ese
archivo para el detalle de versiones y el auto-download del checkpoint base vía kagglehub).

Para correr la evaluación del checkpoint activo hacen falta además dos archivos que están
excluidos de `.gitignore` con una excepción puntual (ver ese archivo): el checkpoint mismo
(`data/outputs/speciesnet_species_best_21classes.pt`) y las imágenes de evaluación
(`data/downloaded_eval_images.zip`), más `camaras-trampa/evaluaciones/Evaluaciones_HPL_Ground_truth.xlsx`
(ground truth, no ignorado).

## Histórico / referencia (no se usan en producción, se mantienen como comparación)

- `eval_speciesnet_finetuned_pipeline.py` — evalúa el checkpoint de 6 clases (iteración 7,
  `speciesnet_species_best.pt`), previo a agregar la clase de rechazo.
- `eval_speciesnet_ensemble_pipeline.py` — experimento de ensemble con la cabeza original de
  SpeciesNet (2498 clases) como filtro de Unknown. No confirmado como reemplazo.
- `eval_speciesnet_pipeline.py` — benchmark de SpeciesNet sin fine-tuning (out of the box),
  usado como punto de comparación externo.
- `eval_real_pipeline.py` — pipeline histórico MegaDetector → ResNet50/ViT (iteraciones previas
  a SpeciesNet).
- `train_classifier.py` — entrenamiento desde cero sobre Snapshot con ResNet50 o ViT.
- `finetune_cv.py` — domain adaptation con 5-fold CV sobre HPL (ViT/ResNet50), 7 clases.
