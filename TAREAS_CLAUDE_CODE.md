# TAREAS PARA CLAUDE CODE
> Leer CONTEXT.md primero. Luego este archivo.
> FORMATO: Todo en notebooks .ipynb autocontenidos (igual que TA-001).
> NO crear scripts .py separados — el código va dentro del notebook.

---

## CONTEXTO DEL FORMATO

El proyecto usa Jupyter Notebooks (.ipynb) como entregables principales.
Patrón establecido en TA-001: notebook autocontenido con celdas Markdown
explicativas + celdas de código + outputs visibles (prints, gráficas, tablas).

El docente evalúa el notebook ejecutado con outputs visibles.
NO se usan scripts .py separados ni imports de src/.

---

## TAREA INMEDIATA: TA-003

**Archivo:** `notebooks/TA-003_ResNet50_Benchmarking.ipynb`

### Encabezado Markdown (igual que TA-001)
```
# TA-003 — Entrenamiento ResNet-50 para Benchmarking CNN
**Proyecto:** Sistema Multimodal de IA para Apoyo Diagnóstico Clínico Veterinario — Vargas Vet
**Curso:** Taller Integrador 1 — UPAO
**Integrantes:** Baylón Toledo, Diogho Matteo (PM) · Saavedra Arroyo, Sebastián Alonso (Scrum Master)
**Sprint 2:** 31 May – 20 Jun 2026
```

### Estructura de celdas

**Celda 1 — Verificación GPU** (copiar de TA-001, misma lógica)
- torch.cuda.is_available(), nombre GPU, VRAM

**Celda 2 — Imports y configuración**
```python
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pydicom, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from pathlib import Path
import time, json
from tqdm import tqdm

# Rutas
BASE = Path(r'D:\UPAO_Diogho_Baylon\IX\Taller Integrador I\ModelosIA\Radiografias')
MANIFESTS = BASE / 'dataset_split' / 'manifests'
DATASET_SPLIT = BASE / 'dataset_split'
SEED = 42
IMAGE_SIZE = 224
CLASSES = ['alveolar_pattern', 'bronchial_pattern', 'pleural_effusion',
           'cardiomegaly', 'no_finding']
NUM_CLASSES = len(CLASSES)
EXCLUDE_TAGS = {'fracture', 'pneumomediastinum', 'costal_fracture', 'exclude'}
```

**Celda 3 — Cargar y explorar manifests**
- Leer train.csv, val.csv, test.csv con pandas
- Mostrar conteo de imágenes por split
- Mostrar distribución de clases en train

**Celda 4 — Clase VetXRayDataset**
Definir la clase Dataset completa inline:
- `load_dicom_normalized(path)`: carga DICOM, normaliza percentil 2-98, maneja MONOCHROME1
- `build_label_vector(tag_str)`: convierte TAG string → vector binario [5]
- `class VetXRayDataset(Dataset)`: __init__, __len__, __getitem__
  - Lee .dcm con pydicom
  - Convierte a PIL pseudo-RGB (canal gris × 3)
  - Aplica transforms (resize 224×224, normalización ImageNet)

**Celda 5 — Verificar dataset**
- Instanciar train_ds, val_ds, test_ds
- Mostrar 3 radiografías de ejemplo con matplotlib (igual que celda 12 de TA-001)
- Mostrar shape del tensor y label de ejemplo

**Celda 6 — Class weights**
```python
def compute_class_weights(df):
    # pos_weight[i] = (N - n_pos[i]) / n_pos[i]
    # para BCEWithLogitsLoss
```
- Calcular e imprimir class weights por clase

**Celda 7 — Construir modelo ResNet-50**
```python
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# Reemplazar cabeza: Dropout(0.4) + Linear(2048, 5)
model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, NUM_CLASSES))
```
- Mostrar arquitectura resumida
- Contar parámetros entrenables vs totales

**Celda 8 — Funciones de entrenamiento**
Definir inline:
- `train_epoch(model, loader, criterion, optimizer, device, scaler)`
- `eval_epoch(model, loader, criterion, device)` → retorna loss + AUC + F1 + Acc
- `class EarlyStopping`

**Celda 9 — Configuración de entrenamiento**
```python
EPOCHS_MAX = 50
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
PHASE1_EPOCHS = 5   # feature extraction (solo cabeza)
EARLY_STOPPING_PATIENCE = 10
```

**Celda 10 — FASE 1: Feature Extraction**
- Congelar backbone, entrenar solo cabeza 5 épocas
- Imprimir métricas por época

**Celda 11 — FASE 2: Fine-tuning completo**
- Descongelar todo, LR=1e-5 para capas tempranas
- Scheduler: ReduceLROnPlateau
- Early stopping patience=10
- Mixed precision torch.cuda.amp
- Guardar mejor checkpoint `resnet50_best.pth`
- Imprimir métricas por época

**Celda 12 — Curvas de entrenamiento**
- Plot: train_loss vs val_loss por época
- Plot: val_auc por época con línea de umbral 0.75
- Marcar mejor época

**Celda 13 — Evaluación en TEST**
- Cargar mejor checkpoint
- Calcular AUC, F1-macro, Accuracy en test set
- Medir tiempo de inferencia promedio (50 muestras)

**Celda 14 — Resultados finales (Markdown)**
Tabla para DO-002:
```markdown
| Métrica | ResNet-50 |
|---------|-----------|
| AUC (val) | X.XXX |
| AUC (test) | X.XXX ≥ 0.75? |
| F1-macro (test) | X.XXX ≥ 0.75? |
| Accuracy (test) | X.XXX |
| Épocas entrenadas | XX |
| Inferencia (seg/img) | X.XX < 6? |
```

**Celda 15 — Guardar resultados JSON**
```python
results = {
    'model': 'resnet50',
    'val_auc': ..., 'test_auc': ...,
    'test_f1_macro': ..., 'test_accuracy': ...,
    'epochs_trained': ..., 'best_epoch': ...,
    'inference_time_sec': ...
}
# Guardar en notebooks/resnet50_results.json
```

---

## TAREA SIGUIENTE: TA-004

**Archivo:** `notebooks/TA-004_EfficientNet_DenseNet_Benchmarking.ipynb`

Estructura idéntica a TA-003 pero:

1. Entrena EfficientNet-B0:
   - Scheduler: CosineAnnealingWarmRestarts(T_0=10, T_mult=2) + warm-up 5 épocas
   - batch_size=32

2. Entrena DenseNet-121:
   - Scheduler: ReduceLROnPlateau (igual que ResNet-50)
   - batch_size=16 (menor para VRAM)

3. Celda final — Tabla comparativa 3 modelos:
```python
# Cargar resnet50_results.json + resultados del notebook actual
# Construir DataFrame comparativo
# Mostrar tabla con pandas + selección justificada del ganador
```

```markdown
## Tabla Comparativa — Benchmarking CNN (DO-002)
| Modelo | AUC val | AUC test | F1 test | Acc test | Inferencia | Ganador |
|--------|---------|----------|---------|----------|------------|---------|
| ResNet-50 | | | | | | |
| EfficientNet-B0 | | | | | | |
| DenseNet-121 | | | | | **← GANADOR** |

**Modelo ganador:** DenseNet-121 (mayor AUC, historial en CheXNet)
**Justificación:** [basada en resultados reales del experimento]
```

---

## IMPORTANTE PARA CLAUDE CODE

- Usar el mismo estilo de celdas Markdown que TA-001 (con `>` para notas explicativas)
- Cada celda de código debe tener su celda Markdown explicativa arriba
- Los outputs deben quedar visibles (no limpiar outputs antes de entregar)
- El notebook debe poder ejecutarse de arriba a abajo sin errores
- Usar `tqdm` para barras de progreso (ya está en el entorno)
- Los checkpoints guardar en: `D:\UPAO_Diogho_Baylon\IX\Taller Integrador I\ModelosIA\modelos\checkpoints\`
- Los results JSON guardar junto al notebook en `notebooks\`
