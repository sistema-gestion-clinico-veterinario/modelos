# CONTEXT — Sistema Multimodal de IA para Apoyo Diagnóstico Veterinario
> Archivo de contexto para Claude Code. Leer completo antes de cualquier tarea.

---

## 1. DESCRIPCIÓN DEL PROYECTO

**Nombre:** Sistema Multimodal para la Asistencia en el Diagnóstico Clínico Veterinario  
**Curso:** Taller Integrador I — UPAO (Universidad Privada Antenor Orrego), Trujillo, Perú  
**Caso piloto:** Vargas Vet (clínica veterinaria privada, Trujillo)  
**Integrantes:**
- Baylón Toledo, Diogho Matteo — Project Manager
- Saavedra Arroyo, Sebastián Alonso — Scrum Master

**Docente evaluador:** Walter Cueva Chávez

**Propósito:** Sistema web con módulo de IA que ayuda al veterinario a interpretar exámenes complementarios (radiografías torácicas + hemogramas) de perros y gatos. **No emite diagnósticos definitivos**, solo recomendaciones de apoyo.

---

## 2. ARQUITECTURA DEL SISTEMA

```
Frontend (Angular) → Vercel
      ↓ HTTP
Backend (Spring Boot Java) → Render
      ↓ HTTP interno
Microservicio IA (FastAPI Python) → Render
      ↓
  ┌───────────────────────────────────┐
  │  Modelo CNN (radiografías)        │
  │  Módulo hemogramas (pdfplumber    │
  │    + EasyOCR)                     │
  │  Motor LLM (API OpenAI GPT)       │
  └───────────────────────────────────┘
Base de datos: PostgreSQL → Render
Entrenamiento: Local RTX 4060 8GB VRAM
```

**Stack completo:**
- Frontend: Angular → Vercel
- Backend: Spring Boot (Java) → Render
- Microservicio IA: FastAPI (Python) → Render
- BD: PostgreSQL → Render
- Entrenamiento local: PyTorch 2.12.0+cu130, Python 3.12.10, RTX 4060

---

## 3. REPOSITORIOS GITHUB

Organización: `sistema-gestion-clinico-veterinario`

| Repo | Contenido | URL |
|------|-----------|-----|
| `modelos` | Notebooks + código IA (CNN, hemogramas, LLM) | https://github.com/sistema-gestion-clinico-veterinario/modelos.git |

**Estructura del repo `modelos`:**
```
modelos/
├── notebooks/
│   ├── TA-001_VetXRay_Dataset.ipynb      ✅ COMPLETADO
│   ├── TA-003_ResNet50_Benchmarking.ipynb
│   ├── TA-004_EfficientNet_DenseNet.ipynb
│   └── TA-005_Modelo_Ganador.ipynb
├── src/
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── hemograma/
│       ├── extractor_pdf.py
│       └── extractor_ocr.py
├── configs/
│   └── base_config.yaml
├── .gitignore
└── README.md
```

---

## 4. RUTAS LOCALES (máquina de desarrollo)

```
E:\Taller Integrador I\ModelosIA\
├── Hemogramas\          ← hemogramas PDF de clínica colaboradora
├── modelos\             ← repo GitHub clonado (AQUÍ SE TRABAJA)
└── Radiografias\        ← dataset completo (~45 GB, NO se sube a GitHub)
    ├── RX_1\RX-1\       ← radiografías DICOM originales
    ├── RX_2\RX-2\
    ├── RX_3\RX-3\
    ├── RX_4\RX-4\
    ├── RX_5\RX-5\
    ├── dataset_split\   ← generado por TA-001
    │   ├── train\       ← ~4,840 imágenes .dcm
    │   ├── val\         ← ~1,038 imágenes .dcm
    │   ├── test\        ← ~1,039 imágenes .dcm
    │   └── manifests\
    │       ├── train.csv
    │       ├── val.csv
    │       ├── test.csv
    │       └── dataset_completo.csv
    └── File list with tags.xlsx
```
> **NOTA:** Migrado de D: a E: (nuevo SSD Kingston 1TB) el 06 Jun 2026. La ruta antigua `D:\UPAO_Diogho_Baylon\IX\...` ya no existe.

---

## 5. ESTADO ACTUAL DEL PROYECTO (06 Jun 2026)

### Sprint 1 — COMPLETADO
| PBI | Descripción | Estado |
|-----|-------------|--------|
| TA-001 | Descargar y estructurar dataset VetXRay | ✅ DONE |
| DO-001 | Reporte dataset VetXRay | ✅ DONE |
| SP-004 | Investigación arquitecturas CNN | ✅ DONE |

### Sprint 2 — Benchmarking CNN (COMPLETADO — revisado por docente hasta DO-003)
| PBI | Descripción | Estado |
|-----|-------------|--------|
| TA-003 | ResNet-50 benchmarking | ✅ DONE — AUC=0.8619 |
| TA-004 | EfficientNet-B0 + DenseNet-121 benchmarking | ✅ DONE — AUC=0.8621 / 0.8630 |
| DO-002 | Reporte benchmarking CNN | ✅ DONE |
| TA-005 | DenseNet-121 modelo ganador | ✅ DONE — AUC test=0.8717 |
| TA-005B | Optimización de umbrales | ✅ DONE — modelo_final_results.json |
| DO-003 | Informe entrenamiento CNN final | ✅ DONE |

### Sprint 2 — Fase 2 (feedback docente 02 Jun 2026 — EN CURSO)
| PBI | Descripción | Depende de |
|-----|-------------|------------|
| W&B Setup | Configurar Weights & Biases para monitoreo en tiempo real | — |
| TA-005D | GAN augmentation para clases minoritarias | W&B Setup |
| TA-003B | ResNet-50 reentrenado con augmented + W&B | TA-005D |
| TA-004B | EfficientNet-B0 reentrenado con augmented + W&B | TA-005D |
| TA-005E | DenseNet-121 reentrenado con augmented + W&B | TA-005D |
| TA-005F | Entrenamiento final del ganador con augmented | TA-003B/004B/005E |
| TA-005G | Optimización de umbrales del ganador final | TA-005F |
| TA-005C | Grad-CAM + muestreo aleatorio por clase | TA-005F |
| TA-006 | Endpoint FastAPI /predict/radiografia | TA-005G |
| TA-007 | Módulo extracción hemogramas | SP-002 |
| TA-008 | Endpoint FastAPI hemogramas | TA-007 |
| EN-006 | Integrar LLM en FastAPI | SP-003 |

---

## 6. DATASET — RADIOGRAFÍAS (resultado de TA-001)

| Parámetro | Valor |
|-----------|-------|
| Fuente | VetXRay (Zenodo) — 9,882 imágenes DICOM |
| Filtro | Quality = 'correct' |
| Total tras filtro | 6,917 imágenes |
| Pacientes únicos | 1,647 |
| Split | 70/15/15 (train/val/test) por paciente, seed=42 |
| Train | ~4,840 imágenes |
| Val | ~1,038 imágenes |
| Test | ~1,039 imágenes |

**Distribución de hallazgos (TAGs):**
| Tag | % |
|-----|---|
| no_finding | 50.0% |
| cardiomegaly | 22.8% |
| alveolar_pattern | 13.4% |
| bronchial_pattern | ~7% |
| pleural_effusion | ~4% |
| otros (excluir) | <2% cada uno |

**Tags a EXCLUIR del entrenamiento** (insuficiente estadística):
- `fracture` (8 muestras)
- `pneumomediastinum` (6 muestras)
- `costal_fracture` (1 muestra)
- `exclude` (44 registros)

**Clases objetivo para TA-005 (O4 del Project Charter):**
1. `alveolar_pattern`
2. `bronchial_pattern`
3. `pleural_effusion`
4. `cardiomegaly`
5. `no_finding`

**Distribución por especie:**
- Dog: 78.5%
- Cat: 20.8%
- Otros: 0.7%

**Tipo de clasificación:** Multi-etiqueta (una imagen puede tener múltiples TAGs simultáneos)

---

## 7. MODELOS CNN — BENCHMARKING (TA-003 y TA-004)

### Hiperparámetros BASE (idénticos para los 3 modelos — NO modificar para comparación justa)

| Hiperparámetro | Valor |
|----------------|-------|
| Input size | 224 × 224 px |
| Canales entrada | 3 (replicar canal gris × 3) |
| Batch size | 32 (DenseNet-121: 16-32) |
| Learning rate inicial | 1e-4 |
| Optimizador | AdamW |
| Función de pérdida | BCEWithLogitsLoss + class weights |
| Épocas máximas | 50 |
| Early stopping patience | 10 |
| Transfer learning | Feature extraction 5 ep → fine-tuning completo |
| Scheduler ResNet/DenseNet | ReduceLROnPlateau |
| Scheduler EfficientNet | CosineAnnealingWarmRestarts (T_0=10, T_mult=2) |
| Weight decay | 1e-4 |
| Dropout clasificador | 0.4–0.5 |
| Mixed precision | torch.cuda.amp (recomendado) |

### Modelos a entrenar

| Modelo | Params | VRAM estimada | AUC esperado |
|--------|--------|---------------|--------------|
| ResNet-50 | 25.6M | ~2.5-3.5 GB | 0.72-0.80 |
| EfficientNet-B0 | 5.3M | ~1.2-1.8 GB | 0.68-0.78 |
| DenseNet-121 | 8.0M | ~1.8-2.5 GB | 0.76-0.84 |

**Predicción del ganador:** DenseNet-121 (por historial en CheXNet — radiografías torácicas)

### Métricas a registrar (para DO-002)
- AUC (métrica principal — umbral mínimo: AUC ≥ 0.75)
- F1-score macro (umbral mínimo: F1 ≥ 0.75 en test)
- Accuracy global
- Tiempo de inferencia promedio por imagen (segundos)
- Curvas loss/AUC por época

---

## 8. MÓDULO HEMOGRAMAS

### Dos rutas automáticas
1. **PDF digital** (texto extraíble) → `pdfplumber`
2. **Imagen / voucher físico / PDF escaneado** → `EasyOCR`

### Datos disponibles
- 1,000 PDFs formato IDEXX (clínica colaboradora — hermano Saavedra)
- Vouchers físicos de Vargas Vet
- Ruta local: `D:\UPAO_Diogho_Baylon\IX\Taller Integrador I\ModelosIA\Hemogramas\`

### Tabla de rangos de referencia (respaldo cuando no hay flags H/L)
Fuentes consolidadas:
- Lab for Vets (labforvets.com)
- Manual MSD/Merck de Veterinaria

Especies: **Solo perros y gatos** (fuera de alcance: bovinos, equinos, ovinos)

**Parámetros clave a extraer:**
- Eritrocitos, Hemoglobina, Hematocrito, VCM, CHCM, Plaquetas (eritrograma)
- Leucocitos totales, Neutrófilos, Bandas, Eosinófilos, Basófilos, Monocitos, Linfocitos (leucograma)
- Enzimas: ALT, AST, Fosfatasa alcalina, GGT
- Metabolitos: Glucosa, Creatinina, Urea, Proteínas Totales, Albumina, Bilirrubina

---

## 9. MOTOR LLM

**API:** OpenAI GPT (no se desarrolla motor propio)  
**Endpoint FastAPI:** `POST /recomendacion`

### 4 Escenarios clínicos
| Escenario | Entradas |
|-----------|---------|
| 1 | Solo Historia Clínica |
| 2 | HC + Radiografía |
| 3 | HC + Hemograma |
| 4 | HC + Radiografía + Hemograma |

**Requisito obligatorio en todo output:** `"Recomendación de apoyo, no reemplaza el criterio profesional del veterinario"`

---

## 10. ENDPOINTS FASTAPI (microservicio IA)

| Endpoint | Método | Descripción | Tiempo máx. |
|----------|--------|-------------|-------------|
| `/predict/radiografia` | POST | Recibe .dcm/.jpg, retorna hallazgos CNN + probabilidades | < 6 seg |
| `/predict/hemograma` | POST | Recibe .pdf/.jpg/.png, retorna lista de alteraciones | < 5 seg |
| `/recomendacion` | POST | Recibe contexto clínico, retorna recomendación GPT | < 10 seg |

---

## 11. CRITERIOS DE ACEPTACIÓN CLAVE

| Objetivo | Métrica | Umbral |
|----------|---------|--------|
| O3 — Benchmarking CNN | AUC en validación | ≥ 0.75 (al menos 1 modelo) |
| O4 — Modelo ganador | AUC en validación | ≥ 0.80 |
| O4 — Modelo ganador | F1-score en test | ≥ 0.75 |
| O4 — Inferencia CNN | Tiempo por imagen | < 6 seg |
| O5 — Hemogramas | Tasa extracción correcta | ≥ 90% |
| O5 — Hemogramas | Detección alteraciones | ≥ 80% |
| O6 — LLM | Tiempo de respuesta | < 10 seg |
| O7 — Validación clínica | Concordancia con veterinario | ≥ 70% |
| O7 — Validación clínica | Usabilidad Likert | ≥ 3.5/5 |
| O7 — Casos validados | N° casos con Vargas Vet | ≥ 15 |

---

## 12. DECISIONES TÉCNICAS TOMADAS (NO cambiar sin consenso)

1. **Clasificación multi-etiqueta** con `BCEWithLogitsLoss` + class weights (no softmax)
2. **Split a nivel de paciente** (no imagen) para evitar data leakage
3. **Radiografías DICOM → pseudo-RGB** replicando canal gris × 3
4. **Preentrenamiento ImageNet** para los 3 modelos del benchmarking (condiciones iguales)
5. **DenseNet-121 puede usar pesos CheXNet** en TA-005 (entrenamiento del ganador)
6. **Tags excluidos**: fracture, pneumomediastinum, costal_fracture, exclude
7. **Mixed precision** (torch.cuda.amp) recomendado para RTX 4060
8. **Gradient checkpointing** para DenseNet-121 si hay problemas de VRAM
9. **pdfplumber** para PDFs digitales, **EasyOCR** para imágenes/vouchers
10. **API OpenAI GPT** integrada en FastAPI (no motor propio)

---

## 13. PRÓXIMA TAREA A EJECUTAR: TA-003

Crear `notebooks/TA-003_ResNet50_Benchmarking.ipynb` con:

1. Cargar manifests desde `dataset_split/manifests/train.csv`, `val.csv`, `test.csv`
2. Implementar `VetXRayDataset` (PyTorch Dataset) que:
   - Lee archivos .dcm con `pydicom`
   - Normaliza con percentil 2-98
   - Maneja MONOCHROME1
   - Replica canal gris → 3 canales (pseudo-RGB)
   - Aplica transform 224×224
3. Calcular class weights desde train.csv
4. Configurar ResNet-50 con transfer learning:
   - Feature extraction 5 épocas (solo cabeza)
   - Fine-tuning completo épocas restantes
   - Dropout 0.4 en clasificador
5. Training loop con:
   - BCEWithLogitsLoss + class weights
   - AdamW lr=1e-4, weight decay=1e-4
   - ReduceLROnPlateau scheduler
   - Early stopping patience=10
   - Mixed precision (torch.cuda.amp)
6. Evaluación: AUC, F1-macro, Accuracy, tiempo inferencia
7. Guardar métricas en formato estandarizado para DO-002
8. Guardar checkpoint `.pth`

**Tags a usar (5 clases):**
```python
CLASSES = ['alveolar_pattern', 'bronchial_pattern', 'pleural_effusion', 
           'cardiomegaly', 'no_finding']
```

---

## 14. DEPENDENCIAS PYTHON (entorno local)

```bash
# Ya instalado y verificado
torch==2.12.0+cu130
torchvision

# Dataset
pydicom
pandas
openpyxl
numpy
scikit-learn
matplotlib
seaborn

# Hemogramas (Sprint 2)
pdfplumber
easyocr
Pillow

# API (Sprint 2)
fastapi
uvicorn
python-multipart
openai

# Config
pyyaml
tqdm
```

---

## 15. NOTAS IMPORTANTES PARA CLAUDE CODE

- **El dataset NO está en el repo**. Está local en `D:\UPAO_Diogho_Baylon\IX\Taller Integrador I\ModelosIA\Radiografias\`
- **Los manifests CSV** son el punto de entrada para todos los notebooks de entrenamiento
- **seed=42 siempre** — no cambiar para mantener reproducibilidad
- **Los 3 modelos deben usar EXACTAMENTE los mismos hiperparámetros** en el benchmarking
- **Variaciones de hiperparámetros** solo se permiten en TA-005 (modelo ganador)
- **El repo `modelos`** solo tiene código — los datos van en `.gitignore`
- **Convención de nombres archivos:** `TA-XXX_Nombre.ipynb`, `SP-XXX_Nombre.docx`, `DO-XXX_Nombre.docx`
