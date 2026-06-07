"""
New strategy: Y-window pairing with tilt correction.
For each item-column token, project neighboring result/ref tokens
using the estimated tilt slope to find closest row mates.
Then reduce the Y_WINDOW to prevent cross-row contamination.
"""
import sys
sys.path.append('.')
import numpy as np
import easyocr
from preprocesador import preprocesar_imagen
from config import OCR_IDIOMAS, OCR_GPU, OCR_CONFIANZA_MINIMA
from corrector import corregir_item, corregir_resultado, corregir_referencia, separar_resultado_referencia

ruta_imagen = 'data/imagenes/spike_vargas_vet.jpg'
imagen = preprocesar_imagen(ruta_imagen)
reader = easyocr.Reader(OCR_IDIOMAS, gpu=OCR_GPU)
resultados = reader.readtext(imagen, detail=1, paragraph=False)

items_ocr = []
for bbox, texto, conf in resultados:
    if conf <= OCR_CONFIANZA_MINIMA:
        continue
    y_centro = (bbox[0][1] + bbox[2][1]) / 2
    x_centro = (bbox[0][0] + bbox[2][0]) / 2
    alto = bbox[2][1] - bbox[0][1]
    items_ocr.append({"texto": texto.strip(), "y": y_centro, "x": x_centro, "h": alto})

# Filter outlier heights
altos = [i["h"] for i in items_ocr]
median_h = float(np.median(altos))
items = [i for i in items_ocr if i["h"] <= median_h * 2.5]
max_x = max(i["x"] for i in items)

ITEM_X_MAX = max_x * 0.30
RESULT_X_MIN = max_x * 0.30
RESULT_X_MAX = max_x * 0.62
REF_X_MIN = max_x * 0.62
REF_X_MAX = max_x * 0.88

item_col = sorted([i for i in items if i["x"] < ITEM_X_MAX], key=lambda i: i["y"])
result_col = sorted([i for i in items if RESULT_X_MIN <= i["x"] < RESULT_X_MAX], key=lambda i: i["y"])
ref_col = sorted([i for i in items if REF_X_MIN <= i["x"] < REF_X_MAX], key=lambda i: i["y"])

# Estimate slope from item-to-result pairings
slopes = []
for it in item_col:
    for res in result_col:
        dy = res["y"] - it["y"]
        dx = res["x"] - it["x"]
        if dx > 50 and abs(dy) < 40:
            slopes.append(dy / dx)

slope = float(np.median(slopes)) if slopes else 0.0
print(f"Slope: {slope:.4f}")

# Project Y for all tokens
for t in items:
    t["y_proj"] = t["y"] - slope * t["x"]

# Use projected Y for anchor pairing with a tight window
# Row height from just the item column
item_hs = [i["h"] for i in item_col]
row_h = float(np.median(item_hs)) if item_hs else 55.0
Y_WINDOW = row_h * 0.55  # tight window to prevent cross-row contamination
print(f"Y_WINDOW: {Y_WINDOW:.1f}")

rows = []
used_result = set()
used_ref = set()

for item_tok in item_col:
    item_py = item_tok["y_proj"]
    
    # Find result tokens close in projected Y
    res_matches = []
    for j, t in enumerate(result_col):
        if abs(t["y_proj"] - item_py) <= Y_WINDOW:
            res_matches.append((j, t))
    
    ref_matches = []
    for j, t in enumerate(ref_col):
        if abs(t["y_proj"] - item_py) <= Y_WINDOW:
            ref_matches.append((j, t))
    
    resultado_str = " ".join(t["texto"] for _, t in sorted(res_matches, key=lambda x: x[1]["x"]))
    referencia_str = " ".join(t["texto"] for _, t in sorted(ref_matches, key=lambda x: x[1]["x"]))
    
    item_clean = corregir_item(item_tok["texto"])
    resultado_clean = corregir_resultado(resultado_str)
    referencia_clean = corregir_referencia(referencia_str)
    resultado_clean, referencia_clean = separar_resultado_referencia(resultado_clean, referencia_clean)
    
    if item_clean:
        rows.append({
            "raw_item": item_tok["texto"],
            "item": item_clean,
            "resultado": resultado_clean,
            "referencia": referencia_clean,
            "y_proj": item_py,
        })

print(f"\nRows parsed: {len(rows)}")
for r in rows:
    print(f"  [{r['raw_item']}] -> item='{r['item']}' res='{r['resultado']}' ref='{r['referencia']}'")
