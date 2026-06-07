"""
Spike image analysis shows:
- The image is tilted ~5 degrees clockwise
- Item column x range: ~130-270
- Result column x range: ~300-520 (items + values merged due to binder on left side of image being wider)
- Ref column x range: ~480-680

The key problem is that EasyOCR is merging result+reference tokens into single blocks
because the tilt makes them seem horizontally adjacent (e.g. "15,28 1091 6-17" = result + ref merged).

Strategy:
1. Use the item column (x<300) as row anchors, sorted by y.
2. For each item row, find all tokens in result range (300-560) and ref range (560-750)
   within a vertical window (based on median row height * 1.5).
3. The corrector.separar_resultado_referencia already handles merged result+ref strings.
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

print(f"Item col: {len(item_col)}, Result col: {len(result_col)}, Ref col: {len(ref_col)}")


Y_WINDOW = median_h * 1.5
rows = []
for item_tok in item_col:
    item_y = item_tok["y"]
    res_tokens = [t for t in result_col if abs(t["y"] - item_y) <= Y_WINDOW]
    ref_tokens = [t for t in ref_col if abs(t["y"] - item_y) <= Y_WINDOW]
    
    resultado_str = " ".join(t["texto"] for t in sorted(res_tokens, key=lambda t: t["x"]))
    referencia_str = " ".join(t["texto"] for t in sorted(ref_tokens, key=lambda t: t["x"]))
    
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
        })

print(f"\nRows parsed: {len(rows)}")
for r in rows:
    print(f"  [{r['raw_item']}] -> item='{r['item']}' resultado='{r['resultado']}' referencia='{r['referencia']}'")
