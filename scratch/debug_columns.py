"""
The real problem: The Spike image is highly tilted AND each hemogram row
consists of an item label on the FAR LEFT (x~130-230) while the result
value and reference are in the CENTER (x~280-530, x~480-660).

Because of the tilt (slope -0.088), each successive row's tokens have a
large delta-y vs the item label. When we use projected Y, most items from
different rows end up in the same bin because the image is so wide and tilted.

We need to switch strategy: instead of projected-Y grouping, use the
ITEM column items as row anchors and pair each item to the closest
result/reference tokens horizontally (within the same y-window).
"""
import sys
sys.path.append('.')
import numpy as np
import easyocr
from preprocesador import preprocesar_imagen
from config import OCR_IDIOMAS, OCR_GPU, OCR_CONFIANZA_MINIMA
from corrector import corregir_item

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
median_h = np.median(altos)
items = [i for i in items_ocr if i["h"] <= median_h * 2.5]

print(f"Filtered items: {len(items)}")


max_x = max(i["x"] for i in items)
print(f"max_x: {max_x}")

item_col = [i for i in items if i["x"] < max_x * 0.30]
result_col = [i for i in items if max_x * 0.30 <= i["x"] < max_x * 0.62]
ref_col = [i for i in items if max_x * 0.62 <= i["x"] < max_x * 0.90]

print(f"\nItem column ({len(item_col)} tokens):")
for i in sorted(item_col, key=lambda x: x['y']):
    print(f"  y={i['y']:.0f} x={i['x']:.0f} text='{i['texto']}'")

print(f"\nResult column ({len(result_col)} tokens):")
for i in sorted(result_col, key=lambda x: x['y']):
    print(f"  y={i['y']:.0f} x={i['x']:.0f} text='{i['texto']}'")

print(f"\nRef column ({len(ref_col)} tokens):")
for i in sorted(ref_col, key=lambda x: x['y']):
    print(f"  y={i['y']:.0f} x={i['x']:.0f} text='{i['texto']}'")
