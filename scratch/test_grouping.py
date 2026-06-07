import easyocr
import numpy as np
from preprocesador import preprocesar_imagen
from config import OCR_IDIOMAS, OCR_GPU

ruta_imagen = 'data/imagenes/spike_vargas_vet.jpg'
imagen = preprocesar_imagen(ruta_imagen)
reader = easyocr.Reader(OCR_IDIOMAS, gpu=OCR_GPU)
resultados = reader.readtext(imagen, detail=1, paragraph=False)

items_ocr = []
for bbox, texto, conf in resultados:
    if conf <= 0.01:
        continue
    y_centro = (bbox[0][1] + bbox[2][1]) / 2
    x_centro = (bbox[0][0] + bbox[2][0]) / 2
    alto = bbox[2][1] - bbox[0][1]
    items_ocr.append({"texto": texto.strip(), "y": y_centro, "x": x_centro, "h": alto})

items_ocr.sort(key=lambda i: i["y"])


altos = [i["h"] for i in items_ocr]
median_h = np.median(altos)
items_filtrados = [i for i in items_ocr if i["h"] <= median_h * 2.5]

print(f"Original items: {len(items_ocr)}")
print(f"Filtered items: {len(items_filtrados)}")


altos_filtrados = [i["h"] for i in items_filtrados]
alto_promedio = float(np.median(altos_filtrados))
tolerancia = alto_promedio * 0.4

filas = []
fila_actual = [items_filtrados[0]]
for item in items_filtrados[1:]:
    if abs(item["y"] - fila_actual[-1]["y"]) <= tolerancia:
        fila_actual.append(item)
    else:
        filas.append(sorted(fila_actual, key=lambda i: i["x"]))
        fila_actual = [item]
filas.append(sorted(fila_actual, key=lambda i: i["x"]))

print(f"Total rows: {len(filas)}")
for idx, f in enumerate(filas):
    textos = [p["texto"] for p in f]
    print(f"Row {idx}: {textos}")
