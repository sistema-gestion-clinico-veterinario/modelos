import easyocr
import numpy as np
import pandas as pd
from config import OCR_IDIOMAS, OCR_GPU, OCR_CONFIANZA_MINIMA, COLUMNAS
from corrector import corregir_item, corregir_resultado, corregir_referencia, separar_resultado_referencia
from preprocesador import preprocesar_imagen
import extractor_ocr

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

items_ocr.sort(key=lambda i: i["y"])
filas = extractor_ocr._agrupar_por_y(items_ocr)

print(f"Total items_ocr: {len(items_ocr)}")
print(f"Total filas grouped: {len(filas)}")

for idx, f in enumerate(filas):
    textos = [p["texto"] for p in f]
    print(f"Fila {idx}: {textos}")
