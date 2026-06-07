import sys
sys.path.append('.')
import easyocr
from preprocesador import preprocesar_imagen
from config import OCR_IDIOMAS, OCR_GPU, OCR_CONFIANZA_MINIMA
import extractor_ocr
import numpy as np

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

print(f"Total items_ocr passed to _agrupar_por_y: {len(items_ocr)}")
filas = extractor_ocr._agrupar_por_y(items_ocr)
print(f"Total filas after _agrupar_por_y: {len(filas)}")
for i, f in enumerate(filas):
    print(f"  Row {i}: {[p['texto'] for p in f]}")

zonas = extractor_ocr._detectar_zonas(filas)
print(f"\nZonas detectadas: {zonas}")

datos = extractor_ocr._mapear_filas(filas, zonas)
print(f"\nDatos mapeados: {len(datos)} filas")
for d in datos:
    print(f"  {d}")
