from extractor_ocr import _agrupar_por_y, _detectar_zonas, _mapear_filas
from preprocesador import preprocesar_imagen
import easyocr

reader = easyocr.Reader(['es', 'en'], gpu=False)
imagen = preprocesar_imagen('data/imagenes/acb4153f-11e2-4f79-b6d9-ffd9ff466ed0.jpg')
resultados = reader.readtext(imagen, detail=1, paragraph=False)

items_ocr = []
for bbox, texto, conf in resultados:
    if conf <= 0.01: 
        continue
    y_centro = (bbox[0][1] + bbox[2][1]) / 2
    x_centro = (bbox[0][0] + bbox[2][0]) / 2
    alto = bbox[2][1] - bbox[0][1]
    items_ocr.append({'texto': texto.strip(), 'y': y_centro, 'x': x_centro, 'h': alto})

items_ocr.sort(key=lambda i: i['y'])
print(f'Total items: {len(items_ocr)}')

header = [i for i in items_ocr if i['texto'].lower() in ['iten', 'resultado', 'referencia', 'nota']]
print(f'Header words: {[(i["texto"], round(i["x"]), round(i["y"])) for i in header]}')

filas = _agrupar_por_y(items_ocr)
print(f'Filas agrupadas: {len(filas)}')
for idx, f in enumerate(filas[:8]):
    textos = [(p['texto'], round(p['x']), round(p['y'])) for p in f]
    print(f'  fila {idx}: {textos}')
print(f'  ...')
for idx, f in enumerate(filas[15:25]):
    textos = [(p['texto'], round(p['x']), round(p['y'])) for p in f]
    print(f'  fila {idx+15}: {textos}')

datos = _mapear_filas(filas, _detectar_zonas(filas))
print(f'\nDatos mapeados: {len(datos)}')
for d in datos[:5]:
    print(f'  {d}')
