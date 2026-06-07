import json, sys
from pathlib import Path
from normalizar import normalizar
from main import procesar
from config import COLUMNAS

archivos = sys.argv[1:] if len(sys.argv) > 1 else [
    'data/pdfs/AKIRA-CHAVARRY-2026-05-14-0914.pdf',
    'data/pdfs/LUKI-CORRALES-2026-05-14-1724.pdf',
    'data/imagenes/23a07b2e-ec35-4d85-8599-9b9df839f9ec.jpg',
    'data/imagenes/5e39366e-9269-4181-98ea-9f6e4987312b.png',
]

with open('ground_truth/ground_truth.json', encoding='utf-8') as f:
    gt = json.load(f)

for archivo in archivos:
    nombre = Path(archivo).name
    df, tipo = procesar(archivo)
    gt_filas = gt.get(nombre, {}).get('rows', [])
    
    print(f'=== {nombre} ({tipo}) ===')
    print()
    
    aciertos = 0
    total = min(len(df), len(gt_filas))
    
    for i in range(total):
        fila_ext = df.iloc[i].to_dict()
        fila_gt = gt_filas[i]
        ok = True
        for col in COLUMNAS:
            if normalizar(fila_ext.get(col, '')) != normalizar(fila_gt.get(col, '')):
                ok = False
        marca = 'OK' if ok else 'XX'
        print(f'  {marca}  {fila_ext["item"]:30s} | {fila_ext["resultado"]:12s} | {fila_ext["flag"]}')
        if not ok:
            print(f'       GT: {fila_gt["item"]:30s} | {fila_gt["resultado"]:12s} | {fila_gt["flag"]}')
        if ok:
            aciertos += 1
    
    print(f'  -> {aciertos}/{total} correctas')
    print()
