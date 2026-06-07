import easyocr
import numpy as np
import pandas as pd
import sys
from config import OCR_IDIOMAS, OCR_GPU, OCR_CONFIANZA_MINIMA, COLUMNAS
from corrector import corregir_item, corregir_resultado, corregir_referencia, separar_resultado_referencia
from preprocesador import preprocesar_imagen
import cv2

reader = easyocr.Reader(OCR_IDIOMAS, gpu=OCR_GPU)


def extraer(ruta_imagen: str) -> pd.DataFrame:
    try:
        imagen = preprocesar_imagen(ruta_imagen)
    except Exception as e:
        print(f"  [ERROR] preprocesar_imagen falló: {e}", file=sys.stderr)
        return pd.DataFrame(columns=COLUMNAS)
    resultados = reader.readtext(imagen, detail=1, paragraph=False)
    return _estructurar_tabla(resultados)


def _estructurar_tabla(resultados):
    items_ocr = []
    for bbox, texto, conf in resultados:
        if conf <= OCR_CONFIANZA_MINIMA:
            continue
        y_centro = (bbox[0][1] + bbox[2][1]) / 2
        x_centro = (bbox[0][0] + bbox[2][0]) / 2
        alto = bbox[2][1] - bbox[0][1]
        items_ocr.append({"texto": texto.strip(), "y": y_centro, "x": x_centro, "h": alto})

    items_ocr.sort(key=lambda i: i["y"])

    filas = _agrupar_por_y(items_ocr)
    if len(filas) < 2:
        return pd.DataFrame(columns=COLUMNAS)

    zonas = _detectar_zonas(filas)
    datos = _mapear_filas(filas, zonas)

    df = pd.DataFrame(datos) if datos else pd.DataFrame(columns=COLUMNAS)
    return df[COLUMNAS]


def _agrupar_por_y(items):
    if not items:
        return []
    

    altos = [i["h"] for i in items]
    median_h = np.median(altos) if altos else 20.0
    items_filtrados = [i for i in items if i["h"] <= median_h * 2.5]
    if not items_filtrados:
        items_filtrados = items


    max_x = max(i["x"] for i in items_filtrados) if items_filtrados else 1000
    col_items = [i for i in items_filtrados if i["x"] < max_x * 0.35]
    col_results = [i for i in items_filtrados if max_x * 0.35 <= i["x"] < max_x * 0.55]
    col_refs = [i for i in items_filtrados if max_x * 0.55 <= i["x"] < max_x * 0.85]

    slopes = []

    for it in col_items:
        for res in col_results:
            dy = res["y"] - it["y"]
            dx = res["x"] - it["x"]
            if dx > 50 and abs(dy) < 40:
                slopes.append(dy / dx)


    for res in col_results:
        for ref in col_refs:
            dy = ref["y"] - res["y"]
            dx = ref["x"] - res["x"]
            if dx > 50 and abs(dy) < 40:
                slopes.append(dy / dx)

    estimated_slope = float(np.median(slopes)) if slopes else 0.0


    for item in items_filtrados:
        item["y_proj"] = item["y"] - estimated_slope * item["x"]


    items_filtrados.sort(key=lambda i: i["y_proj"])


    alto_promedio = float(np.median([i["h"] for i in items_filtrados])) if items_filtrados else 20.0
    tolerancia = alto_promedio * 0.4

    filas = []
    fila_actual = [items_filtrados[0]]
    for item in items_filtrados[1:]:
        if abs(item["y_proj"] - fila_actual[-1]["y_proj"]) <= tolerancia:
            fila_actual.append(item)
        else:
            filas.append(sorted(fila_actual, key=lambda i: i["x"]))
            fila_actual = [item]
    filas.append(sorted(fila_actual, key=lambda i: i["x"]))

    return filas


def _mayormente_upper(texto: str) -> bool:
    texto = texto.strip()
    if not 3 <= len(texto) <= 9:
        return False
    letters = [c for c in texto if c.isalpha()]
    if not letters:
        return False
    upper_count = sum(1 for c in letters if c.isupper())
    return upper_count / len(letters) >= 0.6


_HF = {
    "resultado": {"resultado", "csultado", "resuitado", "resutado", "esuiltalo", "resuhtado", "resuhrado", "resulrado"},
    "referencia": {"referencia", "referenci", "refereei", "referenc", "relerencia", "referenua", "referencis"},
    "nota": {"nota", "hota"},
    "item": {"item", "iten"},
}


def _detectar_zonas(filas):
    xs_all = [p["x"] for fila in filas for p in fila]
    if xs_all:
        max_x_img = max(xs_all)
        left_words = [
            p for fila in filas for p in fila
            if p["x"] < max_x_img * 0.3
            and 3 <= len(p["texto"].strip()) <= 9
            and _mayormente_upper(p["texto"])
        ]
        item_boundary = max(p["x"] for p in left_words) + 80 if left_words else None
    else:
        max_x_img = 600
        item_boundary = None

    header_xs = {"resultado": [], "referencia": [], "nota": [], "item": []}
    for fila in filas:
        for p in fila:
            t = p["texto"].lower().strip()
            for tipo, kws in _HF.items():
                if any(kw in t for kw in kws):
                    header_xs[tipo].append(p["x"])

    has_header = any(bool(v) for v in header_xs.values())
    if has_header:
        res_x = float(np.median(header_xs["resultado"])) if header_xs["resultado"] else None
        ref_x = float(np.median(header_xs["referencia"])) if header_xs["referencia"] else None
        nota_x = float(np.median(header_xs["nota"])) if header_xs["nota"] else None
        item_x = float(np.median(header_xs["item"])) if header_xs["item"] else None

        if item_x is not None:
            c0 = item_x + 40
        elif item_boundary is not None:
            c0 = item_boundary
        elif res_x is not None:
            c0 = res_x - 100
        else:
            c0 = max_x_img * 0.3

        if res_x is not None and ref_x is not None:
            c1 = (res_x + ref_x) / 2
        elif res_x is not None and nota_x is not None:
            c1 = (res_x + nota_x) / 2
        elif res_x is not None:
            c1 = res_x + 200
        elif ref_x is not None:
            c1 = ref_x - 100
        else:
            c1 = max_x_img * 0.55

        if ref_x is not None and nota_x is not None:
            c2 = (ref_x + nota_x) / 2
        elif nota_x is not None:
            c2 = nota_x + 100
        elif ref_x is not None:
            c2 = ref_x + 200
        else:
            c2 = max_x_img * 0.8

        cortes = sorted([c0, c1, c2])
        return [(0, cortes[0]), (cortes[0], cortes[1]), (cortes[1], cortes[2]), (cortes[2], 99999)]

    if item_boundary is not None:
        xs_todas = sorted(xs_all)
        gaps = [xs_todas[i + 1] - xs_todas[i] for i in range(len(xs_todas) - 1)]
        gap_indices = np.argsort(gaps)[-2:]
        gap_indices.sort()
        cortes = [item_boundary] + [(xs_todas[g] + xs_todas[g + 1]) / 2 for g in gap_indices]
        if len(cortes) < 3:
            cortes.extend([cortes[-1] + 200] * (3 - len(cortes)))
        cortes.sort()
        return [(0, cortes[0]), (cortes[0], cortes[1]), (cortes[1], cortes[2]), (cortes[2], 99999)]

    if len(xs_all) < 10:
        return _zonas_fallback(max_x_img)

    xs_todas = sorted(xs_all)
    gaps = [xs_todas[i + 1] - xs_todas[i] for i in range(len(xs_todas) - 1)]
    gap_indices = np.argsort(gaps)[-3:]
    gap_indices.sort()
    cortes = [(xs_todas[g] + xs_todas[g + 1]) / 2 for g in gap_indices]
    cortes.sort()
    return [(0, cortes[0]), (cortes[0], cortes[1]), (cortes[1], cortes[2]), (cortes[2], 99999)]


def _zonas_fallback(ancho_img):
    return [(0, ancho_img * 0.3), (ancho_img * 0.3, ancho_img * 0.55), (ancho_img * 0.55, ancho_img * 0.8), (ancho_img * 0.8, 99999)]


def _mapear_filas(filas, zonas):
    nombres_cols = ["item", "resultado", "referencia", "flag"]
    header_indice = -1
    header_kw_set = set().union(*_HF.values())
    for idx, fila in enumerate(filas):
        textos_lower = [p["texto"].lower().strip() for p in fila]
        if sum(1 for t in textos_lower if any(kw in t for kw in header_kw_set)) >= 1:
            header_indice = idx
            break

    datos = []
    lym_count = 0
    mid_count = 0
    gra_count = 0

    for idx, fila in enumerate(filas):
        if idx <= header_indice:
            continue
        palabras_col = {col: [] for col in nombres_cols}
        for palabra in fila:
            for i, (x_min, x_max) in enumerate(zonas):
                if x_min <= palabra["x"] < x_max and i < len(nombres_cols):
                    palabras_col[nombres_cols[i]].append(palabra["texto"])
                    break
        fila_dict = {col: " ".join(palabras_col[col]).strip() for col in nombres_cols}
        if fila_dict["item"]:
            item_raw = fila_dict["item"]
            item_clean = corregir_item(item_raw)
            if (not item_clean or item_clean == "") and any(x in item_raw.upper() for x in ["ALT", "AIT", "AST"]):
                ref_raw = fila_dict["referencia"].replace(" ", "").replace(",", ".")
                if "200" in ref_raw or "500" in ref_raw or "460" in ref_raw:
                    item_clean = "PLT"
            
           
            item_upper = item_clean.upper()
            if any(x in item_upper for x in ["LYM", "LYN", "LXM", "LYK", "ZXH", "LY"]):
                lym_count += 1
                fila_dict["item"] = "LYM%" if lym_count > 1 else "LYM#"
            elif any(x in item_upper for x in ["MID", "HID", "NID", "KID", "KLD"]):
                mid_count += 1
                fila_dict["item"] = "MID%" if mid_count > 1 else "MID#"
            elif any(x in item_upper for x in ["GRA", "GRD", "GRT", "GRK", "GR"]):
                gra_count += 1
                fila_dict["item"] = "GRA%" if gra_count > 1 else "GRA#"
            else:
                fila_dict["item"] = item_clean
                
            fila_dict["resultado"] = corregir_resultado(fila_dict["resultado"])
            fila_dict["referencia"] = corregir_referencia(fila_dict["referencia"])

            if not fila_dict["resultado"] and fila_dict["referencia"]:
                fila_dict["resultado"], fila_dict["referencia"] = separar_resultado_referencia(
                    fila_dict["referencia"], ""
                )
            else:
                fila_dict["resultado"], fila_dict["referencia"] = separar_resultado_referencia(
                    fila_dict["resultado"], fila_dict["referencia"]
                )
            datos.append(fila_dict)
    return datos
