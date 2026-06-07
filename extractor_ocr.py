import easyocr
import numpy as np
import pandas as pd
from config import OCR_IDIOMAS, OCR_GPU, OCR_CONFIANZA_MINIMA, COLUMNAS
from corrector import corregir_item, corregir_resultado, corregir_referencia, separar_resultado_referencia
from preprocesador import preprocesar_imagen
import cv2

reader = easyocr.Reader(OCR_IDIOMAS, gpu=OCR_GPU)


def extraer(ruta_imagen: str) -> pd.DataFrame:
    try:
        imagen = preprocesar_imagen(ruta_imagen)
    except Exception:
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
    alto_promedio = float(np.mean(altos)) if altos else 20.0
    tolerancia = alto_promedio * 0.5

    filas = []
    fila_actual = [items[0]]
    for item in items[1:]:
        if abs(item["y"] - fila_actual[-1]["y"]) <= tolerancia:
            fila_actual.append(item)
        else:
            filas.append(sorted(fila_actual, key=lambda i: i["x"]))
            fila_actual = [item]
    filas.append(sorted(fila_actual, key=lambda i: i["x"]))

    filas = _separar_filas_por_altura(filas)
    return filas


def _separar_filas_por_altura(filas):
    resultado = []
    for fila in filas:
        if len(fila) < 2:
            resultado.append(fila)
            continue
        altos = np.array([p["h"] for p in fila])
        if max(altos) > min(altos) * 1.8:
            grupo_alto = [p for p in fila if p["h"] > np.median(altos) * 1.5]
            grupo_normal = [p for p in fila if p["h"] <= np.median(altos) * 1.5]
            if grupo_alto and grupo_normal:
                resultado.append(grupo_normal)
                resultado.append(grupo_alto)
            else:
                resultado.append(fila)
        else:
            resultado.append(fila)
    return resultado


def _detectar_zonas(filas):
    header_keywords = {"item", "resultado", "referencia", "nota", "iten", "resuitado"}
    for fila in filas:
        textos = [p["texto"].lower().strip() for p in fila]
        match_count = sum(1 for t in textos if t in header_keywords)
        if match_count >= 2 and len(fila) >= 4:
            xs = sorted([p["x"] for p in fila])
            cortes = [
                (xs[0] + xs[1]) / 2,
                (xs[1] + xs[2]) / 2,
                (xs[2] + xs[3]) / 2 if len(xs) > 3 else xs[2] + 80,
            ]
            return [(0, cortes[0]), (cortes[0], cortes[1]), (cortes[1], cortes[2]), (cortes[2], 99999)]

    xs_todas = sorted([p["x"] for fila in filas for p in fila])
    if len(xs_todas) < 10:
        return _zonas_fallback(600)

    gaps = [xs_todas[i + 1] - xs_todas[i] for i in range(len(xs_todas) - 1)]
    gap_indices = np.argsort(gaps)[-3:]
    gap_indices.sort()
    cortes = [(xs_todas[g] + xs_todas[g + 1]) / 2 for g in gap_indices]

    return [(0, cortes[0]), (cortes[0], cortes[1]), (cortes[1], cortes[2]), (cortes[2], 99999)]


def _zonas_fallback(ancho_est):
    zona = ancho_est / 4
    return [(0, zona), (zona, zona * 2), (zona * 2, zona * 3), (zona * 3, 99999)]


def _mapear_filas(filas, zonas):
    nombres_cols = ["item", "resultado", "referencia", "flag"]
    header_indice = -1
    for idx, fila in enumerate(filas):
        textos_lower = [p["texto"].lower().strip() for p in fila]
        if sum(1 for t in textos_lower if t in {"item", "resultado", "referencia", "nota", "iten", "resuitado"}) >= 2:
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

            fila_dict["resultado"], fila_dict["referencia"] = separar_resultado_referencia(
                fila_dict["resultado"], fila_dict["referencia"]
            )
            datos.append(fila_dict)
    return datos
