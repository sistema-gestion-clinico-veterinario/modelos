from pathlib import Path
import fitz
import cv2
import numpy as np


def extraer_graficos(pdf_path: str, output_dir: str = None) -> list:
    pdf_path = Path(pdf_path)
    if output_dir is None:
        output_dir = pdf_path.parent / (pdf_path.stem + "_graficos")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rutas = []
    doc = fitz.open(str(pdf_path))

    for num_pag in range(len(doc)):
        pagina = doc[num_pag]
        mat = fitz.Matrix(2, 2)
        pix = pagina.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        _, bin = cv2.threshold(gris, 230, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((15, 15), np.uint8)
        dil = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)
        dil = cv2.dilate(dil, kernel, iterations=2)

        contornos, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = []
        for c in contornos:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            relacion = w / h
            if relacion > 3.0:
                continue
            if w < 150 or h < 120:
                continue
            if area < 5000:
                continue
            if area > 200000:
                continue
            rects.append((x, y, w, h))

        rects = _fusionar_rectangulos(rects)

        for idx, (x, y, w, h) in enumerate(rects):
            recorte = img[y:y+h, x:x+w]
            nombre = f"{pdf_path.stem}_pag{num_pag+1}_grafico{idx+1}.png"
            ruta = str(output_dir / nombre)
            cv2.imwrite(ruta, cv2.cvtColor(recorte, cv2.COLOR_RGB2BGR))
            rutas.append(ruta)

    doc.close()
    return rutas


def _fusionar_rectangulos(rects):
    if not rects:
        return []
    rects = sorted(rects, key=lambda r: (r[1], r[0]))
    fusionados = [rects[0]]
    for r in rects[1:]:
        u = fusionados[-1]
        x1, y1, w1, h1 = u
        x2, y2, w2, h2 = r
        if abs(y1 - y2) < 30 and x2 < x1 + w1 + 50:
            x_nuevo = min(x1, x2)
            y_nuevo = min(y1, y2)
            w_nuevo = max(x1 + w1, x2 + w2) - x_nuevo
            h_nuevo = max(y1 + h1, y2 + h2) - y_nuevo
            fusionados[-1] = (x_nuevo, y_nuevo, w_nuevo, h_nuevo)
        else:
            fusionados.append(r)
    return fusionados
