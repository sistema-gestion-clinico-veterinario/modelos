import re
import cv2
import numpy as np
import pdfplumber
import easyocr
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

SECTION_MARKERS = {
    'hematologia': re.compile(r'^\s*Hematology\s*$', re.I),
    'quimica'    : re.compile(r'^\s*Chemistry\s*$',  re.I),
    'serologia'  : re.compile(r'^\s*Serology\s*$',   re.I),
}

SKIP_TOKENS = {
    'test', 'result', 'reference', 'value', 'generated', 'vetconnect',
    'page', 'idexx', 'services', 'hematology', 'chemistry', 'serology',
    'download', 'date', 'pet', 'species', 'breed', 'gender', 'age',
    'patient', 'attending', 'lab', 'order', 'account',
    'laboratories', 'reserved', 'run', 'scatter'
}

SCATTER_ARTIFACTS = {'rbc run', 'wbc run', 'rbc_frag', 'urbc'}

COMMENT_KW = re.compile(
    r'(confirmar|detectad|agregar|anemia|monocitosis|linfocitosis|'
    r'leucocitosis|neutrofilia|trombocitopenia|considere|probable|'
    r'regenerativa|glucocorticoide|frotis|sanguíneo|notificado|'
    r'eritrocitosis|policitemia|leucopenia|eritropenia|hipoplasia|'
    r'hemoconcentración|hemodiluci)', re.I
)


def _val(s) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(str(s).replace(',', '.').strip())
    except Exception:
        return None


def _skip(linea: str) -> bool:
    t = linea.lower().strip()
    if not t:
        return True
    if t in SCATTER_ARTIFACTS:
        return True
    tok = t.split()[0]
    return tok in SKIP_TOKENS


def _extract_lines(ruta: str) -> List[str]:
    lineas = []
    with pdfplumber.open(ruta) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ''
            lineas.extend(txt.splitlines())
    return lineas


def _find_sections(lineas: List[str]) -> Dict[str, int]:
    found: Dict[str, int] = {}
    for i, l in enumerate(lineas):
        for name, pat in SECTION_MARKERS.items():
            if pat.match(l.strip()) and name not in found:
                found[name] = i
    return found


def _section_range(found: Dict[str, int], name: str, total: int) -> Tuple[int, int]:
    idx = found.get(name)
    if idx is None:
        return (0, 0)
    others = sorted(v for k, v in found.items() if k != name and v > idx)
    end = others[0] if others else total
    return (idx + 1, end)


RE_CABECERA = {
    'especie'    : re.compile(r'SPECIES\s*:\s*([A-Za-z]+)', re.I),
    'raza'       : re.compile(r'BREED\s*:\s*(.+?)(?:\s{2,}|$)', re.I),
    'sexo'       : re.compile(r'GENDER\s*:\s*(.+?)(?:\s{2,}|$)', re.I),
    'edad'       : re.compile(r'AGE\s*:\s*(.+?)(?:\s{2,}|$)', re.I),
    'fecha'      : re.compile(r'DATE OF RESULT\s*:\s*(.+?)(?:\s{2,}|$)', re.I),
    'analizadores': re.compile(r'IDEXX Services\s*:\s*(.+?)(?:\s{2,}|$)', re.I),
}


def extraer_cabecera(lineas: List[str]) -> Dict[str, str]:
    cab: Dict[str, str] = {}
    for l in lineas[:50]:
        for key, pat in RE_CABECERA.items():
            m = pat.search(l)
            if m and key not in cab:
                cab[key] = m.group(1).strip()
    return cab


RE_HEMA = re.compile(
    r'^(?P<item>(?:%\s+)?[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s:]{0,40}?)\s+'
    r'\*?\s*(?P<valor>\d+[\.,]?\d*)\s*'
    r'(?:(?P<ref_low>\d+[\.,]?\d*)\s*[-–]\s*(?P<ref_high>\d+[\.,]?\d*)\s*'
    r'(?P<unidad>[^\s\dHL][^\s]{0,14})\s*)?'
    r'(?P<flag>[HL])?\s*$'
)


def extraer_hematologia(lineas: List[str], ini: int, fin: int) -> Dict:
    params: List[Dict] = []
    analizador = ''
    for linea in lineas[ini:fin]:
        l = linea.strip()
        if not l or _skip(l):
            continue
        if 'ProCyte' in l or 'procyte' in l.lower():
            analizador = l
            continue
        m = RE_HEMA.match(l)
        if not m:
            continue
        item = m.group('item').strip().lstrip('% ').strip()
        if not item or item.lower().split()[0] in SKIP_TOKENS:
            continue
        val = _val(m.group('valor'))
        if val is None:
            continue
        ref_lo = _val(m.group('ref_low'))
        ref_hi = _val(m.group('ref_high'))
        flag   = m.group('flag')
        if flag == 'H' or (ref_hi is not None and val > ref_hi):
            estado = 'alto'
        elif flag == 'L' or (ref_lo is not None and val < ref_lo):
            estado = 'bajo'
        else:
            estado = 'normal'
        params.append({
            'test'   : item,
            'valor'  : val,
            'unidad' : (m.group('unidad') or '').strip(),
            'ref_min': ref_lo,
            'ref_max': ref_hi,
            'flag'   : flag,
            'estado' : estado,
        })
    return {'analizador': analizador, 'parametros': params}


RE_CHEM = re.compile(
    r'^(?P<item>[A-Za-zÀ-ÿ:][A-Za-zÀ-ÿ\s:/\.]{1,50}?)\s+'
    r'(?P<valor>[<>]?\d+[\.,]?\d*)\s*'
    r'(?:(?P<ref_low>\d+[\.,]?\d*)\s*[-–]\s*(?P<ref_high>\d+[\.,]?\d*)\s*'
    r'(?P<unidad>[^\s\dHL][^\s]{0,14})\s*)?'
    r'(?P<flag>[HL])?\s*$'
)

CHEM_SKIP    = SKIP_TOKENS | {'rbc', 'wbc', 'download', 'run', 'lym', 'mono', 'baso', 'eos'}
_RATIO_SUFFIX = re.compile(r'^[A-Za-z][A-Za-z\s]{0,30}$')


def _preprocess_chem_lines(lineas: List[str], ini: int, fin: int) -> List[str]:
    result = []
    i = ini
    while i < fin:
        l = lineas[i].strip()
        i += 1
        if not l:
            continue
        m = RE_CHEM.match(l)
        if m:
            item = m.group('item').strip()
            if ':' in item and i < fin:
                peek = lineas[i].strip()
                if _RATIO_SUFFIX.match(peek) and not re.search(r'\d', peek):
                    new_item = item + ' ' + peek
                    rest = l[m.start('valor'):]
                    result.append(new_item + '  ' + rest)
                    i += 1
                    continue
            result.append(l)
        else:
            if _skip(l):
                result.append(l)
                continue
            if i < fin:
                l2 = lineas[i].strip()
                combined = l + ' ' + l2
                if RE_CHEM.match(combined):
                    result.append(combined)
                    i += 1
                    continue
            result.append(l)
    return result


def extraer_quimica(lineas: List[str], ini: int, fin: int) -> Dict:
    params: List[Dict] = []
    analizador = ''
    preprocessed = _preprocess_chem_lines(lineas, ini, fin)
    for l in preprocessed:
        if not l or _skip(l):
            continue
        if 'Catalyst' in l or 'catalyst' in l.lower():
            analizador = l
            continue
        if any(t in l.lower() for t in ['rbc run', 'wbc run', 'download', 'rbc_frag']):
            continue
        m = RE_CHEM.match(l)
        if not m:
            continue
        item = m.group('item').strip()
        if not item or item.lower().split()[0] in CHEM_SKIP:
            continue
        val = _val(m.group('valor'))
        if val is None:
            continue
        ref_lo = _val(m.group('ref_low'))
        ref_hi = _val(m.group('ref_high'))
        flag   = m.group('flag')
        if flag == 'H' or (ref_hi is not None and val > ref_hi):
            estado = 'alto'
        elif flag == 'L' or (ref_lo is not None and val < ref_lo):
            estado = 'bajo'
        else:
            estado = 'normal'
        params.append({
            'test'   : item,
            'valor'  : val,
            'unidad' : (m.group('unidad') or '').strip(),
            'ref_min': ref_lo,
            'ref_max': ref_hi,
            'flag'   : flag,
            'estado' : estado,
        })
    return {'analizador': analizador, 'parametros': params}


RE_RESULT_ONLY = re.compile(r'^(Negative|Positive|Negativo|Positivo)\s*$', re.I)
RE_TIMESTAMP   = re.compile(r'^\d{2}/\d{2}/\d{2}\s+\d{1,2}:\d{2}')


def extraer_serologia(pdf_path: str, lineas: List[str], ini: int, fin: int) -> Dict:
    params: List[Dict] = []
    analizador = ''

    page_words: List[Dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        line_offset = 0
        for page in pdf.pages:
            page_text = page.extract_text() or ''
            n_lines   = len(page_text.splitlines())
            page_ini  = line_offset
            page_fin  = line_offset + n_lines
            if page_fin > ini and page_ini < fin:
                mid_x = page.width / 2
                for w in page.extract_words(x_tolerance=3, y_tolerance=3):
                    page_words.append({
                        'text' : w['text'],
                        'top'  : w['top'],
                        'right': w['x0'] >= mid_x,
                    })
            line_offset = page_fin

    if not page_words:
        return {'analizador': analizador, 'parametros': params}

    rows_by_y: Dict[int, Dict] = {}
    for w in page_words:
        y = round(w['top'] / 5) * 5
        if y not in rows_by_y:
            rows_by_y[y] = {'left': [], 'right': []}
        rows_by_y[y]['right' if w['right'] else 'left'].append(w['text'])

    for y in sorted(rows_by_y):
        left  = ' '.join(rows_by_y[y]['left']).strip()
        right = ' '.join(rows_by_y[y]['right']).strip()

        if not left or _skip(left) or RE_TIMESTAMP.match(left):
            continue
        if 'SNAP' in left or 'Handheld' in left or ('ELISA' in left and len(left) < 40):
            analizador = left
            continue

        if right and RE_RESULT_ONLY.match(right):
            res = right.capitalize()
            params.append({
                'test'     : left,
                'resultado': res,
                'positivo' : res.lower() in {'positive', 'positivo'},
            })
        elif params and not RE_RESULT_ONLY.match(left):
            params[-1]['test'] += ' ' + left

    return {'analizador': analizador, 'parametros': params}


def extraer_comentarios_idexx(lineas: List[str]) -> List[str]:
    comentarios: List[str] = []
    for l in lineas:
        t = l.strip()
        if len(t) < 20:
            continue
        if COMMENT_KW.search(t) and t not in comentarios:
            comentarios.append(t)
    return comentarios


def extraer_pdf_completo(ruta: str) -> Dict:
    lineas = _extract_lines(ruta)
    total  = len(lineas)
    cabecera    = extraer_cabecera(lineas)
    secciones   = _find_sections(lineas)
    comentarios = extraer_comentarios_idexx(lineas)

    resultado: Dict[str, Any] = {
        'fuente'              : 'IDEXX VetConnect PLUS',
        'tipo'                : 'digital_pdf',
        'especie'             : cabecera.get('especie', ''),
        'raza'                : cabecera.get('raza', ''),
        'edad'                : cabecera.get('edad', ''),
        'fecha'               : cabecera.get('fecha', ''),
        'secciones_presentes' : list(secciones.keys()),
        'comentarios_clinicos': comentarios,
    }

    if 'hematologia' in secciones:
        ini, fin = _section_range(secciones, 'hematologia', total)
        resultado['hematologia'] = extraer_hematologia(lineas, ini, fin)

    if 'quimica' in secciones:
        ini, fin = _section_range(secciones, 'quimica', total)
        resultado['quimica'] = extraer_quimica(lineas, ini, fin)

    if 'serologia' in secciones:
        ini, fin = _section_range(secciones, 'serologia', total)
        resultado['serologia'] = extraer_serologia(ruta, lineas, ini, fin)

    return resultado


def _agrupar_filas(dets: list, tolerancia_y: int = 15) -> list:
    if not dets:
        return []
    dets = sorted(dets, key=lambda r: (r[0][0][1] + r[0][2][1]) / 2)
    filas = []
    fila_actual = [dets[0]]
    y_ref = (dets[0][0][0][1] + dets[0][0][2][1]) / 2
    for det in dets[1:]:
        yc = (det[0][0][1] + det[0][2][1]) / 2
        if abs(yc - y_ref) <= tolerancia_y:
            fila_actual.append(det)
        else:
            filas.append(sorted(fila_actual, key=lambda r: (r[0][0][0] + r[0][2][0]) / 2))
            fila_actual = [det]
            y_ref = yc
    filas.append(sorted(fila_actual, key=lambda r: (r[0][0][0] + r[0][2][0]) / 2))
    return filas


def _order_corners(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _warp_perspective(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_corners(pts.reshape(4, 2).astype(np.float32))
    wa   = np.linalg.norm(rect[1] - rect[0])
    wb   = np.linalg.norm(rect[2] - rect[3])
    ha   = np.linalg.norm(rect[3] - rect[0])
    hb   = np.linalg.norm(rect[2] - rect[1])
    W    = int(max(wa, wb))
    H    = int(max(ha, hb))
    if W < 50 or H < 50:
        return img
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M   = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (W, H))


def detectar_ticket_contorno(img: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
    h, w     = img.shape[:2]
    img_area = h * w
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur     = cv2.GaussianBlur(gray, (5, 5), 0)
    best_crop = None
    best_area = 0
    for (t1, t2) in [(30, 120), (50, 150), (80, 200)]:
        edges  = cv2.Canny(blur, t1, t2)
        kernel = np.ones((5, 5), np.uint8)
        dil    = cv2.dilate(edges, kernel, iterations=3)
        cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:6]:
            area = cv2.contourArea(c)
            if area < img_area * 0.04:
                continue
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.025 * peri, True)
            if len(approx) == 4 and area > best_area:
                best_area = area
                best_crop = _warp_perspective(img, approx)
    if best_crop is not None:
        bh, bw = best_crop.shape[:2]
        if bh > bw * 1.2:
            return best_crop, True
    return None, False


def detectar_ticket_brillo(img: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, False
    candidates = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(c)
        if area < h * w * 0.05:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        candidates.append((bh / max(bw, 1), area, x, y, bw, bh))
    if not candidates:
        return None, False
    candidates.sort(reverse=True)
    _, _, x, y, bw, bh = candidates[0]
    return img[y:y + bh, x:x + bw], False


def detectar_y_recortar(img: np.ndarray) -> Tuple[np.ndarray, str]:
    crop, ok = detectar_ticket_contorno(img)
    if ok:
        return crop, 'contorno+perspectiva'
    crop, _ = detectar_ticket_brillo(img)
    if crop is not None:
        return crop, 'umbral_brillo'
    return img, 'sin_deteccion'


def preprocesar_voucher(img: np.ndarray) -> np.ndarray:
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gris.shape
    if w < 700:
        gris = cv2.resize(gris, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    suav     = cv2.bilateralFilter(gris, 9, 75, 75)
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    mejorado = clahe.apply(suav)
    if np.mean(mejorado) < 100:
        mejorado = cv2.bitwise_not(mejorado)
    return mejorado


RE_RANGO      = re.compile(r'([\d]+[\.,]?\d*)\s*[-–]\s*([\d]+[\.,]?\d*)')
CABECERA_VOUC = {
    'animal', 'nombre', 'secuencia', 'muestra', 'maestro', 'sexo',
    'edad', 'tipo', 'test', 'hora', 'item', 'resultado', 'referencia', 'nota'
}


def extraer_voucher(ruta: str, reader: easyocr.Reader) -> Dict:
    img = cv2.imread(ruta)
    if img is None:
        return {'error': f'No se pudo leer: {ruta}', 'tipo': 'voucher_imagen'}

    recortado, metodo = detectar_y_recortar(img)
    img_proc = preprocesar_voucher(recortado)
    dets   = reader.readtext(img_proc, detail=1, paragraph=False)
    utiles = [(b, t, c) for b, t, c in dets if c >= 0.25]

    if not utiles:
        return {'error': 'Sin detecciones OCR', 'tipo': 'voucher_imagen', 'metodo_deteccion': metodo}

    filas = _agrupar_filas(utiles, tolerancia_y=20)
    ancho = img_proc.shape[1]
    X_VAL = ancho * 0.38
    X_REF = ancho * 0.60
    X_NOT = ancho * 0.82

    params: List[Dict] = []
    cabecera_pct: Dict[str, str] = {}

    for fila in filas:
        col_item, col_val, col_ref, col_nota = [], [], [], []
        for bbox, txt, conf in fila:
            xc = (bbox[0][0] + bbox[2][0]) / 2
            if xc < X_VAL:
                col_item.append(txt)
            elif xc < X_REF:
                col_val.append(txt)
            elif xc < X_NOT:
                col_ref.append(txt)
            else:
                col_nota.append(txt)

        item    = ' '.join(col_item).strip().upper()
        val_raw = ' '.join(col_val).strip()
        ref_raw = ' '.join(col_ref).strip()
        nota    = ' '.join(col_nota).strip().upper()

        if not item:
            continue
        if item.lower().split()[0] in CABECERA_VOUC:
            cabecera_pct[item] = val_raw
            continue

        val_clean = re.sub(r'\s*10[\^]?\d+/[Lµ]+', '', val_raw).strip()
        val_num   = _val(re.sub(r'[^\d\.,]', '', val_clean))
        if val_num is None:
            continue

        ref_m  = RE_RANGO.search(ref_raw)
        ref_lo = _val(ref_m.group(1)) if ref_m else None
        ref_hi = _val(ref_m.group(2)) if ref_m else None

        flag = None
        if 'H' in nota and 'L' not in nota:
            flag = 'H'
        elif 'L' in nota and 'H' not in nota:
            flag = 'L'

        if flag == 'H' or (ref_hi is not None and val_num > ref_hi):
            estado = 'alto'
        elif flag == 'L' or (ref_lo is not None and val_num < ref_lo):
            estado = 'bajo'
        else:
            estado = 'normal'

        params.append({
            'test'   : item,
            'valor'  : val_num,
            'unidad' : '',
            'ref_min': ref_lo,
            'ref_max': ref_hi,
            'flag'   : flag,
            'estado' : estado,
        })

    return {
        'fuente'              : 'Voucher VARGAS VET',
        'tipo'                : 'voucher_imagen',
        'metodo_deteccion'    : metodo,
        'cabecera'            : cabecera_pct,
        'secciones_presentes' : ['hematologia'],
        'hematologia'         : {'analizador': 'Hematology Analyzer (portable)', 'parametros': params},
        'comentarios_clinicos': [],
    }


RANGOS_REF: Dict[str, Dict] = {
    'Perro': {
        'WBC'         : (5.05, 16.76, 'K/µL'),
        'RBC'         : (5.65, 8.87,  'M/µL'),
        'HGB'         : (13.1, 20.5,  'g/dL'),
        'HEMOGLOBIN'  : (13.1, 20.5,  'g/dL'),
        'HCT'         : (37.3, 61.7,  '%'),
        'HEMATOCRIT'  : (37.3, 61.7,  '%'),
        'MCV'         : (61.6, 73.5,  'fL'),
        'MCH'         : (21.2, 25.9,  'pg'),
        'MCHC'        : (32.0, 37.9,  'g/dL'),
        'RDW'         : (13.6, 21.7,  '%'),
        'PLT'         : (148,  484,   'K/µL'),
        'PLATELETS'   : (148,  484,   'K/µL'),
        'PDW'         : (9.1,  19.4,  'fL'),
        'MPV'         : (8.7,  13.2,  'fL'),
        'PLATELETCRIT': (0.14, 0.46,  '%'),
        'NEUTROPHILS' : (2.95, 11.64, 'K/µL'),
        'LYMPHOCYTES' : (1.05, 5.10,  'K/µL'),
        'MONOCYTES'   : (0.16, 1.12,  'K/µL'),
        'EOSINOPHILS' : (0.06, 1.23,  'K/µL'),
        'BASOPHILS'   : (0.00, 0.10,  'K/µL'),
        'RETICULOCYTES': (10.0, 110.0, 'K/µL'),
        'GLUCOSE'     : (74,   143,   'mg/dL'),
        'CREATININE'  : (0.5,  1.8,   'mg/dL'),
        'BUN'         : (7,    27,    'mg/dL'),
        'TOTAL PROTEIN': (5.2, 8.2,   'g/dL'),
        'ALBUMIN'     : (2.3,  4.0,   'g/dL'),
        'GLOBULIN'    : (2.5,  4.5,   'g/dL'),
        'ALT'         : (10,   125,   'U/L'),
        'AST'         : (0,    50,    'U/L'),
        'ALP'         : (23,   212,   'U/L'),
        'GGT'         : (0,    11,    'U/L'),
        'AMYLASE'     : (200,  1200,  'U/L'),
        'LIPASE'      : (0,    250,   'U/L'),
        'CALCIUM'     : (9.1,  11.7,  'mg/dL'),
        'PHOSPHORUS'  : (2.9,  6.2,   'mg/dL'),
        'LYM#'        : (0.8,  5.1,   '10^9/L'),
        'MID#'        : (0.0,  1.8,   '10^9/L'),
        'GRA#'        : (4.0,  12.6,  '10^9/L'),
        'LYM%'        : (12,   30,    '%'),
        'MID%'        : (2,    9,     '%'),
        'GRA%'        : (60,   83,    '%'),
        'RDW-CV'      : (11.0, 15.5,  '%'),
        'RDW-SD'      : (35.0, 56.0,  'fL'),
        'PCT'         : (0.1,  0.5,   '%'),
        'MPU'         : (7.0,  12.9,  'fL'),
        'P-LCR'       : (13.0, 43.0,  '%'),
    },
    'Gato': {
        'WBC'        : (5.5,  19.5,  'K/µL'),
        'RBC'        : (5.0,  10.0,  'M/µL'),
        'HGB'        : (8.0,  15.0,  'g/dL'),
        'HEMOGLOBIN' : (8.0,  15.0,  'g/dL'),
        'HCT'        : (24.0, 45.0,  '%'),
        'HEMATOCRIT' : (24.0, 45.0,  '%'),
        'MCV'        : (39.0, 55.0,  'fL'),
        'MCH'        : (12.5, 17.5,  'pg'),
        'MCHC'       : (30.0, 36.0,  'g/dL'),
        'PLT'        : (150,  600,   'K/µL'),
        'NEUTROPHILS': (2.5,  12.5,  'K/µL'),
        'LYMPHOCYTES': (1.5,  7.0,   'K/µL'),
        'GLUCOSE'    : (64,   170,   'mg/dL'),
        'CREATININE' : (0.8,  2.4,   'mg/dL'),
        'BUN'        : (14,   36,    'mg/dL'),
        'ALT'        : (6,    83,    'U/L'),
        'AST'        : (0,    48,    'U/L'),
        'ALP'        : (10,   90,    'U/L'),
    },
}


def generar_alertas(resultado: Dict, especie: str = 'Perro') -> List[str]:
    alertas: List[str] = []
    rangos = RANGOS_REF.get(especie, RANGOS_REF['Perro'])

    for sec_name in ('hematologia', 'quimica'):
        sec = resultado.get(sec_name, {})
        for p in sec.get('parametros', []):
            if p['estado'] == 'normal':
                continue
            test   = p['test'].upper()
            val    = p['valor']
            r_min  = p.get('ref_min')
            r_max  = p.get('ref_max')
            unidad = p.get('unidad', '')
            if r_min is None or r_max is None:
                if test in rangos:
                    r_min, r_max, unidad = rangos[test]
                else:
                    continue
            flecha = '↑ ALTO' if p['estado'] == 'alto' else '↓ BAJO'
            alertas.append(f"{p['test']}: {val} {unidad} — {flecha} (ref: {r_min}–{r_max})")

    for p in resultado.get('serologia', {}).get('parametros', []):
        if p.get('positivo'):
            alertas.append(f"SEROLOGÍA POSITIVO: {p['test']}")

    return alertas


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


def extraer_laboratorio(
    ruta: str,
    especie: str = 'Perro',
    reader: Optional[easyocr.Reader] = None,
) -> Dict:
    ext = Path(ruta).suffix.lower()

    if ext == '.pdf':
        with pdfplumber.open(ruta) as pdf:
            txt = pdf.pages[0].extract_text() or ''
        if len(txt.strip()) >= 50:
            resultado = extraer_pdf_completo(ruta)
        else:
            return {'error': 'PDF escaneado (sin texto)', 'tipo': 'pdf_escaneado'}
    elif ext in IMAGE_EXTENSIONS:
        if reader is None:
            return {'error': 'OCR reader requerido para imágenes'}
        resultado = extraer_voucher(ruta, reader)
    else:
        return {'error': f'Formato no soportado: {ext}'}

    resultado['alertas'] = generar_alertas(resultado, especie)
    return resultado
