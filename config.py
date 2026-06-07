from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
IMAGENES_DIR = DATA_DIR / "imagenes"
GROUND_TRUTH_DIR = BASE_DIR / "ground_truth"
RESULTS_DIR = BASE_DIR / "results"

COLUMNAS = ["item", "resultado", "referencia", "flag"]

TABLE_SETTINGS_PDFPLUMBER = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 5,
    "join_tolerance": 5,
}

TABLE_SETTINGS_TEXT = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
}

OCR_IDIOMAS = ["es", "en"]
OCR_GPU = False
OCR_CONFIANZA_MINIMA = 0.2
OCR_TOLERANCIA_Y_FACTOR = 0.6
OCR_ZONAS_COLUMNAS = 4

PDF_TEXTO_MINIMO = 80

EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}

MAPA_COLUMNAS = {
    "analito": "item",
    "test": "item",
    "parametro": "item",
    "parámetro": "item",
    "valor": "resultado",
    "result": "resultado",
    "rango": "referencia",
    "referencia": "referencia",
    "ref": "referencia",
    "flag": "flag",
    "h/l": "flag",
}
