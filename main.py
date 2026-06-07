import sys
from pathlib import Path
import pandas as pd
import pdfplumber
from config import EXTENSIONES_IMAGEN, PDF_TEXTO_MINIMO, COLUMNAS
from extractor_pdf import extraer as extraer_pdf
from extractor_ocr import extraer as extraer_ocr
from extractor_graficos import extraer_graficos


def detectar_tipo(archivo: str) -> str:
    ext = Path(archivo).suffix.lower()
    if ext == ".pdf":
        return "easyocr" if _es_pdf_escaneado(archivo) else "pdfplumber"
    elif ext in EXTENSIONES_IMAGEN:
        return "easyocr"
    else:
        raise ValueError(f"Formato no soportado: {ext}")


def _es_pdf_escaneado(pdf_path: str) -> bool:
    with pdfplumber.open(pdf_path) as pdf:
        texto = pdf.pages[0].extract_text() or ""
        return len(texto.strip()) < PDF_TEXTO_MINIMO


def procesar(archivo: str) -> tuple:
    tipo = detectar_tipo(archivo)
    if tipo == "pdfplumber":
        df = extraer_pdf(archivo)
    else:
        df = extraer_ocr(archivo)

    for col in COLUMNAS:
        if col not in df.columns:
            df[col] = ""
    return df[COLUMNAS], tipo


def main():
    extraer_graf = "--graficos" in sys.argv
    if extraer_graf:
        sys.argv.remove("--graficos")

    if len(sys.argv) < 2:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            rutas = filedialog.askopenfilenames(
                title="Selecciona archivos (PDF, JPG, PNG...)",
                filetypes=[("Archivos soportados", "*.pdf *.jpg *.jpeg *.png *.tiff *.bmp"), ("Todos", "*.*")]
            )
            root.destroy()
            if not rutas:
                print("No se seleccionó ningún archivo.")
                return
            archivos = list(rutas)
        except ImportError:
            print("Uso: python main.py <archivo_o_directorio>")
            print("  o instala tkinter para usar el selector gráfico.")
            sys.exit(1)
    else:
        ruta = sys.argv[1]
        path = Path(ruta)
        if path.is_file():
            archivos = [str(path)]
        elif path.is_dir():
            archivos = [str(p) for p in path.rglob("*") if p.is_file()]
        else:
            print(f"Ruta no válida: {ruta}")
            sys.exit(1)

    for archivo in archivos:
        try:
            df, tipo = procesar(archivo)
            print(f"\n=== {Path(archivo).name} ({tipo}) ===")
            print(df.to_string(index=False))

            if extraer_graf and tipo == "pdfplumber":
                rutas = extraer_graficos(archivo)
                for r in rutas:
                    print(f"  Grafico guardado: {r}")
        except Exception as e:
            print(f"Error procesando {archivo}: {e}")


if __name__ == "__main__":
    main()
