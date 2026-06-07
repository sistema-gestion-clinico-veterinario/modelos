import sys
import re
from pathlib import Path
import pandas as pd
import pdfplumber
from config import EXTENSIONES_IMAGEN, PDF_TEXTO_MINIMO, COLUMNAS
from extractor_pdf import extraer as extraer_pdf
from extractor_ocr import extraer as extraer_ocr
from extractor_graficos import extraer_graficos





MAPA_UNIDADES_OCR = {
    "WBC": "10^9/L", "LYM#": "10^9/L", "MID#": "10^9/L", "GRA#": "10^9/L", "PLT": "10^9/L",
    "RBC": "10^12/L", "HGB": "g/dL", "MCHC": "g/dL", "MCH": "pg",
    "MCV": "fL", "RDW-SD": "fL", "MPV": "fL",
    "LYM%": "%", "MID%": "%", "GRA%": "%", "RDW-CV": "%", "PDW": "%", "PCT": "%", "P-LCR": "%", "HCT": "%"
}


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


ITEMS_PDF = [
    "RBC", "Hematocrit", "Hemoglobin", "MCV", "MCH", "MCHC", "RDW",
    "% Reticulocytes", "Reticulocytes", "Reticulocyte Hemoglobin", "WBC",
    "% Neutrophils", "% Lymphocytes", "% Monocytes", "% Eosinophils", "% Basophils",
    "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils", "Basophils",
    "Platelets", "PDW", "MPV"
]

ITEMS_OCR = [
    "WBC", "LYM#", "MID#", "GRA#", "LYM%", "MID%", "GRA%",
    "RBC", "HGB", "MCHC", "MCH", "MCV", "RDW-CV", "RDW-SD",
    "HCT", "PLT", "MPV", "PDW", "PCT", "P-LCR"
]


def normalizar_resultado_ocr(item: str, resultado_str: str) -> str:
    if not resultado_str:
        return ""
    res_clean = resultado_str.replace(",", ".")
    

    tokens = res_clean.split()
    unidad_esperada = MAPA_UNIDADES_OCR.get(item.upper(), "")
    
    for idx, t in enumerate(tokens):
        if any(c.isdigit() for c in t) or t.strip() in {"L", "I", "O"}:
            suffix = ""
            if t.endswith("%") or t.endswith("{") or t.endswith("*"):
                suffix = t[-1]
                t = t[:-1]

            if not any(u in t.upper() for u in ["10^", "10*", "109", "1012"]):

                if t and t[0] == 'M' and len(t) > 1 and any(c.isdigit() for c in t[1:]):
                    t = '7' + t[1:]
                t = t.replace("L", "4").replace("I", "1").replace("O", "0").replace("S", "5").replace("B", "8")
            t = t + suffix
            tokens[idx] = t

        elif t.upper() in {"ML", "ML."} and unidad_esperada == "fL":
            tokens[idx] = "fL"
    res_clean = " ".join(tokens)
    
    match = re.search(r'\d+\.\d+|\d+', res_clean)
    if not match:
        return ""
    num_str = match.group().strip(".")
    
    unidad = MAPA_UNIDADES_OCR.get(item.upper())
    if unidad:
        return f"{num_str} {unidad}"
    return num_str


def calcular_flag(resultado_str: str, referencia_str: str) -> str:
    res_match = re.search(r'\d+\.\d+|\d+', resultado_str.replace(',', '.'))
    if not res_match:
        return ""
    try:
        val = float(res_match.group())
    except ValueError:
        return ""
        
    ref_matches = re.findall(r'\d+\.\d+|\d+', referencia_str.replace(',', '.'))
    if len(ref_matches) >= 2:
        try:
            val_min = float(ref_matches[0])
            val_max = float(ref_matches[1])
            if val < val_min:
                return "L"
            elif val > val_max:
                return "H"
        except ValueError:
            pass
    return ""


def procesar(archivo: str) -> tuple:
    tipo = detectar_tipo(archivo)
    if tipo == "pdfplumber":
        df = extraer_pdf(archivo)
        items_estandar = ITEMS_PDF
    else:
        df = extraer_ocr(archivo)
        items_estandar = ITEMS_OCR

    for col in COLUMNAS:
        if col not in df.columns:
            df[col] = ""

    mapa_extraido = {}
    for _, fila in df.iterrows():
        item = str(fila.get("item", "")).strip()
        if item:
            resultado = str(fila.get("resultado", "")).strip()
            referencia = str(fila.get("referencia", "")).strip()
            flag_raw = str(fila.get("flag", "")).strip().upper()
            
            flag = ""
            if "H" in flag_raw:
                flag = "H"
            elif "L" in flag_raw:
                flag = "L"
                
            mapa_extraido[item.upper()] = {
                "resultado": resultado,
                "referencia": referencia,
                "flag": flag
            }


    if tipo == "easyocr":
        items_ocr_upper = [x.upper() for x in items_estandar]
        for item_std in items_ocr_upper:
            datos = mapa_extraido.get(item_std, {"resultado": "", "referencia": "", "flag": ""})
            
            nuevo_res = normalizar_resultado_ocr(item_std, datos["resultado"])
            nueva_ref = datos["referencia"].strip()
            nuevo_flag = datos["flag"]
            
            if not nuevo_flag and nuevo_res and nueva_ref:
                nuevo_flag = calcular_flag(nuevo_res, nueva_ref)
            
            if not nuevo_res:
                nueva_ref = ""
                nuevo_flag = ""
                
            mapa_extraido[item_std] = {
                "resultado": nuevo_res,
                "referencia": nueva_ref,
                "flag": nuevo_flag
            }

    filas_alineadas = []
    for item_std in items_estandar:
        datos_ext = mapa_extraido.get(item_std.upper(), {"resultado": "", "referencia": "", "flag": ""})
        filas_alineadas.append({
            "item": item_std,
            "resultado": datos_ext["resultado"],
            "referencia": datos_ext["referencia"],
            "flag": datos_ext["flag"]
        })

    df_final = pd.DataFrame(filas_alineadas)
    return df_final[COLUMNAS], tipo


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

    resultados_totales = []
    for archivo in archivos:
        ext = Path(archivo).suffix.lower()
        if ext not in EXTENSIONES_IMAGEN and ext != ".pdf":
            continue

        print(f"Procesando: {archivo} ...")
        try:
            df, tipo = procesar(archivo)
            nombre_base = Path(archivo).stem
            dir_results = Path("results")
            dir_results.mkdir(exist_ok=True)
            
            ruta_csv = dir_results / f"{nombre_base}.csv"
            df.to_csv(ruta_csv, index=False)
            
            if extraer_graf and tipo == "easyocr":
                extraer_graficos(archivo, df)
                
            resultados_totales.append({
                "archivo": Path(archivo).name,
                "tipo": tipo,
                "filas": len(df)
            })
        except Exception as e:
            print(f"Error procesando {archivo}: {str(e)}")

    print("\nProceso terminado.")
    print(f"Archivos procesados con éxito: {len(resultados_totales)}")
    for res in resultados_totales:
        print(f"  - {res['archivo']} ({res['tipo']}): {res['filas']} filas")


if __name__ == "__main__":
    main()
