import json
import sys
from pathlib import Path
import pandas as pd
from normalizar import normalizar
from main import procesar
from config import COLUMNAS, RESULTS_DIR


def evaluar(archivos: list, ground_truth_path: str) -> pd.DataFrame:
    with open(ground_truth_path, encoding="utf-8") as f:
        gt = json.load(f)

    resultados = []
    for archivo in archivos:
        nombre = Path(archivo).name
        gt_filas = gt.get(nombre, {}).get("rows", [])
        if not gt_filas:
            print(f"Sin ground truth para {nombre}")
            continue

        df_extraido, tipo = procesar(archivo)
        metricas = {
            "archivo": nombre,
            "tipo": tipo,
            "total_filas": len(gt_filas),
            "filas_completas": 0,
            "col_item": 0,
            "col_resultado": 0,
            "col_referencia": 0,
            "col_flag": 0,
        }

        for i, fila_gt in enumerate(gt_filas):
            if i >= len(df_extraido):
                break
            fila_ext = df_extraido.iloc[i].to_dict()
            correctas = 0
            for col in COLUMNAS:
                if normalizar(fila_ext.get(col, "")) == normalizar(fila_gt.get(col, "")):
                    metricas[f"col_{col}"] += 1
                    correctas += 1
            if correctas == len(COLUMNAS):
                metricas["filas_completas"] += 1

        resultados.append(metricas)

    return pd.DataFrame(resultados)


def generar_informe(df: pd.DataFrame):
    df = df[df["total_filas"] > 0].copy()
    n = df["total_filas"]
    df["% item"] = (df["col_item"] / n * 100).round(1)
    df["% resultado"] = (df["col_resultado"] / n * 100).round(1)
    df["% referencia"] = (df["col_referencia"] / n * 100).round(1)
    df["% flag"] = (df["col_flag"] / n * 100).round(1)
    df["% completo"] = (df["filas_completas"] / n * 100).round(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "metricas_detalladas.csv", index=False, encoding="utf-8")

    columnas_porcentaje = ["% item", "% resultado", "% referencia", "% flag", "% completo"]
    resumen = df.groupby("tipo")[columnas_porcentaje].mean().round(1)
    resumen.to_csv(RESULTS_DIR / "resumen_por_estrategia.csv", encoding="utf-8")

    informe_path = RESULTS_DIR / "informe_comparativo.md"
    with open(informe_path, "w", encoding="utf-8") as f:
        f.write("# Informe Comparativo: pdfplumber vs EasyOCR\n\n")
        f.write("## Resumen por estrategia\n\n")
        f.write(resumen.to_string() + "\n\n")
        f.write("## Detalle por archivo\n\n")
        f.write(df.to_string(index=False) + "\n")

    print("=== RESUMEN POR ESTRATEGIA ===")
    print(resumen.to_string())
    print(f"\nInformes guardados en: {RESULTS_DIR}")


def main():
    if len(sys.argv) < 2:
        print("Uso: python metricas.py <ground_truth.json> [archivos...]")
        sys.exit(1)

    gt_path = sys.argv[1]
    archivos = sys.argv[2:] if len(sys.argv) > 2 else []

    if not archivos:
        print("No se especificaron archivos. Usando data/")
        from config import PDF_DIR, IMAGENES_DIR
        archivos = (
            [str(p) for p in PDF_DIR.glob("*") if p.is_file()]
            + [str(p) for p in IMAGENES_DIR.glob("*") if p.is_file()]
        )

    df = evaluar(archivos, gt_path)
    if df.empty:
        print("No se generaron métricas (sin ground truth o archivos).")
        return

    generar_informe(df)


if __name__ == "__main__":
    main()
