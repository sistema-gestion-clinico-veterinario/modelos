import easyocr
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from preprocesador import preprocesar_imagen

def main():
    if len(sys.argv) < 2:
        print("Uso: python test_ocr.py <imagen>")
        return
    ruta = sys.argv[1]
    reader = easyocr.Reader(['es', 'en'], gpu=False)
    imagen = preprocesar_imagen(ruta)
    
    cv2.imwrite("scratch/preprocessed_debug.png", imagen)
    
    resultados = reader.readtext(imagen, detail=1, paragraph=False)
    print("=== RAW OCR RESULTS ===")
    for bbox, texto, conf in resultados:
        print(f"Conf: {conf:.2f} | Texto: {texto} | BBox: {bbox}")

if __name__ == "__main__":
    main()
