import cv2
import numpy as np


def preprocesar_imagen(ruta_imagen: str) -> np.ndarray:
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    angulo = _detectar_angulo(gris)
    if abs(angulo) > 1.5:
        gris = _rotar(gris, angulo)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gris = clahe.apply(gris)
    return gris


def _detectar_angulo(gris: np.ndarray) -> float:
    bordes = cv2.Canny(gris, 50, 150)
    lineas = cv2.HoughLines(bordes, 1, np.pi / 180, 100)
    if lineas is None:
        return 0.0
    angulos = []
    for linea in lineas:
        theta = np.degrees(linea[0][1]) - 90
        if abs(theta) <= 15:
            angulos.append(theta)
    if not angulos:
        return 0.0
    return float(np.median(angulos))


def _rotar(imagen: np.ndarray, angulo: float) -> np.ndarray:
    h, w = imagen.shape[:2]
    centro = (w // 2, h // 2)
    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    return cv2.warpAffine(
        imagen, matriz, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
