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

    h, w = gris.shape
    min_dim = 3000
    if w < min_dim or h < min_dim:
        factor = min_dim / min(w, h)
        gris = cv2.resize(gris, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

    gris = _unsharp_mask(gris, sigma=1.0, strength=1.5)

    clip = 2.0 if _es_buena_calidad(gris) else 4.0
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    gris = clahe.apply(gris)

    gris = cv2.bilateralFilter(gris, 5, 75, 75)

    return gris


def _es_buena_calidad(gris: np.ndarray) -> bool:
    lap = cv2.Laplacian(gris, cv2.CV_64F).var()
    return lap > 80


def _unsharp_mask(img: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)


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
