import numpy as np
import pydicom
from io import BytesIO
from pathlib import Path
from PIL import Image
from config import IMG_SIZE

_DICOM_MAGIC        = b'DICM'
_DICOM_MAGIC_OFFSET = 128
_DICOM_EXTENSIONS   = {'.dcm', '.dicom'}
_IMAGE_EXTENSIONS   = {'.png', '.jpg', '.jpeg'}


def is_dicom(data: bytes, filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    if ext in _DICOM_EXTENSIONS:
        return True
    if ext in _IMAGE_EXTENSIONS:
        return False
    if len(data) > _DICOM_MAGIC_OFFSET + 4:
        return data[_DICOM_MAGIC_OFFSET:_DICOM_MAGIC_OFFSET + 4] == _DICOM_MAGIC
    return False


def load_dicom(data: bytes) -> Image.Image:
    dcm = pydicom.dcmread(BytesIO(data), force=True)
    arr = dcm.pixel_array.astype(np.float32)
    if (
        hasattr(dcm, 'PhotometricInterpretation')
        and dcm.PhotometricInterpretation == 'MONOCHROME1'
    ):
        arr = arr.max() - arr
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    arr     = np.clip(arr, p2, p98)
    arr     = (arr - p2) / (p98 - p2 + 1e-8)
    gray    = Image.fromarray((arr * 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), Image.LANCZOS
    )
    return Image.merge('RGB', [gray, gray, gray])


def load_image(data: bytes) -> Image.Image:
    gray = (
        Image.open(BytesIO(data))
        .convert('L')
        .resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    )
    return Image.merge('RGB', [gray, gray, gray])


def prepare(data: bytes, filename: str) -> tuple[Image.Image, str]:
    if is_dicom(data, filename):
        return load_dicom(data), 'dicom'
    return load_image(data), 'image'
