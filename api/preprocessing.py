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


def check_image_quality(data: bytes) -> dict:
    """
    Analiza si una imagen PNG/JPG tiene características de radiografía.
    Retorna un dict con flags y el motivo si no parece radiografía.
    """
    try:
        img = Image.open(BytesIO(data)).convert('RGB').resize((256, 256), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        diff_rg = np.mean(np.abs(r - g))
        diff_rb = np.mean(np.abs(r - b))
        diff_gb = np.mean(np.abs(g - b))
        color_deviation = float((diff_rg + diff_rb + diff_gb) / 3)

        gray = 0.299 * r + 0.587 * g + 0.114 * b
        contrast = float(gray.std())

        issues = []
        if color_deviation > 15:
            issues.append(
                f'la imagen presenta colores (desviación cromática: {color_deviation:.1f}); '
                f'las radiografías son en escala de grises'
            )
        if contrast < 20:
            issues.append(
                f'el contraste es muy bajo ({contrast:.1f}); '
                f'las radiografías tienen alto contraste'
            )

        return {
            'parece_radiografia': len(issues) == 0,
            'color_deviation': round(color_deviation, 2),
            'contrast': round(contrast, 2),
            'issues': issues,
        }
    except Exception:
        return {'parece_radiografia': True, 'issues': []}


def prepare(data: bytes, filename: str) -> tuple[Image.Image, str]:
    if is_dicom(data, filename):
        return load_dicom(data), 'dicom'
    return load_image(data), 'image'


def extract_dicom_metadata(data: bytes) -> dict:
    try:
        dcm = pydicom.dcmread(BytesIO(data), force=True, stop_before_pixels=True)
    except Exception:
        return {}

    def safe(tag: str) -> str:
        try:
            val = getattr(dcm, tag, None)
            return str(val).strip() if val is not None else ''
        except Exception:
            return ''

    def parse_age(raw: str) -> str:
        raw = raw.strip().upper()
        if raw.endswith('Y'):
            return raw[:-1] + ' años'
        if raw.endswith('M'):
            return raw[:-1] + ' meses'
        if raw.endswith('D'):
            return raw[:-1] + ' días'
        return raw

    def parse_date(raw: str) -> str:
        if len(raw) == 8 and raw.isdigit():
            return f'{raw[6:8]}/{raw[4:6]}/{raw[:4]}'
        return raw

    VIEW_MAP = {
        'PA':  'Posteroanterior (PA)',
        'AP':  'Anteroposterior (AP)',
        'LAT': 'Lateral',
        'LL':  'Lateral izquierda',
        'RL':  'Lateral derecha',
    }
    SEX_MAP = {'M': 'Macho', 'F': 'Hembra', 'O': 'Otro'}
    MODALITY_MAP = {
        'CR': 'Radiografía computarizada (CR)',
        'DX': 'Radiografía digital (DX)',
        'RG': 'Radiografía convencional (RG)',
    }

    view_raw     = safe('ViewPosition').upper()
    sex_raw      = safe('PatientSex').upper()
    modality_raw = safe('Modality').upper()
    age_raw      = safe('PatientAge')
    date_raw     = safe('StudyDate')

    meta = {
        'body_part':     safe('BodyPartExamined').title() or None,
        'view_position': VIEW_MAP.get(view_raw, view_raw) or None,
        'modality':      MODALITY_MAP.get(modality_raw, modality_raw) or None,
        'patient_age':   parse_age(age_raw) if age_raw else None,
        'patient_sex':   SEX_MAP.get(sex_raw, sex_raw) or None,
        'study_date':    parse_date(date_raw) if date_raw else None,
        'institution':   safe('InstitutionName') or None,
        'kvp':           safe('KVP') or None,
    }
    return {k: v for k, v in meta.items() if v}
