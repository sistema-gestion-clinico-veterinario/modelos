import tempfile
from pathlib import Path
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from laboratorio import extraer_laboratorio, IMAGE_EXTENSIONS

router_lab = APIRouter(prefix='/ia', tags=['laboratorio'])

ALLOWED_EXTENSIONS = {'.pdf'} | IMAGE_EXTENSIONS

_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        print('[laboratorio] Inicializando EasyOCR...')
        _ocr_reader = easyocr.Reader(['es', 'en'], gpu=False)
        print('[laboratorio] EasyOCR listo.')
    return _ocr_reader


@router_lab.post('/laboratorio')
async def analizar_laboratorio(
    archivo: UploadFile = File(...),
    especie: str = Form('Perro'),
):
    ext = Path(archivo.filename).suffix.lower() if archivo.filename else ''
    if not ext or ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f'Formato no soportado: "{ext}". Acepta: PDF, JPG, PNG, BMP, TIFF.',
        )

    data = await archivo.read()
    if not data:
        raise HTTPException(status_code=400, detail='El archivo está vacío.')

    reader = _get_ocr_reader() if ext in IMAGE_EXTENSIONS else None

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        resultado = extraer_laboratorio(tmp_path, especie=especie, reader=reader)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if 'error' in resultado:
        codigo = 422 if resultado.get('tipo') in {'pdf_escaneado', 'voucher_imagen'} else 400
        raise HTTPException(status_code=codigo, detail=resultado['error'])

    return resultado
