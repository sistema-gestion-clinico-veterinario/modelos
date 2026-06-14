from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from config import ALLOWED_EXTENSIONS, CLASSES, THRESHOLDS
from predictor import predictor

router = APIRouter()


@router.get('/health')
def health():
    return {
        'status':       'ok',
        'model_loaded': predictor.model is not None
    }


@router.get('/info')
def info():
    return {
        'model':               'densenet121',
        'checkpoint':          'best_densenet_v2.pth',
        'classes':             CLASSES,
        'thresholds':          THRESHOLDS,
        'auc_macro':           0.8844,
        'f1_macro_optimizado': 0.6358,
        'formatos_aceptados':  sorted(ALLOWED_EXTENSIONS)
    }


@router.post('/predict/radiografia')
async def predict_radiografia(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower() if file.filename else ''
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f'Formato no soportado: {ext}. Usa DICOM (.dcm), PNG o JPG.'
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail='El archivo está vacío.')

    try:
        return predictor.predict(data, file.filename or 'upload')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
