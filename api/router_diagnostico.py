import json
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from diagnostico import (
    DiagnosticoRequest,
    LaboratorioResult,
    RadiografiResult,
    RADIO_EXTS,
    build_messages,
    image_to_b64,
)
from laboratorio import IMAGE_EXTENSIONS, extraer_laboratorio
from predictor import predictor

router_diag = APIRouter(prefix='/ia', tags=['diagnostico'])

_client: AsyncOpenAI | None = None
_ocr_reader = None
_MODEL              = os.getenv('OPENAI_MODEL', 'gpt-4o')
_MAX_TOKENS         = int(os.getenv('OPENAI_MAX_TOKENS', '4096'))
_MAX_CONTINUACIONES = 3

LAB_EXTS = {'.pdf'} | IMAGE_EXTENSIONS


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['es', 'en'], gpu=False, verbose=False)
    return _ocr_reader


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            raise RuntimeError('OPENAI_API_KEY no está configurada.')
        _client = AsyncOpenAI(api_key=api_key)
    return _client


def _parse_lab(raw: dict) -> Optional[LaboratorioResult]:
    try:
        return LaboratorioResult.model_validate(raw)
    except Exception:
        return None


def _combinar_predicciones(all_preds: List[dict]) -> RadiografiResult:
    """Toma el máximo de probabilidad por clase entre todas las radiografías."""
    combined: Dict[str, dict] = {}
    for pred in all_preds:
        for cls, data in pred.get('predictions', {}).items():
            if cls not in combined or data['probability'] > combined[cls]['probability']:
                combined[cls] = dict(data)
    diagnoses = [cls for cls, data in combined.items() if data.get('positive', False)]
    return RadiografiResult(diagnoses=diagnoses, predictions=combined)


async def _sse_stream(messages: list, escenario: str) -> AsyncGenerator[str, None]:
    client = _get_client()

    yield f"data: {json.dumps({'type': 'meta', 'escenario': escenario, 'modelo': _MODEL})}\n\n"

    current_messages = list(messages)
    continuaciones   = 0

    while True:
        stream = await client.chat.completions.create(
            model=_MODEL,
            messages=current_messages,
            max_tokens=_MAX_TOKENS,
            temperature=0.3,
            stream=True,
        )

        parte_texto  = ''
        finish_reason = None

        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if choice:
                delta         = choice.delta.content or ''
                finish_reason = choice.finish_reason
                if delta:
                    parte_texto += delta
                    yield f"data: {json.dumps({'type': 'chunk', 'text': delta})}\n\n"

        if finish_reason == 'length' and continuaciones < _MAX_CONTINUACIONES:
            continuaciones += 1
            yield f"data: {json.dumps({'type': 'continuando', 'parte': continuaciones})}\n\n"
            current_messages = current_messages + [
                {'role': 'assistant', 'content': parte_texto},
                {'role': 'user', 'content': 'Continúa desde donde te quedaste, sin repetir lo ya escrito.'},
            ]
            continue

        break

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@router_diag.post('/diagnostico')
async def generar_diagnostico(
    motivo_consulta: str         = Form(..., min_length=10),
    especie: str                 = Form('Perro'),
    edad: Optional[str]          = Form(None),
    sexo: Optional[str]          = Form(None),
    peso: Optional[str]          = Form(None),
    archivo_radiografia: List[UploadFile] = File(default=[]),
    archivo_hemograma:   List[UploadFile] = File(default=[]),
):
    try:
        _get_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    radios = [f for f in archivo_radiografia if f and f.filename]
    labs   = [f for f in archivo_hemograma   if f and f.filename]

    # ── Radiografías (múltiples) ─────────────────────────────────────────────
    images_b64:  List[str] = []
    radio_result: Optional[RadiografiResult] = None

    if radios:
        all_preds: List[dict] = []
        for f in radios:
            ext = Path(f.filename).suffix.lower()
            if ext not in RADIO_EXTS:
                raise HTTPException(400, f'Formato de radiografía no soportado: "{ext}".')
            raw = await f.read()
            if not raw:
                continue
            try:
                images_b64.append(image_to_b64(raw, ext))
            except Exception as e:
                raise HTTPException(422, f'No se pudo procesar {f.filename}: {e}')
            try:
                pred = predictor.predict(raw, f.filename)
                all_preds.append(pred)
            except Exception:
                pass

        if all_preds:
            radio_result = _combinar_predicciones(all_preds)
        elif images_b64:
            radio_result = RadiografiResult()

    # ── Hemogramas (múltiples) ───────────────────────────────────────────────
    lab_results: List[LaboratorioResult] = []

    for f in labs:
        ext = Path(f.filename).suffix.lower()
        if ext not in LAB_EXTS:
            raise HTTPException(400, f'Formato de hemograma no soportado: "{ext}".')
        raw = await f.read()
        if not raw:
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            reader = _get_ocr_reader() if ext in IMAGE_EXTENSIONS else None
            resultado = extraer_laboratorio(tmp_path, especie=especie, reader=reader)
            if 'error' not in resultado:
                parsed = _parse_lab(resultado)
                if parsed:
                    lab_results.append(parsed)
        except Exception:
            pass
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ── Construir request y mensajes ─────────────────────────────────────────
    req = DiagnosticoRequest(
        motivo_consulta=motivo_consulta,
        especie=especie,
        edad=edad,
        sexo=sexo,
        peso=peso,
        radiografia=radio_result,
        laboratorio=lab_results if lab_results else None,
    )

    escenario, messages = build_messages(req, images_b64)

    return StreamingResponse(
        _sse_stream(messages, escenario),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )
