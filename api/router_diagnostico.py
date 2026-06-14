import json
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Optional

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
_MODEL              = os.getenv('OPENAI_MODEL', 'gpt-4o')
_MAX_TOKENS         = int(os.getenv('OPENAI_MAX_TOKENS', '4096'))
_MAX_CONTINUACIONES = 3

LAB_EXTS = {'.pdf'} | IMAGE_EXTENSIONS


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


async def _sse_stream(messages: list, escenario: str) -> AsyncGenerator[str, None]:
    client = _get_client()

    yield f"data: {json.dumps({'type': 'meta', 'escenario': escenario, 'modelo': _MODEL})}\n\n"

    current_messages = list(messages)
    continuaciones = 0

    while True:
        stream = await client.chat.completions.create(
            model=_MODEL,
            messages=current_messages,
            max_tokens=_MAX_TOKENS,
            temperature=0.3,
            stream=True,
        )

        parte_texto = ''
        finish_reason = None

        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if choice:
                delta = choice.delta.content or ''
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
    motivo_consulta: str  = Form(..., min_length=10),
    especie: str          = Form('Perro'),
    edad: Optional[str]   = Form(None),
    sexo: Optional[str]   = Form(None),
    peso: Optional[str]   = Form(None),
    archivo_radiografia: Optional[UploadFile] = File(None),
    archivo_hemograma:   Optional[UploadFile] = File(None),
):
    try:
        _get_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # ── Radiografía ─────────────────────────────────────────────────────────
    radio_result: Optional[RadiografiResult] = None
    image_b64:   Optional[str]              = None

    if archivo_radiografia and archivo_radiografia.filename:
        ext = Path(archivo_radiografia.filename).suffix.lower()
        if ext not in RADIO_EXTS:
            raise HTTPException(400, f'Formato de radiografía no soportado: "{ext}".')
        raw_radio = await archivo_radiografia.read()
        if not raw_radio:
            raise HTTPException(400, 'El archivo de radiografía está vacío.')

        try:
            image_b64 = image_to_b64(raw_radio, ext)
        except Exception as e:
            raise HTTPException(422, f'No se pudo procesar la imagen: {e}')

        try:
            pred = predictor.predict(raw_radio, archivo_radiografia.filename)
            radio_result = RadiografiResult(
                diagnoses=pred.get('diagnoses', []),
                predictions=pred.get('predictions', {}),
            )
        except Exception:
            radio_result = RadiografiResult()

    # ── Hemograma ────────────────────────────────────────────────────────────
    lab_result: Optional[LaboratorioResult] = None

    if archivo_hemograma and archivo_hemograma.filename:
        ext = Path(archivo_hemograma.filename).suffix.lower()
        if ext not in LAB_EXTS:
            raise HTTPException(400, f'Formato de hemograma no soportado: "{ext}".')
        raw_lab = await archivo_hemograma.read()
        if not raw_lab:
            raise HTTPException(400, 'El archivo de hemograma está vacío.')

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(raw_lab)
            tmp_path = tmp.name
        try:
            resultado = extraer_laboratorio(tmp_path, especie=especie)
            if 'error' not in resultado:
                lab_result = _parse_lab(resultado)
            else:
                raise HTTPException(422, resultado['error'])
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f'Error al procesar hemograma: {e}')
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
        laboratorio=lab_result,
    )

    escenario, messages = build_messages(req, image_b64)

    return StreamingResponse(
        _sse_stream(messages, escenario),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )
