from __future__ import annotations
import base64
import io
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


CLASES_ES: Dict[str, str] = {
    'alveolar_pattern': 'Patrón alveolar',
    'bronchial_pattern': 'Patrón bronquial',
    'pleural_effusion': 'Efusión pleural',
    'cardiomegaly': 'Cardiomegalia',
    'no_finding': 'Sin hallazgos patológicos',
}

SYSTEM_PROMPT = (
    'Eres un veterinario clínico experto con especialización en diagnóstico por imágenes '
    '(radiología torácica y abdominal) y medicina interna de pequeños animales. '
    'Cuando se te adjunta una imagen radiográfica (DICOM convertido a PNG, o JPG/PNG), '
    'la analizas visualmente de forma directa e integras tus hallazgos con los resultados '
    'estructurados del clasificador DenseNet-121 (clases: patrón alveolar, patrón bronquial, '
    'efusión pleural, cardiomegalia, sin hallazgos). '
    'Integras historia clínica, resultados de laboratorio y hallazgos radiológicos para '
    'generar diagnósticos diferenciales precisos y planes de manejo veterinario. '
    'Responde siempre en español, de forma clara y estructurada. '
    'No inventes datos que no estén presentes en la información proporcionada. '
    'Si la información es insuficiente para una conclusión firme, indícalo explícitamente.'
)


RADIO_EXTS: set[str] = {'.dcm', '.dicom', '.png', '.jpg', '.jpeg', '.bmp'}


def _dicom_to_png_bytes(raw: bytes) -> bytes:
    import numpy as np
    import pydicom
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as f:
        f.write(raw)
        tmp = f.name
    try:
        ds = pydicom.dcmread(tmp)
        arr = ds.pixel_array.astype(float)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * 255.0
        img = Image.fromarray(arr.astype('uint8'))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    finally:
        os.unlink(tmp)


def image_to_b64(raw: bytes, ext: str) -> str:
    from PIL import Image
    if ext in {'.dcm', '.dicom'}:
        png = _dicom_to_png_bytes(raw)
    else:
        img = Image.open(io.BytesIO(raw))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        png = buf.getvalue()
    return base64.b64encode(png).decode()


# ── Schemas de entrada ──────────────────────────────────────────────────────

class RadiografiResult(BaseModel):
    diagnoses: List[str] = []
    predictions: Dict[str, Any] = {}


class ParametroLab(BaseModel):
    test: str
    valor: float
    unidad: str = ''
    ref_min: Optional[float] = None
    ref_max: Optional[float] = None
    flag: Optional[str] = None
    estado: str = 'normal'


class SeccionLab(BaseModel):
    analizador: str = ''
    parametros: List[ParametroLab] = []


class SeroParm(BaseModel):
    test: str
    resultado: str
    positivo: bool = False


class SeroSec(BaseModel):
    analizador: str = ''
    parametros: List[SeroParm] = []


class LaboratorioResult(BaseModel):
    especie: str = ''
    hematologia: Optional[SeccionLab] = None
    quimica: Optional[SeccionLab] = None
    serologia: Optional[SeroSec] = None
    alertas: List[str] = []
    comentarios_clinicos: List[str] = []


class DiagnosticoRequest(BaseModel):
    motivo_consulta: str = Field(..., min_length=10,
                                 description='Motivo de consulta / anamnesis')
    especie: str = Field('Perro', description='Perro | Gato | Otro')
    edad: Optional[str] = None
    sexo: Optional[str] = None
    peso: Optional[str] = None
    radiografia: Optional[RadiografiResult] = None
    laboratorio: Optional[LaboratorioResult] = None


# ── Schema de salida ─────────────────────────────────────────────────────────

class DiagnosticoResponse(BaseModel):
    escenario: str
    respuesta: str
    modelo: str
    tokens_prompt: int
    tokens_respuesta: int
    tokens_total: int


# ── Construcción del prompt ──────────────────────────────────────────────────

def _tiene_lab(lab: Optional[LaboratorioResult]) -> bool:
    if lab is None:
        return False
    return bool(
        (lab.hematologia and lab.hematologia.parametros)
        or (lab.quimica and lab.quimica.parametros)
        or (lab.serologia and lab.serologia.parametros)
    )


def _tiene_radio(radio: Optional[RadiografiResult]) -> bool:
    if radio is None:
        return False
    return bool(radio.diagnoses or radio.predictions)


def build_prompt(req: DiagnosticoRequest, tiene_imagen: bool = False) -> Tuple[str, str]:
    """Devuelve (escenario, user_prompt)."""
    con_lab   = _tiene_lab(req.laboratorio)
    con_radio = _tiene_radio(req.radiografia)

    if con_lab and con_radio:
        escenario = 'HC_Hemograma_Radiografia'
    elif con_lab:
        escenario = 'HC_Hemograma'
    elif con_radio:
        escenario = 'HC_Radiografia'
    else:
        escenario = 'HC_solo'

    p: List[str] = ['## HISTORIA CLÍNICA\n']
    p.append(f'**Especie:** {req.especie}')
    if req.edad:
        p.append(f'**Edad:** {req.edad}')
    if req.sexo:
        p.append(f'**Sexo:** {req.sexo}')
    if req.peso:
        p.append(f'**Peso:** {req.peso}')
    p.append(f'\n**Motivo de consulta:**\n{req.motivo_consulta}\n')

    if con_radio:
        radio = req.radiografia
        p.append('## HALLAZGOS RADIOLÓGICOS (DenseNet-121)\n')
        if tiene_imagen:
            p.append('*La imagen radiográfica se adjunta para análisis visual directo.*\n')
        positivos = [d for d in radio.diagnoses if d != 'no_finding']
        if positivos:
            p.append('**Hallazgos positivos:**')
            for d in positivos:
                nombre = CLASES_ES.get(d, d)
                info   = radio.predictions.get(d, {})
                prob   = info.get('probability', 0)
                thr    = info.get('threshold', 0.5)
                p.append(f'  - {nombre} (probabilidad: {prob:.0%}, umbral diagnóstico: {thr:.0%})')
        else:
            p.append('**Resultado:** Sin hallazgos patológicos significativos')

        liminares = [
            (cls, data) for cls, data in radio.predictions.items()
            if cls not in radio.diagnoses and cls != 'no_finding'
            and data.get('probability', 0) >= 0.25
        ]
        if liminares:
            p.append('**Hallazgos limítrofes (no alcanzan umbral diagnóstico):**')
            for cls, data in liminares:
                p.append(f'  - {CLASES_ES.get(cls, cls)}: {data["probability"]:.0%}')
        p.append('')

    if con_lab:
        lab = req.laboratorio
        p.append('## RESULTADOS DE LABORATORIO\n')

        if lab.alertas:
            p.append('**Alertas:**')
            for a in lab.alertas:
                p.append(f'  ⚠ {a}')
            p.append('')

        if lab.hematologia and lab.hematologia.parametros:
            p.append('**Hematología:**')
            for param in lab.hematologia.parametros:
                flag = f' [{param.flag}]' if param.flag else ''
                p.append(f'  - {param.test}: {param.valor} {param.unidad}{flag} → {param.estado}')
            p.append('')

        if lab.quimica and lab.quimica.parametros:
            p.append('**Química sanguínea:**')
            for param in lab.quimica.parametros:
                flag = f' [{param.flag}]' if param.flag else ''
                p.append(f'  - {param.test}: {param.valor} {param.unidad}{flag} → {param.estado}')
            p.append('')

        if lab.serologia and lab.serologia.parametros:
            p.append('**Serología:**')
            for param in lab.serologia.parametros:
                pos = ' ⚠ POSITIVO' if param.positivo else ''
                p.append(f'  - {param.test}: {param.resultado}{pos}')
            p.append('')

        if lab.comentarios_clinicos:
            p.append('**Comentarios del laboratorio (IDEXX):**')
            for c in lab.comentarios_clinicos:
                p.append(f'  - {c}')
            p.append('')

    p.append('## SOLICITUD\n')

    tareas = {
        'HC_solo': (
            'Con base únicamente en la historia clínica proporcionada, indica:\n'
            '1. **Diagnósticos diferenciales** (ordenados de mayor a menor probabilidad)\n'
            '2. **Exámenes complementarios recomendados** (laboratorio, imagen, etc.)\n'
            '3. **Manejo inicial sugerido**\n'
            '4. **Signos de alarma** que requieren atención inmediata'
        ),
        'HC_Hemograma': (
            'Con base en la historia clínica y los resultados de laboratorio, indica:\n'
            '1. **Interpretación integrada de los hallazgos de laboratorio** en contexto clínico\n'
            '2. **Diagnósticos diferenciales** (ordenados de mayor a menor probabilidad)\n'
            '3. **Exámenes complementarios recomendados**\n'
            '4. **Plan terapéutico sugerido**\n'
            '5. **Pronóstico preliminar**'
        ),
        'HC_Radiografia': (
            'Con base en la historia clínica y los hallazgos radiológicos, indica:\n'
            '1. **Interpretación de los hallazgos radiológicos** en contexto clínico\n'
            '2. **Diagnósticos diferenciales** (ordenados de mayor a menor probabilidad)\n'
            '3. **Exámenes complementarios recomendados** (laboratorio u otras imágenes)\n'
            '4. **Plan terapéutico sugerido**\n'
            '5. **Pronóstico preliminar**'
        ),
        'HC_Hemograma_Radiografia': (
            'Con base en la historia clínica, los hallazgos radiológicos y los resultados '
            'de laboratorio, indica:\n'
            '1. **Correlación clínico-radiológico-laboratorial** (integración de todos los hallazgos)\n'
            '2. **Diagnósticos diferenciales** (ordenados de mayor a menor probabilidad)\n'
            '3. **Exámenes complementarios adicionales** (si aplica)\n'
            '4. **Plan terapéutico integral** (medicación, seguimiento y monitoreo)\n'
            '5. **Pronóstico**'
        ),
    }
    p.append(tareas[escenario])

    return escenario, '\n'.join(p)


def build_messages(
    req: DiagnosticoRequest,
    image_b64: Optional[str] = None,
) -> Tuple[str, list]:
    """Devuelve (escenario, messages list para OpenAI)."""
    escenario, text = build_prompt(req, tiene_imagen=image_b64 is not None)

    if image_b64:
        user_content: Any = [
            {'type': 'text', 'text': text},
            {
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/png;base64,{image_b64}',
                    'detail': 'high',
                },
            },
        ]
    else:
        user_content = text

    return escenario, [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': user_content},
    ]
