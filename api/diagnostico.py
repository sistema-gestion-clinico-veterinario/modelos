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
    'Eres un asistente de inteligencia artificial especializado en medicina veterinaria, '
    'actuando como veterinario clínico experto con especialización en diagnóstico por imágenes '
    '(radiología torácica y abdominal) y medicina interna de pequeños animales. '
    'Todas las imágenes que recibirás son radiografías de pacientes animales (perros, gatos u '
    'otras especies) tomadas en contexto clínico veterinario. '
    'Nunca son imágenes de pacientes humanos. '
    'Cuando se te adjunta una imagen radiográfica (DICOM convertido a PNG, o JPG/PNG), '
    'la analizas visualmente de forma directa e integras tus hallazgos con los resultados '
    'estructurados del clasificador DenseNet-121 (clases: patrón alveolar, patrón bronquial, '
    'efusión pleural, cardiomegalia, sin hallazgos). '
    'Integras historia clínica, resultados de laboratorio y hallazgos radiológicos para '
    'generar diagnósticos diferenciales precisos y planes de manejo veterinario. '
    'REGLAS DE REDACCIÓN — debes cumplirlas siempre:\n'
    '1. INICIA tu respuesta con un encabezado breve del paciente usando exactamente los datos '
    'recibidos: especie, edad, sexo y peso. Ejemplo: '
    '"**Paciente:** Perro | Macho | 6 años 9 meses | 12.5 kg"\n'
    '2. Cita los VALORES NUMÉRICOS EXACTOS del laboratorio (con unidad y rango de referencia '
    'si están disponibles) al momento de interpretarlos. No generalices: escribe '
    '"Hematocrito 18 % (ref. 37-55 %)" en lugar de solo "anemia grave".\n'
    '3. Relaciona EXPLÍCITAMENTE las características del paciente (especie, edad, sexo, peso) '
    'con cada diagnóstico diferencial. Indica por qué este perfil hace que un diagnóstico '
    'sea más o menos probable.\n'
    '4. Si hay varias consultas en el historial, haz un RECORRIDO CRONOLÓGICO: describe la '
    'evolución de los signos, los cambios en parámetros entre visitas y la respuesta a '
    'tratamientos previos antes de dar el diagnóstico diferencial.\n'
    '5. Responde siempre en español, de forma clara y estructurada. '
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
    nombre_paciente: Optional[str] = None
    edad: Optional[str] = None
    sexo: Optional[str] = None
    peso: Optional[str] = None
    radiografia: Optional[RadiografiResult] = None
    laboratorio: Optional[List[LaboratorioResult]] = None


# ── Schema de salida ─────────────────────────────────────────────────────────

class DiagnosticoResponse(BaseModel):
    escenario: str
    respuesta: str
    modelo: str
    tokens_prompt: int
    tokens_respuesta: int
    tokens_total: int


# ── Construcción del prompt ──────────────────────────────────────────────────

def _tiene_lab(lab: Optional[List[LaboratorioResult]]) -> bool:
    if not lab:
        return False
    return any(
        (r.hematologia and r.hematologia.parametros)
        or (r.quimica and r.quimica.parametros)
        or (r.serologia and r.serologia.parametros)
        for r in lab
    )


def _tiene_radio(radio: Optional[RadiografiResult]) -> bool:
    if radio is None:
        return False
    return bool(radio.diagnoses or radio.predictions)


def build_prompt(req: DiagnosticoRequest, n_imagenes: int = 0) -> Tuple[str, str]:
    """Devuelve (escenario, user_prompt)."""
    con_lab   = _tiene_lab(req.laboratorio)
    con_radio = _tiene_radio(req.radiografia) or n_imagenes > 0

    if con_lab and con_radio:
        escenario = 'HC_Hemograma_Radiografia'
    elif con_lab:
        escenario = 'HC_Hemograma'
    elif con_radio:
        escenario = 'HC_Radiografia'
    else:
        escenario = 'HC_solo'

    p: List[str] = ['## DATOS DEL PACIENTE\n']
    if req.nombre_paciente:
        p.append(f'**Nombre:** {req.nombre_paciente}')
    p.append(f'**Especie:** {req.especie}')
    if req.edad:
        p.append(f'**Edad:** {req.edad}')
    if req.sexo:
        p.append(f'**Sexo:** {req.sexo}')
    if req.peso:
        p.append(f'**Peso:** {req.peso}')
    p.append(f'\n## HISTORIA CLÍNICA\n')
    p.append(f'{req.motivo_consulta}\n')

    if con_radio:
        radio = req.radiografia
        p.append('## HALLAZGOS RADIOLÓGICOS (DenseNet-121)\n')
        if n_imagenes > 0:
            s = 's' if n_imagenes > 1 else ''
            p.append(f'*Se adjuntan {n_imagenes} imagen{s} radiográfica{s} para análisis visual directo.*\n')
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
        labs = req.laboratorio
        total = len(labs)
        titulo = f'## RESULTADOS DE LABORATORIO ({total} archivos)\n' if total > 1 else '## RESULTADOS DE LABORATORIO\n'
        p.append(titulo)

        for idx, lab in enumerate(labs):
            if total > 1:
                p.append(f'### Hemograma {idx + 1}\n')

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
            'Con base únicamente en la historia clínica proporcionada, redacta tu análisis así:\n'
            '1. **Resumen del paciente** — repite especie, edad, sexo y peso del encabezado; '
            'luego resume los motivos de consulta y la evolución cronológica de los signos.\n'
            '2. **Diagnósticos diferenciales** — lista de mayor a menor probabilidad; para cada '
            'uno indica por qué los datos de ESTE paciente (especie, edad, sexo, peso, signos '
            'descritos) lo respaldan o lo hacen menos probable.\n'
            '3. **Exámenes complementarios recomendados** (laboratorio, imagen, etc.)\n'
            '4. **Manejo inicial sugerido**\n'
            '5. **Signos de alarma** que requieren atención inmediata'
        ),
        'HC_Hemograma': (
            'Con base en la historia clínica y los resultados de laboratorio, redacta tu análisis así:\n'
            '1. **Resumen del paciente** — especie, edad, sexo, peso; evolución cronológica de '
            'los signos clínicos a lo largo de las consultas registradas.\n'
            '2. **Interpretación analítica del hemograma** — menciona CADA parámetro alterado '
            'con su valor exacto y el rango de referencia; explica su significado clínico para '
            'ESTE paciente (especie, edad, sexo, peso).\n'
            '3. **Diagnósticos diferenciales** — ordenados de mayor a menor probabilidad; '
            'argumenta con los valores del laboratorio y las características del paciente.\n'
            '4. **Exámenes complementarios recomendados**\n'
            '5. **Plan terapéutico sugerido**\n'
            '6. **Pronóstico preliminar**'
        ),
        'HC_Radiografia': (
            'Con base en la historia clínica y los hallazgos radiológicos, redacta tu análisis así:\n'
            '1. **Resumen del paciente** — especie, edad, sexo, peso; evolución cronológica de '
            'los signos clínicos.\n'
            '2. **Interpretación radiológica detallada** — describe cada hallazgo (positivo o '
            'limítrofe) con su probabilidad; analiza qué estructuras anatómicas están afectadas '
            'y si el tamaño/peso del paciente influye en la presentación.\n'
            '3. **Diagnósticos diferenciales** — ordenados de mayor a menor probabilidad; '
            'argumenta con los hallazgos radiológicos y el perfil del paciente.\n'
            '4. **Exámenes complementarios recomendados** (laboratorio u otras imágenes)\n'
            '5. **Plan terapéutico sugerido**\n'
            '6. **Pronóstico preliminar**'
        ),
        'HC_Hemograma_Radiografia': (
            'Con base en la historia clínica, los hallazgos radiológicos y los resultados '
            'de laboratorio, redacta tu análisis así:\n'
            '1. **Resumen del paciente** — especie, edad, sexo, peso; evolución cronológica de '
            'los signos clínicos a lo largo de las consultas registradas.\n'
            '2. **Interpretación analítica del hemograma** — cada parámetro alterado con valor '
            'exacto y rango de referencia; significado clínico para ESTE paciente.\n'
            '3. **Interpretación radiológica** — cada hallazgo con su probabilidad; descripción '
            'anatómica y relevancia clínica.\n'
            '4. **Correlación clínico-radiológico-laboratorial** — integra los tres conjuntos '
            'de datos para construir la hipótesis diagnóstica más probable.\n'
            '5. **Diagnósticos diferenciales** — ordenados de mayor a menor probabilidad; '
            'fundamentados en los datos reales del paciente.\n'
            '6. **Exámenes complementarios adicionales** (si aplica)\n'
            '7. **Plan terapéutico integral** (medicación, seguimiento y monitoreo)\n'
            '8. **Pronóstico**'
        ),
    }
    p.append(tareas[escenario])

    return escenario, '\n'.join(p)


def build_messages(
    req: DiagnosticoRequest,
    images_b64: List[str] = [],
) -> Tuple[str, list]:
    """Devuelve (escenario, messages list para OpenAI)."""
    escenario, text = build_prompt(req, n_imagenes=len(images_b64))

    if images_b64:
        especie = req.especie or 'animal'
        nombre  = req.nombre_paciente or 'el paciente'
        aviso   = (
            f'A continuación se adjuntan {len(images_b64)} imagen(es) radiográfica(s) '
            f'de {nombre}, un(a) {especie}. '
            f'Son imágenes veterinarias de uso clínico. '
            f'Por favor analízalas como veterinario experto según las instrucciones anteriores.'
        )
        user_content: Any = [
            {'type': 'text', 'text': text},
            {'type': 'text', 'text': aviso},
        ]
        for b64 in images_b64:
            user_content.append({
                'type': 'image_url',
                'image_url': {'url': f'data:image/png;base64,{b64}', 'detail': 'high'},
            })
    else:
        user_content = text

    return escenario, [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': user_content},
    ]
