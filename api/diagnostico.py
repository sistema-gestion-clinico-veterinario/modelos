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
    'Eres un asistente de redacción clínica veterinaria basado en inteligencia artificial. '
    'Tu función es generar reportes preliminares de apoyo para el médico veterinario responsable. '
    'NO analizas imágenes directamente ni emites diagnósticos definitivos. '
    'Trabajas exclusivamente con: (1) las predicciones estructuradas de un modelo CNN '
    '(DenseNet-121 entrenado en radiología veterinaria), (2) los datos clínicos ingresados '
    'por el veterinario, y (3) los resultados de laboratorio cuando estén disponibles. '
    'Todo reporte que generes debe dejar claro que es un instrumento de apoyo a la '
    'interpretación clínica y que requiere validación del médico veterinario responsable. '
    'REGLAS DE REDACCIÓN — debes cumplirlas siempre:\n'
    '1. INICIA con un encabezado del paciente usando exactamente los datos recibidos: '
    'nombre, especie, edad, sexo y peso. '
    'Ejemplo: "**Paciente:** BRAKO | Perro | Macho | 6 años 9 meses | 12.5 kg"\n'
    '2. Cuando incluyas hallazgos radiológicos, cita la PROBABILIDAD EXACTA del CNN '
    '(ej. "Patrón alveolar — probabilidad 78 %") y menciona las limitaciones del modelo.\n'
    '3. Cuando incluyas laboratorio, cita los VALORES NUMÉRICOS con unidad y rango de referencia. '
    'No generalices: escribe "Hematocrito 18 % (ref. 37-55 %)" en lugar de "anemia grave".\n'
    '4. Relaciona EXPLÍCITAMENTE el perfil del paciente (especie, edad, sexo, peso) '
    'con cada hallazgo: indica por qué ese perfil hace un hallazgo más o menos probable.\n'
    '5. Cada consulta en el historial tiene el encabezado "=== CONSULTA N | FECHA: DD de MMMM de YYYY ===". '
    'DEBES citar esa fecha exacta cada vez que hagas referencia a esa consulta, '
    'aunque solo haya una. Formato: "El DD de MMMM de YYYY, el paciente fue presentado por...". '
    'Nunca uses "en la primera consulta", "recientemente" o "posteriormente" sin la fecha exacta.\n'
    '6. Usa lenguaje clínico radiológico apropiado. Evita mencionar nombres de modelos de IA, '
    'arquitecturas de redes neuronales o términos técnicos de machine learning (no escribas '
    '"DenseNet", "CNN", "modelo", "umbral", "probabilidad del modelo" ni similares en el texto '
    'del reporte). Traduce las predicciones a hallazgos clínicos: en lugar de '
    '"el modelo detectó patrón alveolar con 78%", escribe '
    '"se identifican hallazgos compatibles con patrón alveolar".\n'
    '7. Evita frases como "el sistema diagnostica" o "el diagnóstico es". '
    'Usa: "los hallazgos son compatibles con", "se sugiere considerar", '
    '"el médico veterinario deberá valorar".\n'
    '8. Responde siempre en español. No inventes datos ausentes en la información. '
    'Si la información es insuficiente, indícalo explícitamente.'
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
    n_radiografias: int = 0
    radiografia: Optional[RadiografiResult] = None
    laboratorio: Optional[List[LaboratorioResult]] = None
    dicom_metadata: Optional[Dict[str, Any]] = None
    image_quality: Optional[Dict[str, Any]] = None


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


def build_prompt(req: DiagnosticoRequest) -> Tuple[str, str]:
    """Devuelve (escenario, user_prompt)."""
    con_lab   = _tiene_lab(req.laboratorio)
    con_radio = _tiene_radio(req.radiografia) or req.n_radiografias > 0

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

    if req.dicom_metadata:
        meta = req.dicom_metadata
        p.append('## METADATOS DEL ESTUDIO RADIOLÓGICO\n')
        labels = {
            'body_part':     'Región evaluada',
            'view_position': 'Proyección',
            'modality':      'Modalidad',
            'study_date':    'Fecha del estudio',
            'institution':   'Institución',
            'kvp':           'Tensión (kVp)',
            'patient_age':   'Edad (DICOM)',
            'patient_sex':   'Sexo (DICOM)',
        }
        for key, label in labels.items():
            if key in meta:
                p.append(f'- {label}: {meta[key]}')
        p.append('')

        TORAX_KEYWORDS = {'chest', 'thorax', 'torax', 'thoracic', 'lung', 'pulmon', 'tórax', 'pulmón'}
        body_raw = meta.get('body_part', '').lower()
        if body_raw and not any(kw in body_raw for kw in TORAX_KEYWORDS):
            region = meta['body_part']
            p.append(
                f'*Nota para la redacción — IMPORTANTE: los metadatos DICOM identifican esta imagen '
                f'como **{region}**, NO como tórax. En el reporte DEBES nombrar explícitamente esta '
                f'región con esa palabra (ejemplo de frase a usar: "Los metadatos del estudio indican '
                f'que la imagen corresponde a {region.lower()} y no al tórax"). No te limites a decir '
                f'"otra región" o "región diferente": usa el nombre concreto **{region}**. Luego: '
                f'(a) proporciona el análisis disponible con los resultados del modelo, aclarando que '
                f'no son fiables para esta región; '
                f'(b) explica que el modelo CNN fue entrenado exclusivamente en radiología torácica '
                f'veterinaria, por lo que sus predicciones sobre {region.lower()} carecen de validez '
                f'clínica; '
                f'(c) indica que no es posible corroborar hallazgos para {region.lower()} con este '
                f'modelo y que el médico veterinario debe solicitar la evaluación radiológica '
                f'apropiada para esa zona.*\n'
            )

    if con_radio:
        radio = req.radiografia
        n = req.n_radiografias
        s = 's' if n > 1 else ''
        p.append(f'## PREDICCIONES DEL MODELO CNN — {n} radiografía{s} procesada{s}\n')
        p.append('*Modelo: DenseNet-121 entrenado en radiología veterinaria torácica. '
                 'Las predicciones son probabilísticas y no constituyen diagnóstico definitivo.*\n')

        if req.dicom_metadata:
            proy = req.dicom_metadata.get('view_position')
            mod  = req.dicom_metadata.get('modality')
            if proy or mod:
                datos_estudio = ' · '.join(filter(None, [
                    f'Proyección: {proy}' if proy else None,
                    f'Modalidad: {mod}' if mod else None,
                ]))
                p.append(
                    f'*Nota para la redacción: el estudio tiene estos datos técnicos — {datos_estudio}. '
                    f'DEBES iniciar la sección de Hallazgos radiológicos con esta línea exacta: '
                    f'"**Datos del estudio:** {datos_estudio}". Luego continúa con los hallazgos.*\n'
                )

        quality = req.image_quality
        if quality and quality.get('issues'):
            issues_txt = '; '.join(quality['issues'])
            p.append(
                f'*Nota para la redacción: el análisis visual de la imagen PNG/JPG detectó '
                f'las siguientes anomalías: {issues_txt}. '
                f'En el reporte menciona que la imagen presenta características que no son '
                f'típicas de una radiografía torácica estándar, proporciona el análisis '
                f'disponible con los resultados del modelo, pero indica que no es posible '
                f'confirmar con certeza que corresponde a un estudio radiológico torácico '
                f'y que el médico veterinario debe verificar el archivo antes de interpretar '
                f'estos hallazgos.*\n'
            )
        elif radio and radio.predictions:
            max_prob = max((v.get('probability', 0) for v in radio.predictions.values()), default=0)
            if max_prob < 0.20:
                p.append(
                    '*Nota para la redacción: las probabilidades del análisis son muy bajas '
                    f'en todas las categorías (máximo: {max_prob:.0%}), lo que puede indicar '
                    'que la imagen no corresponde a una radiografía torácica válida o que la '
                    'calidad técnica es insuficiente. Menciona esta limitación en el reporte '
                    'e indica que el médico veterinario debe verificar el archivo.*\n'
                )

        positivos = [d for d in radio.diagnoses if d != 'no_finding'] if radio else []
        if positivos:
            p.append('**Hallazgos con probabilidad sobre umbral diagnóstico:**')
            for d in positivos:
                nombre = CLASES_ES.get(d, d)
                info   = radio.predictions.get(d, {})
                prob   = info.get('probability', 0)
                thr    = info.get('threshold', 0.5)
                p.append(f'  - {nombre}: {prob:.0%} de probabilidad (umbral: {thr:.0%})')
        else:
            p.append('**Resultado CNN:** Sin hallazgos patológicos sobre umbral diagnóstico')

        liminares = [
            (cls, data) for cls, data in (radio.predictions.items() if radio else [])
            if cls not in (radio.diagnoses if radio else []) and cls != 'no_finding'
            and data.get('probability', 0) >= 0.25
        ]
        if liminares:
            p.append('**Hallazgos limítrofes (por debajo del umbral, requieren valoración clínica):**')
            for cls, data in liminares:
                p.append(f'  - {CLASES_ES.get(cls, cls)}: {data["probability"]:.0%}')

        p.append('\n**Limitaciones del modelo CNN:**')
        p.append('  - Entrenado para radiografía torácica; aplicación en otras regiones reduce precisión')
        p.append(f'  - {"Una sola proyección disponible" if n == 1 else f"{n} proyecciones procesadas de forma independiente"}')
        p.append('  - No incluye información de región anatómica ni posicionamiento del paciente')
        p.append('  - Resultados deben ser validados por el médico veterinario responsable')
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

    AVISO = (
        'Recuerda: eres un asistente de redacción clínica. '
        'No emitas diagnóstico definitivo. Usa frases como "los hallazgos son compatibles con", '
        '"se sugiere considerar" o "el médico veterinario deberá valorar". '
        'Finaliza el reporte con: "**Nota:** Este reporte es preliminar y de apoyo. '
        'Requiere validación del médico veterinario responsable."'
    )

    CRONOLOGIA = (
        '**Resumen del paciente y evolución cronológica** — indica especie, edad, sexo y peso. '
        'El historial incluye una o más consultas; cada una tiene el encabezado '
        '"=== CONSULTA N | FECHA: DD de MMMM de YYYY ===". '
        'DEBES usar esa fecha exacta cada vez que menciones esa consulta. '
        'Formato obligatorio: "El DD de MMMM de YYYY, el paciente fue presentado por [motivo]. '
        '[Signos, examen físico, hallazgos relevantes]." '
        'Si hay más de una consulta, narra cada una con su fecha exacta y destaca cambios '
        'entre visitas (mejoría, deterioro, nuevos síntomas). '
        'NUNCA uses frases como "en la primera consulta", "posteriormente" o "recientemente" '
        'sin acompañarlas de la fecha exacta.'
    )

    tareas = {
        'HC_solo': (
            'Redacta un reporte preliminar de apoyo con la siguiente estructura:\n'
            f'1. {CRONOLOGIA}\n'
            '2. **Impresión clínica orientativa** — hallazgos compatibles con, ordenados '
            'de mayor a menor probabilidad; argumenta con el perfil del paciente.\n'
            '3. **Estudios complementarios sugeridos** (laboratorio, imagen u otros)\n'
            '4. **Recomendaciones de manejo inicial**\n'
            '5. **Signos de alarma** que requieren atención veterinaria inmediata\n'
            f'6. {AVISO}'
        ),
        'HC_Hemograma': (
            'Redacta un reporte preliminar de apoyo con la siguiente estructura:\n'
            f'1. {CRONOLOGIA}\n'
            '2. **Interpretación de laboratorio** — cita cada parámetro alterado con su valor '
            'exacto y rango de referencia; describe su significado clínico para este perfil '
            'de paciente (especie, edad, sexo, peso).\n'
            '3. **Impresión clínica orientativa** — hallazgos compatibles con, ordenados '
            'de mayor a menor probabilidad; argumentados con valores del laboratorio.\n'
            '4. **Estudios complementarios sugeridos**\n'
            '5. **Recomendaciones terapéuticas orientativas** (a validar por el veterinario)\n'
            '6. **Limitaciones del análisis**\n'
            f'7. {AVISO}'
        ),
        'HC_Radiografia': (
            'Redacta un reporte radiológico preliminar de apoyo con la siguiente estructura:\n'
            f'1. {CRONOLOGIA}\n'
            '2. **Hallazgos radiológicos** — basándote ÚNICAMENTE en los resultados del '
            'análisis radiológico proporcionados (no analices ninguna imagen directamente). '
            'Si el análisis no reporta hallazgos patológicos significativos, NO lo presentes '
            'como una respuesta vacía. En cambio: (a) indica que el estudio radiológico '
            'torácico no evidenció alteraciones patológicas en las estructuras evaluadas '
            '(campo pulmonar, silueta cardíaca, mediastino, diafragma); (b) explica el valor '
            'clínico de ese resultado negativo — qué condiciones quedan descartadas o menos '
            'probables (ej. neumonía, efusión pleural, cardiomegalia, masas torácicas); '
            '(c) señala cómo ese hallazgo negativo redirige la búsqueda clínica hacia otros '
            'sistemas (digestivo, hematológico, metabólico) dado los signos del paciente. '
            'Si hay hallazgos positivos, descríbelos con localización y relevancia clínica. '
            'NO menciones nombres de modelos de IA ni valores de probabilidad numérica.\n'
            '3. **Limitaciones del estudio** — número de proyecciones disponibles, '
            'región anatómica evaluada y otras consideraciones clínicas relevantes.\n'
            '4. **Impresión clínica orientativa** — hallazgos compatibles con, en orden '
            'de mayor a menor probabilidad, relacionados con los signos clínicos.\n'
            '5. **Estudios complementarios sugeridos** (proyecciones adicionales, laboratorio)\n'
            '6. **Recomendaciones orientativas** (a validar por el veterinario)\n'
            f'7. {AVISO}'
        ),
        'HC_Hemograma_Radiografia': (
            'Redacta un reporte clínico preliminar de apoyo integrado con la siguiente estructura:\n'
            f'1. {CRONOLOGIA}\n'
            '2. **Hallazgos radiológicos** — basándote ÚNICAMENTE en los resultados del '
            'análisis radiológico proporcionados (no analices ninguna imagen directamente). '
            'Si no se reportan hallazgos patológicos: indica qué estructuras torácicas fueron '
            'evaluadas y que se encuentran sin alteraciones evidentes; explica qué condiciones '
            'descarta ese resultado negativo y cómo redirige el enfoque diagnóstico hacia '
            'otros sistemas dado los signos clínicos del paciente. '
            'Si hay hallazgos positivos, descríbelos con localización y relevancia clínica. '
            'NO menciones modelos de IA ni valores de probabilidad numérica.\n'
            '3. **Interpretación de laboratorio** — cada parámetro alterado con valor exacto '
            'y rango de referencia; significado clínico para este perfil.\n'
            '4. **Correlación clínica integrada** — integra hallazgos radiológicos, '
            'laboratorio e historia clínica; explica cómo se relacionan entre sí.\n'
            '5. **Impresión clínica orientativa** — hallazgos compatibles con, ordenados '
            'de mayor a menor probabilidad; fundamentados en los datos reales.\n'
            '6. **Estudios complementarios sugeridos** (si aplica)\n'
            '7. **Recomendaciones orientativas** (medicación, seguimiento, monitoreo)\n'
            '8. **Limitaciones del estudio**\n'
            f'9. {AVISO}'
        ),
    }
    p.append(tareas[escenario])

    return escenario, '\n'.join(p)


def build_messages(req: DiagnosticoRequest) -> Tuple[str, list]:
    """Devuelve (escenario, messages list para OpenAI)."""
    escenario, text = build_prompt(req)
    return escenario, [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': text},
    ]
