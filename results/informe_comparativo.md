# Informe Comparativo: pdfplumber vs EasyOCR

## Resumen por estrategia

tipo          % item  % resultado  % referencia  % flag  % completo
---------  --------  ------------  -------------  ------  -----------
easyocr        100.0          92.1           93.6    95.7         86.4
pdfplumber     100.0         100.0          100.0   100.0        100.0

## Detalle por archivo

archivo                                   tipo        total  completas  % item  % res  % ref  % flag  % completo
----------------------------------------  ---------  ------  ---------  ------  -----  -----  ------  -----------
AKIRA-CHAVARRY-2026-05-14-0914.pdf        pdfplumber     24         24   100.0  100.0  100.0   100.0        100.0
LUKI-CORRALES-2026-05-14-1724.pdf         pdfplumber     24         24   100.0  100.0  100.0   100.0        100.0
ZEUS-CABRERA-2026-05-24-1035.pdf          pdfplumber     24         24   100.0  100.0  100.0   100.0        100.0
16823793-d19c-4539-8d5d-196161e1199b.png  easyocr        20         19   100.0   95.0  100.0   100.0         95.0
23a07b2e-ec35-4d85-8599-9b9df839f9ec.jpg  easyocr        20         16   100.0   80.0   95.0   100.0         80.0
5ce50946-7def-4e60-8063-9a8bc6de3cd6.png  easyocr        20         15   100.0   85.0   80.0    95.0         75.0
5e39366e-9269-4181-98ea-9f6e4987312b.png  easyocr        20         15   100.0   90.0   95.0    80.0         75.0
a76eab4a-1ddb-4f3d-8aa7-b308f38090dc.jpg  easyocr        20         17   100.0   95.0   85.0   100.0         85.0
aaaa8592-ff9a-439a-83a4-8153aa0f2310.png  easyocr        20         20   100.0  100.0  100.0   100.0        100.0
b3403d54-7fc4-414c-9eb0-0058886202d5.png  easyocr        20         19   100.0  100.0  100.0    95.0         95.0

## Análisis

### pdfplumber: 100 % en todas las columnas

Los tres PDFs de IDEXX ProCyte Dx se procesan con precisión perfecta. La razón es estructural: son documentos generados digitalmente con texto seleccionable y tablas con coordenadas fijas. pdfplumber extrae el texto con su posición exacta, y el parseador token-based alinea cada fila contra las 3 tablas internas del PDF (WBC Diff, RBC, PLT). Como la estructura del PDF nunca varía entre informes IDEXX, no hay ambigüedad.

Las 24 filas del hemograma (entre parámetros y porcentajes) se extraen correctamente, incluyendo valores, rangos de referencia y flags H/L.

### EasyOCR: 86.4 % completo — tipos de error

EasyOCR alcanza 86.4 % de filas completas en promedio, con un rango de 75 % a 100 % según la calidad de la imagen. Los errores se clasifican en 4 categorías:

**1. Confusión de caracteres (más frecuente)**
- Dígitos: `L`→`4`, `I`→`1`, `O`→`0`, `S`→`5`, `B`→`8`
- Punto decimal perdido: `934` en vez de `9.34` (RBC), `94` en vez de `14.0` (HGB)
- Items: `BBC`→`RBC`, `6RAY`→`GRA%`, `HID`→`MID%`, `UPC`→`WBC`

**2. Notación de potencia de 10**
- `10^9/L` se lee como `1o )'9l`, `10*9`, `10'5`, `10 9/L`, `19*9`, etc.
- El módulo `_normalizar_potencias()` en corrector.py unifica 15+ variantes a `10^9/L` o `10^12/L`

**3. Unidades y símbolos**
- `g/dL` → `w*`, `W4 !`, `y/DL`, etc.
- `fL` → `AL`, `€L`, `(L`, etc.
- `%` → `{`, `*`, etc.

**4. Zona de flag**
- En imágenes anguladas o con columnas desplazadas, el flag H/L se fusiona con la referencia o desaparece
- Ejemplo: `5e39366e.png` solo detectó 16 flags de 20 (80 %)

### Imagen que falló por calidad de entrada

`WhatsApp Image 2026-06-06 at 23.05.56.jpeg` (77 KB, 900×1600 px, razón de compresión ~54.7:1) no está incluida en las métricas porque no existe ground truth, pero se documenta como caso de estudio. La imagen fue enviada por WhatsApp y re-comprimida por el servicio, destruyendo el detalle de los caracteres (varianza Laplaciana de 42.3, umbral de nitidez en 80).

Se probaron 8 variantes de preprocesado (denoising, LANCZOS, umbral adaptativo, estiramiento de contraste, morfología matemática, CLAHE con distintos parámetros) y ningún pipeline logró extraer más de 3 valores correctos. También se evaluó Real-ESRGAN 4x (super-resolución por deep learning), que tomó 438 segundos en CPU y no mejoró significativamente la tasa de acierto.

**Conclusión**: es una falla de calidad de la fuente, no del pipeline. El sistema no puede crear información que la compresión JPEG destruyó.

### Recomendación para producción

| Contexto | Estrategia | Precisión esperada |
|---|---|---|
| PDF IDEXX digital | pdfplumber (ruta automática) | 100 % |
| Voucher escaneado ≥ 1200 px, ≥ 100 KB | EasyOCR + preprocesado actual | 85-100 % |
| Imagen WhatsApp comprimida < 100 KB | Rechazar — pedir reenvío sin comprimir | — |
| Cualquier imagen | Control de calidad previo: varianza Laplaciana ≥ 80, ancho ≥ 1200 px | — |

Para producción se recomienda:
1. **Ruta automática** por extensión de archivo (ya implementada)
2. **Filtro de calidad de entrada** — medir varianza Laplaciana y resolución antes de procesar; rechazar imágenes bajo umbral con mensaje claro al usuario
3. **PaddleOCR** como alternativa a EasyOCR para imágenes de calidad media donde EasyOCR falla por caracteres rotos o bajo contraste
4. **Post-corrección por perfil** opcional — el usuario puede activar el rellenado de referencias desde perfil (perro/gato) cuando la extracción deje campos vacíos

### Hallazgos técnicos del Spike

- **Slope correction**: la corrección de inclinación por Y-projection (parejas de palabras entre columnas) redujo errores de alineamiento en vouchers escaneados torcidos
- **Desambiguación LYM/MID/GRA**: la primera ocurrencia se asigna a `#` (valor absoluto) y la segunda a `%` (porcentaje), siguiendo el orden del layout del voucher
- **Separación resultado+referencia**: 3 patrones regex en cascada (estricto, flexible, búsqueda global) recuperan valores cuando el OCR fusiona columnas adyacentes
- **Corrector de 110+ entradas**: el diccionario de correcciones captura las confusiones más frecuentes de EasyOCR con la tipografía del voucher Vargas Vet
- **Preprocesador adaptativo**: la detección de calidad (Laplacian var > 80) selecciona automáticamente entre CLAHE clip=2.0 (imágenes nítidas) y clip=4.0 (imágenes borrosas)
