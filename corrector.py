import re

CORRECCIONES_ITEM = {
    "UBC": "WBC", "VBC": "WBC", "HBC": "WBC", "URC": "WBC", "LBC": "WBC", "NBC": "WBC",
    "LYMA": "LYM#", "LYM#": "LYM#", "LYN": "LYM%", "LYNY": "LYM%", "LYNY,": "LYM%",
    "LXM#": "LYM#", "LXM": "LYM#", "LXN": "LYM%", "LYME": "LYM#", "ZXHE": "LYM#", "LYK": "LYM#", "LYK:": "LYM#",
    "LY": "LYM#", "LYMA#": "LYM#", "LHIY": "LYM%", 
    "HID#": "MID#", "HID": "MID%", "HIDA": "MID%", "MID": "MID%", "MiD#": "MID#", "MiD": "MID%", "NID#": "MID#", "HIDE": "MID#", "HID/": "MID%", "ND3": "MID#", "ND": "MID#",
    "KIV": "MID%", 
    "GRAk": "GRA#", "GRAv": "GRA%", "GRAY": "GRA%", "GRAy": "GRA%", "GRAK": "GRA#", "GRAT": "GRA%", "GRAAE": "GRA#", "GRAX": "GRA#",
    "6RA:": "GRA#", "6RAK": "GRA#",  
    "HGB": "HGB", "HGb": "HGB", "H6B": "HGB", "468": "HGB", "48": "HGB",
    "68F": "HGB",  
    "HCHC": "MCHC", "HChC": "MCHC", "MCHO": "MCHC", "KCHC": "MCHC", "KOC": "MCHC", "KOX": "MCHC",
    "KB": "MCHC",  
    "HCh": "MCH", "CH": "MCH", "KOI": "MCH",
    "KH": "MCH",  
    "HCV": "MCV", "HV": "MCV", "HQV": "MCV", "KCV": "MCV", "KCY": "MCV",
    "KV": "MCV",  
    "HCi": "HCT", "HCI": "HCT", "HCT": "HCT", "FCT": "HCT", "KCT": "HCT",
    "K;": "HCT", 
    "RDV-V": "RDW-CV", "RDV(V": "RDW-CV", "RDV-(V": "RDW-CV", "RDV-CV": "RDW-CV", "RDU-QV": "RDW-CV", "RDU-WV": "RDW-CV", "RWU-WV": "RDW-CV", "ROY-CY": "RDW-CV", "ROX-CY": "RDW-CV",
    "RN-@V": "RDW-CV",  
    "RDU-SD": "RDW-SD", "RDU-SD)": "RDW-SD", "RDH-SD": "RDW-SD", "RD-SD": "RDW-SD", "RDV-SD": "RDW-SD", "RDX-SO": "RDW-SD", "ROX-SD": "RDW-SD",
    "RU-SB]": "RDW-SD", 
    "PLI": "PLT", "PLl": "PLT",
    "FLF": "PLT", 
    "PDU": "PDW", "PDH": "PDW", "PDw": "PDW", "PDu": "PDW", "POX": "PDW",
    "PCI": "PCT",
    "P-LoR": "P-LCR", "P-LOR": "P-LCR", "P-LG": "P-LCR",
    "RbC": "RBC", "RBC": "RBC",
    "AIt": "", "AIT": "", "Alt": "",
    "ID": "LYM%",
    "HPV": "MPV", "HVP": "MPV", "NPV": "MPV", "HPY": "MPV", "KPV": "MPV",
}
CORRECCIONES_ITEM = {k.upper(): v for k, v in CORRECCIONES_ITEM.items()}
CORRECCIONES_ITEM["RDU-WV"] = "RDW-CV"
CORRECCIONES_ITEM["RWU-WV"] = "RDW-CV"
CORRECCIONES_ITEM["RDU-SD"] = "RDW-SD"
CORRECCIONES_ITEM["P-LCR"] = "P-LCR"
CORRECCIONES_ITEM["P-LOR"] = "P-LCR"

CORRECCIONES_ITEM_PARCIAL = {
    "MCH": "MCH", "MCHC": "MCHC", "MCV": "MCV", "MPV": "MPV", "RBC": "RBC",
}

CORRECCIONES_UNIDAD = {
    "AL": "fL", "\u20acL": "fL", "(L": "fL", "{L": "fL", "[L": "fL", "ML": "fL",
    "s/dL": "g/dL", "g/d1": "g/dL", "y/dL": "g/dL", "Y/DL": "g/dL", "g/di": "g/dL", "g/aL": "g/dL", "9/dL": "g/dL",
    "PY": "pg", "py": "pg",
}

PATRONES_ITEM = [
    (re.compile(r"^LYN[%]?$"), "LYM%"),
    (re.compile(r"^MID[%]?$"), "MID%"),
    (re.compile(r"^GRA[vykY]?$"), "GRA%"),
    (re.compile(r"^[HL][YN]+$"), "LYM%"),
]


_PATRON_RANGO = re.compile(
    r'^(.+?)\s+(\d[\d.,]*\s*-\s*\d[\d.,]*)$'
)


def corregir_item(texto):
    if not texto:
        return texto
    texto = texto.strip().upper()
    if texto in CORRECCIONES_ITEM:
        return CORRECCIONES_ITEM[texto]
    for palabra, reemplazo in CORRECCIONES_ITEM.items():
        if len(palabra) >= 3 and re.search(r'\b' + re.escape(palabra) + r'\b', texto):
            return reemplazo
    for patron, reemplazo in PATRONES_ITEM:
        if patron.match(texto):
            return reemplazo
    return texto


def _normalizar_potencias(texto):
    texto = re.sub(r'10[\s\'\`\*\^\)\(]*12\s*[/\\]?\s*[Ll1]?', '10^12/L', texto)
    texto = re.sub(r'10[\s\'\`\*\^\)\(]*9\s*[/\\]?\s*[Ll1]?', '10^9/L', texto)
    texto = re.sub(r'(10\^(?:9|12)/L)\s*[1lL/\\]+', r'\1', texto)
    return texto


def corregir_resultado(texto):
    if not texto:
        return texto

    texto = texto.replace(",", ".")
    texto = re.sub(r'\.\.+', '.', texto)
    texto = _normalizar_potencias(texto)
    texto = re.sub(r'(\d)\s*["\*>]+\s*$', r'\1 %', texto)
    texto = re.sub(r'(\d)\s+6\s*$', r'\1 %', texto)
    for original, corregido in CORRECCIONES_UNIDAD.items():
        texto = texto.replace(original, corregido)

    return texto.strip()


def corregir_referencia(texto):
    if not texto:
        return texto

    texto = texto.replace(",", ".")
    texto = re.sub(r'\.\.+', '.', texto)
    texto = re.sub(r'^(\d{2})(\d{2})(\d)$', lambda m: m.group(1) + '-' + m.group(2) + m.group(3), texto)

    for original, corregido in CORRECCIONES_UNIDAD.items():
        texto = texto.replace(original, corregido)

    return texto.strip()


def separar_resultado_referencia(resultado, referencia):
   
    if referencia:
        return resultado, referencia
    m = _PATRON_RANGO.match(resultado.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return resultado, referencia
