import re

CORRECCIONES_ITEM = {
    "UBC": "WBC", "VBC": "WBC", "HBC": "WBC",
    "LYMA": "LYM#", "LYM#": "LYM#", "LYN": "LYM%", "LYNY": "LYM%", "LYNY,": "LYM%",
    "HID#": "MID#", "HID": "MID%", "HIDA": "MID%", "MID": "MID%", "MiD#": "MID#", "MiD": "MID%", "NID#": "MID#",
    "GRAk": "GRA#", "GRAv": "GRA%", "GRAY": "GRA%", "GRAy": "GRA%", "GRAK": "GRA#",
    "HGB": "HGB", "HGb": "HGB",
    "HCHC": "MCHC", "HChC": "MCHC", "MCHO": "MCHC", "KCHC": "MCHC",
    "HCh": "MCH",
    "HCV": "MCV",
    "HCi": "HCT", "HCI": "HCT", "HCT": "HCT",
    "RDV-V": "RDW-CV", "RDV(V": "RDW-CV", "RDV-(V": "RDW-CV",
    "RDU-SD": "RDW-SD", "RDU-SD)": "RDW-SD", "RDH-SD": "RDW-SD",
    "PLI": "PLT", "PLl": "PLT",
    "PDU": "PDW", "PDH": "PDW", "PDw": "PDW", "PDu": "PDW",
    "PCI": "PCT",
    "P-LoR": "P-LCR", "P-LOR": "P-LCR",
    "RbC": "RBC", "HBC": "RBC",
    "AIt": "", "AIT": "", "Alt": "",
    "ID": "LYM%",
    "HPV": "MPV", "HVP": "MPV",
}
CORRECCIONES_ITEM = {k.upper(): v for k, v in CORRECCIONES_ITEM.items()}

CORRECCIONES_ITEM_PARCIAL = {
    "MCH": "MCH", "MCHC": "MCHC", "MCV": "MCV", "MPV": "MPV", "RBC": "RBC",
}

CORRECCIONES_UNIDAD = {
    "AL": "fL", "€L": "fL",
    "s/dL": "g/dL", "g/d1": "g/dL", "y/dL": "g/dL",
    "10*9/1": "10^9/L", "10*9/L": "10^9/L", "10^9/L": "10^9/L",
    "10*12/1": "10^12/L",
    "10'9/": "10^9/L",
    ",": ".",
}

PATRONES_ITEM = [
    (re.compile(r"^LYN[%]?$"), "LYM%"),
    (re.compile(r"^MID[%]?$"), "MID%"),
    (re.compile(r"^GRA[vykY]?$"), "GRA%"),
    (re.compile(r"^[HL][YN]+$"), "LYM%"),
]


def corregir_item(texto):
    if not texto:
        return texto
    texto = texto.strip().upper()
    if texto in CORRECCIONES_ITEM:
        return CORRECCIONES_ITEM[texto]
    for patron, reemplazo in PATRONES_ITEM:
        if patron.match(texto):
            return reemplazo
    return texto


def corregir_resultado(texto):
    if not texto:
        return texto
    for original, corregido in CORRECCIONES_UNIDAD.items():
        texto = texto.replace(original, corregido)
    return texto


def corregir_referencia(texto):
    if not texto:
        return texto
    for original, corregido in CORRECCIONES_UNIDAD.items():
        texto = texto.replace(original, corregido)
    return texto
