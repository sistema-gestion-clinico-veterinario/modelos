import re


def normalizar(valor: str) -> str:
    if not valor:
        return ""
    valor = str(valor).strip().upper()
    valor = re.sub(r'\s+', ' ', valor)
    valor = valor.replace(",", ".")
    try:
        return str(float(valor))
    except ValueError:
        return valor
