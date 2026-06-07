import numpy as np
import sys
sys.path.append('.')
import scratch.debug_spike as ds

items_ocr = ds.items_ocr


altos = [i["h"] for i in items_ocr]
median_h = np.median(altos)
items = [i for i in items_ocr if i["h"] <= median_h * 2.5]


col_items = [i for i in items if i["x"] < 300]
col_results = [i for i in items if 300 <= i["x"] < 480]
col_refs = [i for i in items if 480 <= i["x"] < 700]

slopes = []
for it in col_items:
    for res in col_results:
        dy = res["y"] - it["y"]
        dx = res["x"] - it["x"]
        if dx > 50 and abs(dy) < 40:
            slopes.append(dy / dx)

for res in col_results:
    for ref in col_refs:
        dy = ref["y"] - res["y"]
        dx = ref["x"] - res["x"]
        if dx > 50 and abs(dy) < 40:
            slopes.append(dy / dx)

estimated_slope = float(np.median(slopes)) if slopes else 0.0
print(f"Using Slope: {estimated_slope}")


for item in items:
    item["y_proj"] = item["y"] - estimated_slope * item["x"]


items.sort(key=lambda i: i["y_proj"])


alto_promedio = float(np.median([i["h"] for i in items]))
tolerancia = alto_promedio * 0.4

filas = []
fila_actual = [items[0]]
for item in items[1:]:
    if abs(item["y_proj"] - fila_actual[-1]["y_proj"]) <= tolerancia:
        fila_actual.append(item)
    else:
        filas.append(sorted(fila_actual, key=lambda i: i["x"]))
        fila_actual = [item]
filas.append(sorted(fila_actual, key=lambda i: i["x"]))

print(f"Total rows grouped with projected Y: {len(filas)}")
for idx, f in enumerate(filas):
    textos = [p["texto"] for p in f]
    print(f"Row {idx}: {textos}")
