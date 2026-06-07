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

print(f"Col items: {len(col_items)}")
print(f"Col results: {len(col_results)}")
print(f"Col refs: {len(col_refs)}")

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

if slopes:
    estimated_slope = float(np.median(slopes))
    print(f"Estimated Slope: {estimated_slope} (Angle: {np.degrees(np.arctan(estimated_slope)):.2f} degrees)")
else:
    estimated_slope = 0.0
    print("No slopes found, default to 0.0")
