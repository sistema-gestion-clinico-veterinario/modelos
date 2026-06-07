import sys
sys.path.append('.')
import scratch.debug_spike as ds

print("=== items_ocr ===")
for idx, item in enumerate(ds.items_ocr):
    print(f"{idx}: text='{item['texto']}' y={item['y']} x={item['x']} h={item['h']}")
