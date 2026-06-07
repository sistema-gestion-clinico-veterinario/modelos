import json
import pandas as pd
import pathlib

gt = json.load(open('ground_truth/ground_truth.json'))

for fname, data in gt.items():
    stem = pathlib.Path(fname).stem
    csv_path = pathlib.Path(f'results/{stem}.csv')
    if not csv_path.exists():
        print(f'\nNo CSV for: {fname}')
        continue
    df = pd.read_csv(csv_path)
    rows_gt = {r['item']: r for r in data['rows']}
    misses = []
    for _, row in df.iterrows():
        gt_row = rows_gt.get(row['item'])
        if gt_row:
            got_res = str(row['resultado']).strip() if not pd.isna(row['resultado']) else ''
            exp_res = str(gt_row['resultado']).strip()
            got_flag = str(row['flag']).strip() if not pd.isna(row['flag']) else ''
            exp_flag = str(gt_row.get('flag', '')).strip()
            if got_res != exp_res:
                misses.append(f"  [resultado] {row['item']}: got='{got_res}' expect='{exp_res}'")
            if got_flag != exp_flag:
                misses.append(f"  [flag]      {row['item']}: got='{got_flag}' expect='{exp_flag}'")
    if misses:
        print(f'\n=== {fname} ===')
        for m in misses:
            print(m)
    else:
        print(f'\n=== {fname} -> PERFECT ===')
