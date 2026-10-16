"""Rescore saved forward/backward checks without rerunning the GPU.

Segmented rule variants: a check FAILS only when its back trace covered at least --min-back voxels and less than
--on-frac of it stayed within tolerance of the forward path; a back trace that stopped early is 'unknown' (pass).
Each trace is cut at its first failing check (keeping up to the previous check). Full round trip: keep traces whose
back trace returned to the seed. Prints kept correct / wrong length (mm) per run next to the uncut baseline.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

UM = 0.0192
out = []
for run in map(Path, sys.argv[1:]):
    if not (run/'fwdback_segments.csv').exists():
        continue
    cfg = json.loads((run/'config.json').read_text())
    rows = pd.read_csv(run/'rows.csv')
    seg = pd.read_csv(run/'fwdback_segments.csv')
    full = pd.read_csv(run/'fwdback_full.csv')
    c, w = rows['correct'].to_numpy(float), rows['offtrack'].to_numpy(float)
    base = dict(source=cfg['name'], rule='none', correct_mm=c.sum()*UM, wrong_mm=w.sum()*UM)
    out.append(base)
    ret = np.zeros(len(rows), bool)
    ret[full['trace'].to_numpy()] = full['returned'].to_numpy(bool)
    out.append(dict(source=cfg['name'], rule='full round trip returned', correct_mm=c[ret].sum()*UM, wrong_mm=w[ret].sum()*UM))
    for min_back in (32, 64, 128):
        for on_frac in (0.5, 0.7, 0.9):
            compared = seg['compared_len'] if 'compared_len' in seg else seg['back_len']
            fail = (compared >= min_back) & (seg['frac_on'] < on_frac)
            cut = np.full(len(rows), np.inf)
            for t, g in seg[fail].groupby('trace'):
                cut[t] = max(0.0, float(g['at_len'].min()) - 128.0)
            kc = np.minimum(cut, c).sum()
            kw = np.maximum(0, np.minimum(cut, c + w) - c).sum()
            out.append(dict(source=cfg['name'], rule=f'segmented min_back={min_back} on_frac<{on_frac}',
                            correct_mm=kc*UM, wrong_mm=kw*UM, checks_failed=int(fail.sum()), checks=len(seg)))
df = pd.DataFrame(out)
df['precision'] = df['correct_mm']/(df['correct_mm']+df['wrong_mm'])
pd.set_option('display.width', 200)
print(df.round(3).to_string(index=False))
df.to_csv(Path(sys.argv[1]).parent.parent/'fwdback_rescored.csv' if len(sys.argv) > 1 else 'fwdback_rescored.csv', index=False)
