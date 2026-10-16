"""Fiber-clustered check of the fwd/back vs confidence recall difference on hard switches (reviewer request).

Hard switches from one fiber are not independent, so the unit is the fiber: a fiber-cluster bootstrap CI of the recall
difference and a cluster sign-flip permutation test (H0: per-fiber differences symmetric around zero). B = 10000, seed
7349. Reads results/v2/detection_events.csv; writes results/verification/cluster_test.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UNSEEN = ['PHerc0139', 'PHerc1451', 'PHerc0332']
B = 10000
rng = np.random.default_rng(7349)

ev = pd.read_csv(ROOT/'results/v2/detection_events.csv')
h = ev[(ev.cls == 'hard_switch') & ev.checked].copy()
h['fid'] = h.source + '/' + h.fiber_name
h['d'] = h.caught_fb.astype(int) - h.caught_conf.astype(int)
out = []
for label, e in [('unseen pooled', h[h.source.isin(UNSEEN)])] + [(s, h[h.source == s]) for s in ['paris4_heldout'] + UNSEEN]:
    g = e.groupby('fid').d.agg(['sum', 'size'])
    s, n = g['sum'].to_numpy(float), g['size'].to_numpy(float)
    obs = s.sum()/n.sum()
    idx = rng.integers(0, len(g), (B, len(g)))
    boot = s[idx].sum(1)/n[idx].sum(1)
    perm = (rng.choice([-1, 1], (B, len(g)))*s).sum(1)/n.sum()
    out.append(dict(source=label, hard_switches=int(n.sum()), fibers=len(g), fb_recall=e.caught_fb.mean(),
                    conf_recall=e.caught_conf.mean(), diff=obs, lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5),
                    p_cluster_signflip=(np.sum(np.abs(perm) >= abs(obs))+1)/(B+1)))
out = pd.DataFrame(out)
out.to_csv(ROOT/'results/verification/cluster_test.csv', index=False)
print(out.round(4).to_string(index=False))
