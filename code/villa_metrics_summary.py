"""Summary of villa_metrics.py outputs. Usage: python villa_metrics_summary.py results/v3/apples"""
import numpy as np, pandas as pd
import sys
D = sys.argv[1].rstrip('/') + '/'  # results/v3/apples
SCROLLS = ['paris4_heldout', 'PHerc0139', 'PHerc1451', 'PHerc0332']
UNSEEN = SCROLLS[1:]
df = {s: pd.read_csv(D+'known_'+s+'.csv') for s in SCROLLS}
def ft(d):
    g = d.groupby('fiber_name')
    return pd.DataFrame(dict(w15=g.within_known_1p5.sum(), w3=g.within_known_3.sum(), kl=g.known_len.sum(), L=g.length.sum(), n=g.size())).to_numpy(float)
FT = {s: ft(d) for s, d in df.items()}
def agg(parts):
    a = np.sum([p.sum(0) for p in parts], 0)
    return dict(w15=a[0]/a[2], w3=a[1]/a[2], ks=a[2]/a[3], n=a[4])
B = 2000; rng = np.random.default_rng(7349)
draws = [{s: FT[s][rng.integers(0, len(FT[s]), len(FT[s]))] for s in SCROLLS} for _ in range(B)]
def boot(groups): return np.array([agg([dr[s] for s in groups])['w15'] for dr in draws])
ci = lambda x: np.percentile(x, [2.5, 97.5])
rows = []
for name, groups in [(s, [s]) for s in SCROLLS] + [('unseen pooled', UNSEEN)]:
    a = agg([FT[s] for s in groups]); lo, hi = ci(boot(groups))
    rows.append(dict(scroll=name, within_1p5_share_known=a['w15'], within_3_share_known=a['w3'], known_share_of_path=a['ks'],
                     n_traces=int(a['n']), n_fibers=sum(len(FT[s]) for s in groups), within_1p5_share_known_lo=lo, within_1p5_share_known_hi=hi))
# unseen pooled: p_correct+p_offtrack sanity
u, p = boot(UNSEEN), boot(['paris4_heldout']); lo, hi = ci(u/p)
rows.append(dict(scroll='unseen/Paris4', within_1p5_share_known=agg([FT[s] for s in UNSEEN])['w15']/agg([FT['paris4_heldout']])['w15'],
                 within_1p5_share_known_lo=lo, within_1p5_share_known_hi=hi))
out = pd.DataFrame(rows); out.to_csv(D+'summary_known.csv', index=False)
pd.set_option('display.width', 250, 'display.max_columns', 100, 'display.float_format', lambda x: f'{x:.4f}')
print(out.to_string(index=False))
