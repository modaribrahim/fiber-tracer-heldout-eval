"""Independent re-implementation of the v2 headline numbers from raw run files (shares no code with stats.py/detect.py)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
SRC = ['paris4_heldout', 'PHerc0139', 'PHerc1451', 'PHerc0332']
UNSEEN = SRC[1:]
VX_CM = 0.0192 / 10  # one trace-grid voxel (8 x 2.4 um) in cm
frozen = json.loads((ROOT/'results/final/detector_frozen.json').read_text())
T_FB, T_CONF = frozen['thresholds']['fb'], frozen['thresholds']['conf']
BACK = 192.0


def classify(L, d):
    """My own excursion classifier: list of (onset_len, cls) in the scored region."""
    bad = d > 3.0
    out, i = [], 0
    n = len(d)
    while i < n:
        # next sustained run of bad points spanning >= 3 vx
        onset = None
        while i < n:
            if bad[i]:
                j = i
                while j < n and bad[j]:
                    if L[j] - L[i] >= 3.0:
                        onset = i
                        break
                    j += 1
                if onset is not None:
                    break
                i = j
            else:
                i += 1
        if onset is None:
            break
        # return = first good run spanning >= 16 vx after onset
        ret, k = None, onset
        while k < n:
            if not bad[k]:
                m = k
                while m < n and not bad[m]:
                    m += 1
                if L[m-1] - L[k] >= 16.0:
                    ret = k
                    break
                k = m
            else:
                k += 1
        stop = ret if ret is not None else n
        peak = d[onset:stop].max()
        if ret is not None:
            cls = 'returned'
        elif L[-1] - L[onset] < 16.0:
            cls = 'ended'
        else:
            cls = 'hard_switch' if peak > 15.0 else 'drift'
        out.append((L[onset], cls))
        if ret is None:
            break
        i = ret
    return out


dec = pd.read_csv(ROOT/'results/v2/decisions.csv.gz', usecols=['trace', 'start_len', 'conf16', 'source'])
res, ev_all = {}, []
for s in SRC:
    run = ROOT/'runs/v2'/s
    rows = pd.read_csv(run/'rows.csv')
    seg = pd.read_csv(run/'fwdback_segments.csv')
    dd = dec[dec.source == s]
    res[s] = dict(traces=len(rows), fibers=rows.fiber_name.nunique(), dep=int(rows.diverged.sum()),
                  correct_cm=rows.correct.sum()*VX_CM, prec=rows.correct.sum()/(rows.correct.sum()+rows.offtrack.sum()))
    for t, line in enumerate(open(run/'paths.jsonl')):
        p = json.loads(line)
        P = np.asarray(p['path'], float).reshape(-1, 3)
        d = np.asarray(p['gt_distance'], float)
        if len(P) < 2:
            continue
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        r = rows.iloc[t]
        assert r.fiber_name == p['fiber_name']
        keep = L <= r.correct + 1e-9 if r.crossed_endpoint else np.ones(len(L), bool)
        if keep.sum() < 2:
            continue
        cks = seg[seg.trace == t]
        dt = dd[dd.trace == t]
        for on, cls in classify(L[keep], d[keep]):
            c = cks[(cks.at_len >= on) & (cks.at_len <= on + BACK)]
            fb = any((1 - x.frac_on > T_FB) if x.compared_len >= 16 else False for x in c.itertuples())
            cf = False
            for x in c.itertuples():
                w = dt[(dt.start_len > x.at_len - BACK) & (dt.start_len <= x.at_len)].conf16
                cf |= bool(len(w) and (1 - w.min()) > T_CONF)
            ev_all.append(dict(source=s, trace=t, fiber=r.fiber_name, onset=on, cls=cls, checked=len(c) > 0, fb=fb, conf=cf))
ev = pd.DataFrame(ev_all)
ev.to_csv(ROOT/'results/verification/independent_events.csv', index=False)

print('source traces fibers strict_prec dep/cm')
for s in SRC:
    r = res[s]
    print(s, r['traces'], r['fibers'], round(r['prec'], 3), round(r['dep']/r['correct_cm'], 3))
u = sum(res[s]['dep'] for s in UNSEEN)/sum(res[s]['correct_cm'] for s in UNSEEN)
print('unseen/Paris4 departures per cm ratio:', round(u/(res[SRC[0]]['dep']/res[SRC[0]]['correct_cm']), 3))
print('\nevent classes:', ev.groupby(['source', 'cls']).size().unstack(fill_value=0).to_string())
print('\nhard-switch recall (checked):')
for name, m in [(s, ev.source == s) for s in SRC] + [('unseen pooled', ev.source.isin(UNSEEN))]:
    h = ev[m & (ev.cls == 'hard_switch') & ev.checked]
    b, c = int((h.fb & ~h.conf).sum()), int((~h.fb & h.conf).sum())
    print(f'{name:14s} n={len(h):4d} fb={h.fb.mean():.3f} conf={h.conf.mean():.3f} fb_only={b} conf_only={c} '
          f'McNemar p={binomtest(b, b+c, .5).pvalue if b+c else 1:.2g}')
print('unchecked departures (all classes):', int((~ev.checked).sum()), 'of', len(ev))
