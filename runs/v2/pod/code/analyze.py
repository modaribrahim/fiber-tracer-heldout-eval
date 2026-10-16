"""Analyze held-out tracing runs: summaries, departure geometry, per-decision error signals, trim curves.

Inputs: run directories written by heldout_eval.py (rows.csv, paths.jsonl, decisions_<split>.npz).
Outputs (--out): summary.csv, departures.csv, decisions.csv.gz, signal_auc.csv, trim_curves.csv
Units: trace-grid voxels (8 base voxels = 19.2 um at 2.4 um scans); lengths also reported in mm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

from vesuvius.neural_tracing.tracers.shared.annotations.vc3d import load_fibers

UM_PER_TRACE_VOXEL = 19.2
SIGNALS = ('s_conf16', 's_conf_commit', 's_overlap', 's_dnormal', 's_dheading', 's_short_commit', 's_turn')


def angle(a, b):
    a, b = a/np.linalg.norm(a, axis=-1, keepdims=True), b/np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(a*b, -1)), 0, 1)))


def load_run(run):
    rows = pd.read_csv(run/'rows.csv')
    paths = [json.loads(line) for line in open(run/'paths.jsonl')]
    dec = {}
    for f in sorted(run.glob('decisions_*.npz')):
        d = np.load(f)
        dec.update({k: (np.concatenate([dec[k], d[k]]) if k in dec else d[k]) for k in d.files if k != 'diag_keys'})
    return rows, paths, dec


def decision_table(rows, paths, dec):
    """One row per decision with signals and labels relative to the trace's first departure."""
    out = []
    trace_ids = dec['trace']
    for t in np.unique(trace_ids):
        sel = np.where(trace_ids == t)[0]
        sel = sel[np.argsort(dec['decision'][sel])]
        p = np.asarray(paths[t]['path'], float)
        if len(p) < 2:
            continue
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        tree = cKDTree(p)
        dep_idx = paths[t]['departure_index']
        prev = None
        for k in sel:
            c = int(dec['n_commit'][k])
            world, conf, frame = dec['world'][k], dec['confidence'][k], dec['frame'][k]
            start = int(tree.query(dec['pos'][k])[1])
            end = int(tree.query(world[c-1])[1]) if c > 0 else start
            r = dict(trace=int(t), decision=int(dec['decision'][k]), n_commit=c, start_idx=start, end_idx=end,
                     start_len=float(seg[start]), conf16=float(conf[min(15, len(conf)-1)]),
                     conf_commit=float(conf[max(c-1, 0)]), conf_last=float(conf[-1]))
            r['s_conf16'], r['s_conf_commit'] = 1-r['conf16'], 1-r['conf_commit']
            r['s_short_commit'] = float(8 - c)
            if prev is not None and prev['c'] > 0:
                pred = prev['world'][prev['c']:prev['c']+24]
                cur = world[:8]
                r['s_overlap'] = float(cKDTree(pred).query(cur)[0].mean()) if len(pred) else np.nan
                r['s_dnormal'] = float(angle(frame[:, 0], prev['frame'][:, 0]))
                r['s_dheading'] = float(angle(frame[:, 2], prev['frame'][:, 2]))
                d_prev = prev['world'][min(prev['c'], len(prev['world']))-1]-prev['world'][0]
                d_cur = world[min(max(c, 2), len(world))-1]-world[0]
                r['s_turn'] = float(angle(d_cur, d_prev)) if np.linalg.norm(d_prev) > 0 and np.linalg.norm(d_cur) > 0 else np.nan
            else:
                r.update(s_overlap=np.nan, s_dnormal=np.nan, s_dheading=np.nan, s_turn=np.nan)
            if dep_idx is None:
                r['label'] = 'clean_trace'
            elif end < dep_idx - 16:
                r['label'] = 'before'
            elif start <= dep_idx:
                r['label'] = 'at_departure'
            else:
                r['label'] = 'after'
            out.append(r)
            prev = dict(world=world, c=c, frame=frame)
    return pd.DataFrame(out)


def departure_table(rows, paths, dec, fibers_by_name, source):
    """Geometry of each first departure: direction in the crop frame (u normal, v in-sheet, f heading)
    and whether the path lands on another annotated fiber."""
    names = list(fibers_by_name)
    all_pts = np.concatenate([fibers_by_name[n].points for n in names])
    owner = np.concatenate([[i]*len(fibers_by_name[n].points) for i, n in enumerate(names)])
    tree = cKDTree(all_pts)
    out = []
    for t, (row, pr) in enumerate(zip(rows.itertuples(), paths)):
        if pr['departure_index'] is None:
            continue
        p = np.asarray(pr['path'], float)
        i0 = pr['departure_index']
        i1 = min(len(p)-1, i0 + 12)
        f = fibers_by_name[row.fiber_name]
        gt_tree = cKDTree(f.points)
        d0, j0 = gt_tree.query(p[i1])
        disp = p[i1] - f.points[j0]
        sel = np.where(dec['trace'] == t)[0] if len(dec) else []
        if len(sel):
            k = sel[np.argmin(np.abs(np.array([np.linalg.norm(dec['pos'][s]-p[i0]) for s in sel])))]
            frame = dec['frame'][k]
            comp = disp @ frame  # components along (u, v, f)
            conf16 = float(dec['confidence'][k][min(15, dec['confidence'].shape[1]-1)])
        else:
            comp, conf16 = np.full(3, np.nan), np.nan
        other_d, other_j = tree.query(p[i1], k=8)
        other = [(d, names[owner[j]]) for d, j in zip(np.atleast_1d(other_d), np.atleast_1d(other_j))
                 if names[owner[j]] != row.fiber_name]
        out.append(dict(source=source, trace=t, fiber_name=row.fiber_name, family=row.family, annotator=row.annotator,
                        departure_len=float(row.departure_arclength), departure_mm=float(row.departure_arclength)*UM_PER_TRACE_VOXEL/1000,
                        offtrack=float(row.offtrack), dist_gt_after12=float(d0), comp_u_normal=float(comp[0]),
                        comp_v_insheet=float(comp[1]), comp_f_along=float(comp[2]),
                        direction=('normal' if abs(comp[0]) > abs(comp[1]) else 'in_sheet') if np.isfinite(comp).all() else None,
                        nearest_other_fiber=other[0][1] if other else None,
                        nearest_other_dist=float(other[0][0]) if other else np.nan, conf16_at_departure=conf16,
                        dep_x=row.dep_x, dep_y=row.dep_y, dep_z=row.dep_z))
    return pd.DataFrame(out)


def trim_curves(rows, dtab, signal, thresholds):
    """Cut every trace at its first decision whose signal exceeds the threshold."""
    res = []
    by_trace = {t: g.sort_values('decision') for t, g in dtab.groupby('trace')}
    correct = rows['correct'].to_numpy(float)
    wrong = rows['offtrack'].to_numpy(float)
    for thr in thresholds:
        kept_c = kept_w = 0.0
        for t in range(len(rows)):
            g = by_trace.get(t)
            cut = np.inf
            if g is not None:
                hit = g[g[signal] > thr]
                if len(hit):
                    cut = float(hit['start_len'].iloc[0])
            kept_c += min(cut, correct[t])
            kept_w += max(0.0, min(cut, correct[t] + wrong[t]) - correct[t])
        res.append(dict(signal=signal, threshold=float(thr), correct_kept=kept_c, wrong_kept=kept_w,
                        correct_frac=kept_c/max(correct.sum(), 1e-9), wrong_frac=kept_w/max(wrong.sum(), 1e-9)))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', nargs='+', help='run directories (…/c050/PHerc1451 …)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries, deps, decs, aucs, trims = [], [], [], [], []
    for run in map(Path, args.runs):
        cfg = json.loads((run/'config.json').read_text())
        source, conf = cfg['name'], cfg['operating_policy']['confidence']
        rows, paths, dec = load_run(run)
        fibers = {f.name: f for f in load_fibers(cfg['fibers_dir'], grid_scale=cfg['grid_scale'])}
        for split, g in list(rows.groupby('split')) + [('all', rows)]:
            traced = g['length'].sum()
            summaries.append(dict(
                source=source, confidence=conf, split=split, traces=len(g), departures=int(g['diverged'].sum()),
                departure_rate=float(g['diverged'].mean()), traced_mm=traced*UM_PER_TRACE_VOXEL/1000,
                correct_mm=g['correct'].sum()*UM_PER_TRACE_VOXEL/1000, wrong_mm=g['offtrack'].sum()*UM_PER_TRACE_VOXEL/1000,
                length_precision=g['correct'].sum()/max(g['correct'].sum()+g['offtrack'].sum(), 1e-9),
                coverage=g['followed'].sum()/max(g['avail'].sum(), 1e-9),
                departures_per_cm=g['diverged'].sum()/max(traced*UM_PER_TRACE_VOXEL/1e4, 1e-9),
                median_followed_mm=float(g['followed'].median())*UM_PER_TRACE_VOXEL/1000,
                stop_confidence=int((g['reason'] == 'confidence').sum()), stop_maxlen=int((g['reason'] == 'max_len').sum())))
        dt = departure_table(rows, paths, dec, fibers, source)
        dt['confidence'] = conf
        deps.append(dt)
        if dec:
            d = decision_table(rows, paths, dec)
            d['source'], d['confidence'] = source, conf
            decs.append(d)
            lab = d[d['label'].isin(['before', 'clean_trace', 'at_departure'])]
            y = (lab['label'] == 'at_departure').to_numpy()
            for s in SIGNALS:
                v = lab[s].to_numpy(float)
                ok = np.isfinite(v)
                auc = roc_auc_score(y[ok], v[ok]) if ok.sum() and 0 < y[ok].sum() < ok.sum() else np.nan
                aucs.append(dict(source=source, confidence=conf, signal=s, auc=auc, positives=int(y[ok].sum()), n=int(ok.sum())))
                vals = d[s].to_numpy(float)
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    for tr in trim_curves(rows, d, s, np.unique(np.quantile(vals, np.linspace(0.5, 0.999, 40)))):
                        trims.append(dict(source=source, confidence=conf, **tr))
    pd.DataFrame(summaries).to_csv(out/'summary.csv', index=False)
    pd.concat(deps).to_csv(out/'departures.csv', index=False)
    if decs:
        pd.concat(decs).to_csv(out/'decisions.csv.gz', index=False)
    pd.DataFrame(aucs).to_csv(out/'signal_auc.csv', index=False)
    pd.DataFrame(trims).to_csv(out/'trim_curves.csv', index=False)
    s = pd.DataFrame(summaries)
    print(s[s['split'] == 'all'].round(3).to_string(index=False))
    if aucs:
        print(pd.DataFrame(aucs).pivot_table(index='signal', columns=['source', 'confidence'], values='auc').round(3).to_string())


if __name__ == '__main__':
    main()
