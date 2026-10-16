"""CT image features at every excursion onset of a run, plus matched on-fiber control points (exploratory analysis).

Per point (the path vertex where the excursion starts, or a control vertex on the fiber), in the decision's crop
frame (u = sheet normal, v = across the fiber in the sheet, f = along the fiber), at CT level 2 (9.6 um/px):
  layer_gap_vx   distance along +-u from the point to the nearest neighbouring bright layer (|s| > 4 px)
  fiber_gap_vx   distance along +-v, within the sheet (max projection |u| <= 2), to the nearest neighbouring ridge (|s| > 2 px)
  contrast       p95 - p5 of the 48^3 px block around the point; center_rel = intensity at the point, block-normalized
  gt_neighbor_vx distance to the nearest point of any other annotated fiber (trace-grid voxels)
Peaks: local maxima of the Gaussian-smoothed (sigma 1 px) profile with prominence >= 0.25 x the block contrast.
Controls: 2 per trace at random vertices within 1.5 vx of the GT fiber and >= 64 vx from every onset (RNG 7349).
Also renders gallery panels (gallery.py style) for hard switches and drifts. Writes into --out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from vesuvius.neural_tracing.tracers.shared.annotations.vc3d import load_fibers
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import RemoteChunkedArray

import stats as S
from gallery import SCALE, plane, project

HALF = 24  # CT px


def profile(block, lo, c, axis, others=(), mip=0):
    """Intensity along c + s*axis, s in [-32, 32] px step 0.5, averaged over a 3x3 px patch in the two other axes."""
    s = np.arange(-32, 32.01, 0.5)
    vals = []
    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            pts = c + s[:, None]*axis + da*others[0] + db*others[1]
            if mip:
                v = np.max([map_coordinates(block, (pts + m*others[0] - lo)[:, ::-1].T, order=1, mode='nearest')
                            for m in range(-mip, mip + 1)], axis=0)
            else:
                v = map_coordinates(block, (pts - lo)[:, ::-1].T, order=1, mode='nearest')
            vals.append(v)
    return s, gaussian_filter1d(np.mean(vals, axis=0), 2.0)  # sigma 1 px = 2 samples


def nearest_peak(s, p, contrast, min_abs):
    peaks, _ = find_peaks(p, prominence=0.25*contrast)
    d = np.abs(s[peaks])
    d = d[d > min_abs]
    return float(d.min()) if len(d) else np.nan


def features(arr, c, u, v, f):
    lo = np.floor(c - HALF - 40).astype(int)
    size = np.full(3, 2*HALF + 80)
    block = arr.read(lo[::-1], size[::-1]).astype(np.float32)  # zyx
    inner = block[40:-40, 40:-40, 40:-40]
    p5, p95 = np.percentile(inner, [5, 95])
    contrast = float(p95 - p5)
    su, pu = profile(block, lo, c, u, (v, f))
    sv, pv = profile(block, lo, c, v, (u, f), mip=2)
    center = float(map_coordinates(block, [[c[2]-lo[2]], [c[1]-lo[1]], [c[0]-lo[0]]], order=1)[0])
    return dict(layer_gap_vx=nearest_peak(su, pu, contrast, 4.0)/SCALE, fiber_gap_vx=nearest_peak(sv, pv, contrast, 2.0)/SCALE,
                contrast=contrast, center_rel=(center - p5)/max(contrast, 1e-6), mean=float(inner.mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cache-dir', default='/workspace/cache')
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    (out/'gallery').mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run/'config.json').read_text())
    rows = pd.read_csv(run/'rows.csv')
    paths = [json.loads(line) for line in open(run/'paths.jsonl')]
    dec = {}
    for fn in sorted(run.glob('decisions_*.npz')):
        d = np.load(fn)
        dec.update({k: (np.concatenate([dec[k], d[k]]) if k in dec else d[k]) for k in ('trace', 'pos', 'frame')})
    by_trace = {t: np.flatnonzero(dec['trace'] == t) for t in np.unique(dec['trace'])}
    fibers = {fb.name: fb for fb in load_fibers(cfg['fibers_dir'], grid_scale=cfg['grid_scale'])}
    names = list(fibers)
    allpts = np.concatenate([fibers[n].points for n in names])
    owner = np.concatenate([np.full(len(fibers[n].points), i) for i, n in enumerate(names)])
    tree = cKDTree(allpts)
    arr = RemoteChunkedArray(cfg['ct'], cfg['ct_level'], args.cache_dir, 4 << 30)
    rng = np.random.default_rng(7349)

    recs = []
    for t, (row, pr) in enumerate(zip(rows.itertuples(), paths)):
        P = np.asarray(pr['path'], float).reshape(-1, 3)
        d = np.asarray(pr['gt_distance'], float)
        if len(P) < 2 or t not in by_trace:
            continue
        L = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        scored = L <= row.correct + 1e-9 if row.crossed_endpoint else np.ones(len(L), bool)
        eps = S.episodes(L[scored], d[scored]) if scored.sum() > 1 else []
        onsets = [e['onset_idx'] for e in eps]
        on = np.flatnonzero(scored & (d <= 1.5))
        far = [i for i in on if all(abs(L[i] - L[o]) >= 64 for o in onsets)]
        ctrl = rng.choice(far, size=min(2, len(far)), replace=False) if far else []
        pts = [(e['onset_idx'], e['cls'], e['onset_len'], e['peak']) for e in eps] + [(int(i), 'control', float(L[i]), np.nan) for i in ctrl]
        own = names.index(row.fiber_name)
        sel = by_trace[t]
        for i, kind, olen, peak in pts:
            k = sel[np.argmin(np.linalg.norm(dec['pos'][sel] - P[i], axis=1))]
            u, v, f = (dec['frame'][k][:, j] for j in range(3))
            c = P[i]*SCALE
            dd, ii = tree.query(P[i], k=64)
            other = dd[owner[ii] != own]
            rec = dict(source=cfg['name'], trace=t, fiber_name=row.fiber_name, kind=kind, idx=int(i), onset_len=olen,
                       peak=peak, gt_neighbor_vx=float(other[0]) if len(other) else np.nan,
                       decision_offset_vx=float(np.linalg.norm(dec['pos'][k] - P[i])), **features(arr, c, u, v, f))
            recs.append(rec)
            if kind in ('hard_switch', 'drift'):
                gt = fibers[row.fiber_name].points*SCALE
                oth = allpts[owner != own]*SCALE
                p2 = P*SCALE
                fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=80)
                for ax, (b, n, title, thick) in zip(axes, ((v, u, 'in-sheet (f x v), MIP |u|<=2', 2), (u, v, 'cross-section (f x u)', 0))):
                    img = plane(arr, c, f, b, 120, 50, normal=n, thick=thick)
                    ax.imshow(img, cmap='gray', extent=(-120, 120, 50, -50), vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
                    for q, col, sz in ((oth, 'deepskyblue', 1), (gt, 'lime', 2), (p2, 'red', 2)):
                        qa, qb = project(q, c, f, b, n, 6)
                        ax.scatter(qa, qb, s=sz, c=col, lw=0, alpha=0.5 if col == 'deepskyblue' else 1)
                    ax.plot([0], [0], 'y+', ms=12, mew=2)
                    ax.set_xlim(-120, 120); ax.set_ylim(50, -50)
                    ax.set_title(title, fontsize=8); ax.tick_params(labelsize=6)
                fig.suptitle(f"{cfg['name']} t{t:04d} {kind} at {olen:.0f}vx peak={peak:.1f}vx {row.fiber_name} "
                             f"layer_gap={rec['layer_gap_vx']:.1f} fiber_gap={rec['fiber_gap_vx']:.1f} gtnb={rec['gt_neighbor_vx']:.1f}", fontsize=7)
                fig.tight_layout()
                fig.savefig(out/'gallery'/f"{cfg['name']}_t{t:04d}_{int(olen):05d}_{kind}.png"); plt.close(fig)
    df = pd.DataFrame(recs)
    df.to_csv(out/f"image_features_{cfg['name']}.csv", index=False)
    print(cfg['name'], df.groupby('kind')[['layer_gap_vx', 'fiber_gap_vx', 'contrast', 'gt_neighbor_vx']].median().round(2).to_string(),
          df.kind.value_counts().to_dict(), flush=True)


if __name__ == '__main__':
    main()
