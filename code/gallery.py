"""Render departures for visual audit: CT in the decision's crop frame with GT (green) and trace (red).

Per departure, two panels at CT level 2 (9.6 um/px):
  left  in-sheet view (f along fiber x v across fiber), thin max projection over the sheet normal u in [-2, 2]
  right cross-section (f x u), the sheet layers; a trace that changes layer leaves the bright band
Points within a +-6 px slab of the plane are drawn; the departure point is a yellow cross.
Writes one PNG per departure and contact sheets of 6 into --out, plus index.csv.
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
from scipy.ndimage import map_coordinates

from vesuvius.neural_tracing.tracers.shared.annotations.vc3d import load_fibers
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import RemoteChunkedArray

SCALE = 2.0  # trace-grid voxel -> CT level-2 voxel


def plane(arr, center, a, b, half_a, half_b, normal=None, thick=0):
    """Sample a plane through center (CT xyz) spanned by unit vectors a, b; optional MIP along normal."""
    ia = np.arange(-half_a, half_a + 1)
    ib = np.arange(-half_b, half_b + 1)
    A, B = np.meshgrid(ia, ib, indexing='xy')
    pts = center + A[..., None]*a + B[..., None]*b
    offsets = range(-thick, thick + 1) if normal is not None else [0]
    lo = np.floor(pts.reshape(-1, 3).min(0) - thick - 2).astype(int)
    hi = np.ceil(pts.reshape(-1, 3).max(0) + thick + 2).astype(int)
    block = arr.read(lo[::-1], (hi - lo)[::-1]).astype(np.float32)  # zyx
    out = None
    for o in offsets:
        q = pts + (o*normal if normal is not None else 0) - lo
        v = map_coordinates(block, [q[..., 2], q[..., 1], q[..., 0]], order=1, mode='nearest')
        out = v if out is None else np.maximum(out, v)
    return out


def project(points, center, a, b, n, slab):
    rel = (points - center)
    pa, pb, pn = rel @ a, rel @ b, rel @ n
    keep = np.abs(pn) <= slab
    return pa[keep], pb[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-max-after', type=float, default=0.0, help='only departures whose max GT distance exceeds this')
    ap.add_argument('--cache-dir', default='/workspace/cache')
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run/'config.json').read_text())
    rows = pd.read_csv(run/'rows.csv')
    paths = [json.loads(line) for line in open(run/'paths.jsonl')]
    dec = {}
    for f in sorted(run.glob('decisions_*.npz')):
        d = np.load(f)
        dec.update({k: (np.concatenate([dec[k], d[k]]) if k in dec else d[k]) for k in ('trace', 'pos', 'frame')})
    fibers = {f.name: f for f in load_fibers(cfg['fibers_dir'], grid_scale=cfg['grid_scale'])}
    others = np.concatenate([f.points for f in fibers.values()]) * SCALE
    arr = RemoteChunkedArray(cfg['ct'], cfg['ct_level'], args.cache_dir, 2 << 30)

    index, files = [], []
    for t, (row, pr) in enumerate(zip(rows.itertuples(), paths)):
        i0 = pr['departure_index']
        if i0 is None:
            continue
        gd = np.asarray(pr['gt_distance'])
        if gd[i0:].max() <= args.min_max_after:
            continue
        p = np.asarray(pr['path'], float) * SCALE
        sel = np.where(dec['trace'] == t)[0]
        k = sel[np.argmin(np.linalg.norm(dec['pos'][sel]*SCALE - p[i0], axis=1))]
        u, v, f = (dec['frame'][k][:, j] for j in range(3))
        c = p[i0]
        gt = fibers[row.fiber_name].points * SCALE
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=90)
        for ax, (b, n, title, thick) in zip(axes, ((v, u, 'in-sheet (f x v), MIP |u|<=2', 2), (u, v, 'cross-section (f x u)', 0))):
            img = plane(arr, c, f, b, 120, 50, normal=n, thick=thick)
            ax.imshow(img, cmap='gray', extent=(-120, 120, 50, -50), vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
            oa, ob = project(others, c, f, b, n, 6)
            ax.scatter(oa, ob, s=1, c='deepskyblue', alpha=0.5, lw=0)
            ga, gb = project(gt, c, f, b, n, 6)
            ax.scatter(ga, gb, s=2, c='lime', lw=0)
            ta, tb = project(p, c, f, b, n, 6)
            ax.scatter(ta, tb, s=2, c='red', lw=0)
            ax.plot([0], [0], 'y+', ms=12, mew=2)
            ax.set_xlim(-120, 120); ax.set_ylim(50, -50)
            ax.set_title(title, fontsize=8); ax.tick_params(labelsize=6)
        name = f"{cfg['name']}_t{t:03d}"
        fig.suptitle(f"{name} {row.fiber_name} {row.family} ann={row.annotator} split={row.split} "
                     f"maxGTdist={gd[i0:].max():.1f}vx offtrack={row.offtrack*0.0192:.1f}mm", fontsize=8)
        fig.tight_layout()
        fp = out/f'{name}.png'
        fig.savefig(fp); plt.close(fig)
        files.append(fp)
        index.append(dict(image=fp.name, trace=t, fiber=row.fiber_name, family=row.family, annotator=row.annotator,
                          max_after=float(gd[i0:].max()), offtrack_mm=row.offtrack*0.0192,
                          dep_x=row.dep_x, dep_y=row.dep_y, dep_z=row.dep_z))
    pd.DataFrame(index).to_csv(out/f"index_{cfg['name']}.csv", index=False)
    for s in range(0, len(files), 6):
        imgs = [plt.imread(f) for f in files[s:s+6]]
        fig, axes = plt.subplots(len(imgs), 1, figsize=(9, 3.4*len(imgs)), dpi=90)
        for ax, im in zip(np.atleast_1d(axes), imgs):
            ax.imshow(im); ax.axis('off')
        fig.tight_layout(pad=0.2)
        fig.savefig(out/f"sheet_{cfg['name']}_{s//6:02d}.png"); plt.close(fig)
    print(f"{cfg['name']}: {len(files)} departures rendered", flush=True)


if __name__ == '__main__':
    main()
