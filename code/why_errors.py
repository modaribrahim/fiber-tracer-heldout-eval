"""v3 stages 1 and 2a (ANALYSIS_PLAN_v3.md): human-fiber loading, the per-vertex trace table, the data checks, and the
covariate / exposure / episode tables.

Usage: python -I why_errors.py DATA FIBERS VILLA OUT [--checks] [--tables] [--flags] [--crossdiag] [--corrected] [--s6]
  DATA    root with runs/v2/<src>/{rows.csv,paths.jsonl,config.json,manifest.json}
  FIBERS  directory with one human-fiber directory per scroll (last part of config['fibers_dir'])
  VILLA   villa vesuvius/src (only its fiber loader and scorer are imported)
  OUT     output directory
  --checks  stage 1: data checks -> OUT/data_checks.json
  --tables  stage 2a: OUT/tables/{covariates_<src>.npz, exposure.npz, episodes.csv, traces.csv, sanity.json}
  --flags   stage 2c (post-hoc, reads OUT/tables): OUT/results/{episode_flags.csv, strict_exposure.npz, flags_2c.json}
  --crossdiag  stage 2c diagnostic (not used by any table): OUT/results/episode_crossing_diag.csv, other crossing rules per terminal episode
  --corrected  stage 2d (post-hoc): episodes recomputed on the region cut at the annotation end -> OUT/tables_2d/{exposure.npz, episodes.csv,
               traces.csv, strict_exposure.npz} (+ links to the stage-2a covariates and sanity.json), OUT/results/{episodes_corrected.csv,
               episode_transitions_2d.csv, corrected_2d.json}
  --s6         stage 2e (S6 on the corrected hard switches; needs --corrected): OUT/results/{s6_switches.csv, past_end_example_data.npz}
Nothing here relates event rates to geometry: that is a later stage, which imports these functions.

Frames: trace grid, 1 vx = 19.2 um (full-resolution JSON xyz / grid_scale). Trace i of a scroll = rows.csv row i =
paths.jsonl line i = stats.load trace i.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import resource
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not put the script directory on the path
import stats as S

V = {}  # villa 'scoring' module and 'load_fibers', set by init_villa
EXPECT = {'hard_switch': dict(paris4_heldout=54, PHerc0139=281, PHerc1451=52, PHerc0332=85),
          'returned': dict(paris4_heldout=130, PHerc0139=811, PHerc1451=109, PHerc0332=153),
          'drift': dict(paris4_heldout=3, PHerc0139=136, PHerc1451=30, PHerc0332=18),
          'ended': dict(paris4_heldout=2, PHerc0139=48, PHerc1451=10, PHerc0332=8)}


def init_villa(src):
    """Import villa's scorer and fiber loader without its heavy package (CT-reading deps are stubbed)."""
    src = Path(src)
    sys.path.insert(0, str(src))
    for name, sub in [('vesuvius', 'vesuvius'), ('vesuvius.neural_tracing', 'vesuvius/neural_tracing')]:
        m = types.ModuleType(name); m.__path__ = [str(src/sub)]; sys.modules[name] = m  # skip heavy package __init__

    class _Stub(types.ModuleType):  # CT-reading deps imported but unused here
        __path__ = []
        def __getattr__(self, k): return _Stub(k) if not k.startswith('__') else super().__getattribute__(k)
        def __call__(self, *a, **k): return self
        def __mro_entries__(self, bases): return (object,)

    while True:
        try:
            from vesuvius.neural_tracing.tracers.fibers.evaluation import scoring; break
        except ModuleNotFoundError as e:
            if e.name.split('.')[0] in ('vesuvius', 'vc3d_fiber_format'): raise
            sys.modules[e.name] = _Stub(e.name)
    from vesuvius.neural_tracing.tracers.shared.annotations.vc3d import load_fibers
    V['scoring'], V['load_fibers'] = scoring, load_fibers


# ------------------------------------------------------------------------------------------------------ human fibers
@dataclass
class Fiber:
    """One human fiber after villa's load_fibers (trace-grid vx, 1 vx spacing, kinks repaired)."""
    name: str
    points: np.ndarray
    s: np.ndarray
    tag: str  # H/V family
    ctrl: np.ndarray  # arc positions of the human control points (span starts, then the last span end)
    endpoint_stop: tuple
    source_hash: str
    tf: object  # villa's TracedFiber, for villa's own matching


def load_gt(fiber_dir, grid_scale=8.0):
    """(name -> Fiber, junctions) for a scroll; a junction is dict(owner, branch_file, xyz) from the raw JSON `branches`."""
    gt = {}
    for f in V['load_fibers'](str(fiber_dir), grid_scale):
        ctrl = np.r_[f.spans[0].start, [sp.end for sp in f.spans]]
        gt[f.name] = Fiber(f.name, f.points, f.s, f.tag, ctrl, tuple(f.endpoint_stop), f.source_hash, f)
    junc = []
    for p in sorted(glob.glob(os.path.join(fiber_dir, '*.json'))):
        for b in json.load(open(p, encoding='utf-8')).get('branches') or []:
            junc.append(dict(owner=os.path.basename(p), branch_file=b.get('branch_file'),
                             xyz=np.asarray(b['control_point_position'], float)/grid_scale))
    return gt, junc


# --------------------------------------------------------------------------------------------------------- traces
def run_dir(data, src, conf='v2'):
    return Path(data)/'runs'/conf/src


def stream(run):
    """paths.jsonl records one line at a time (the file is large)."""
    with open(Path(run)/'paths.jsonl') as fh:
        for line in fh:
            yield json.loads(line)


def trace_table(data, src, gt, pts, conf='v2'):
    """Scored-region vertices of every trace of one scroll, mapped to the nearest point of the trace's own fiber.

    List indexed by trace: dict(fiber, t0, sign, P (n,3), L, d (stored GT distance), gt_idx, s, dist), with
    n = len(pts[(src, t)][0]); gt_idx / s / dist are the global cKDTree nearest point of the fiber, its arc, its distance."""
    run = run_dir(data, src, conf)
    rows = pd.read_csv(run/'rows.csv')
    trees, out = {}, []
    for t, (r, p) in enumerate(zip(rows.itertuples(), stream(run))):
        Ls, ds = pts[(src, t)]
        P = np.asarray(p['path'], float).reshape(-1, 3)[:len(Ls)]
        f = gt.get(r.fiber_name)
        if f is None or not len(P):
            idx = np.full(len(P), -1); sv = np.full(len(P), np.nan); dist = np.full(len(P), np.nan)
        else:
            if r.fiber_name not in trees:
                trees[r.fiber_name] = cKDTree(f.points)
            dist, idx = trees[r.fiber_name].query(P)
            sv = f.s[idx]
        out.append(dict(fiber=r.fiber_name, t0=float(r.t0), sign=float(r.sign), P=P, L=Ls, d=ds, gt_idx=idx, s=sv, dist=dist))
    return out


# ------------------------------------------------------------------------------------------------------ data checks
def check_hash(rows, man, gt, fdir):
    """a. Loaded source_hash of every evaluated fiber vs the manifest's (by name)."""
    ev = list(dict.fromkeys(rows.fiber_name))
    want = {f['name']: f['source_hash'] for f in man['fibers']}
    absent = [n for n in ev if not (Path(fdir)/n).exists()]
    return dict(n_evaluated=len(ev), n_matched=sum(n in gt and want.get(n) == gt[n].source_hash for n in ev),
                mismatched=[n for n in ev if n in gt and n in want and want[n] != gt[n].source_hash],
                missing_from_dir=absent, in_dir_but_not_loaded=[n for n in ev if n not in gt and n not in absent],
                not_in_manifest=[n for n in ev if n not in want])


def check_distance(run, rows, gt):
    """b. Villa's matched_profile (what scored the traces) on the re-downloaded GT vs the stored gt_distance (< 10 vx)."""
    diffs, tmax, skipped = [], [], 0
    for r, p in zip(rows.itertuples(), stream(run)):
        P = np.asarray(p['path'], float).reshape(-1, 3)
        if r.fiber_name not in gt or not len(P):
            skipped += 1
            continue
        L = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        d, _ = V['scoring'].matched_profile(P, L, gt[r.fiber_name].tf, r.t0, r.sign, S.TOL)
        st = np.asarray(p['gt_distance'], float)
        m = st < 10
        diff = np.abs(d[m]-st[m])
        diffs.append(diff)
        tmax.append(diff.max() if len(diff) else 0.0)
    x = np.concatenate(diffs)
    return dict(n_vertices=len(x), median=float(np.median(x)), p99=float(np.percentile(x, 99)), max=float(x.max()),
                n_traces=len(tmax), n_traces_max_gt_0p01=int(np.sum(np.asarray(tmax) > 0.01)), n_traces_skipped=skipped)


def check_arc(tt):
    """c. On vertices with stored d <= 3: steps of the mapped s against `sign` by > 2 vx; first vertex vs t0.
    Also (extra): global cKDTree distance vs the stored one, vertices with stored d < 10."""
    steps = back = first = first_ok = 0
    n10 = n10_bad = 0
    gap = []
    for q in tt:
        if not len(q['P']) or np.isnan(q['s']).any():
            continue
        first += 1
        first_ok += abs(q['s'][0]-q['t0']) <= 2
        ok = q['d'] <= 3
        pair = ok[:-1] & ok[1:]
        step = np.diff(q['s'])*q['sign']
        steps += int(pair.sum())
        back += int((pair & (step < -2)).sum())
        m = q['d'] < 10
        diff = np.abs(q['dist'][m]-q['d'][m])
        n10 += int(m.sum())
        n10_bad += int((diff > 0.01).sum())
        gap.append(diff)
    gap = np.concatenate(gap) if gap else np.zeros(1)
    return dict(n_steps=steps, share_steps_back_gt_2=back/max(steps, 1), n_traces=first, share_first_within_2_of_t0=first_ok/max(first, 1),
                kdtree_vs_stored=dict(n_vertices=n10, share_diff_gt_0p01=n10_bad/max(n10, 1), p99=float(np.percentile(gap, 99)),
                                      max=float(gap.max())))


def check_junctions(junc, gt):
    """d. Distance of each junction point (control_point_position / grid_scale) to its owner fiber and to its branch_file fiber."""
    trees = {}
    def near(name, xyz):
        if name not in trees:
            trees[name] = cKDTree(gt[name].points)
        return float(trees[name].query(xyz)[0])
    own = [near(j['owner'], j['xyz']) for j in junc if j['owner'] in gt]
    br = [near(j['branch_file'], j['xyz']) for j in junc if j['branch_file'] in gt]
    q = lambda a: dict(median=float(np.median(a)), p99=float(np.percentile(a, 99)), max=float(np.max(a))) if len(a) else None
    return dict(n=len(junc), n_unique_positions=len({tuple(np.round(j['xyz'], 3)) for j in junc}),
                owner_loaded=len(own), owner=q(own), branch_file_loaded=len(br), branch_file_not_loaded=len(junc)-len(br),
                branch_file=q(br))


def check_counts(fdir, gt, rows, tt):
    """e. Fiber files, GT points, control-point spacing, traces and scored-region vertices."""
    n_files = len(glob.glob(os.path.join(fdir, '*.json')))
    spacing = lambda fs: float(np.median(np.concatenate([np.diff(f.ctrl) for f in fs])))
    ev = [gt[n] for n in dict.fromkeys(rows.fiber_name) if n in gt]
    return dict(n_files=n_files, n_loaded=len(gt), n_skipped=n_files-len(gt), gt_points=int(sum(len(f.points) for f in gt.values())),
                median_ctrl_spacing=spacing(gt.values()), median_ctrl_spacing_evaluated=spacing(ev),
                n_traces=len(rows), scored_vertices=int(sum(len(q['P']) for q in tt)))


def check_episodes(eps):
    """f. Episode counts by class and scroll against the values quoted in the plan context."""
    got = {c: {s: int(((eps.source == s) & (eps.cls == c)).sum()) for s in S.SOURCES} for c in EXPECT}
    return dict(counts=got, expected=EXPECT, match=got == EXPECT)


def checks_main(data, fibers, villa, out):
    t_start = time.time()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    init_villa(villa)
    S.ROOT = Path(data)
    rows_all, eps, pts = S.load('v2')  # asserts that the stored profiles reproduce villa's scoring
    print(f'stats.load ok ({time.time()-t_start:.0f}s); rounding ties: {len(S.TIES)}', flush=True)
    res = dict(command=' '.join(sys.argv), data=str(data), n_rounding_ties=len(S.TIES),
               inputs={f'{s}/{f}': hashlib.sha256((run_dir(data, s)/f).read_bytes()).hexdigest()
                       for s in S.SOURCES for f in ('rows.csv', 'config.json', 'manifest.json')},
               episodes=check_episodes(eps), scrolls={})
    for src in S.SOURCES:
        t1 = time.time()
        run = run_dir(data, src)
        cfg = json.loads((run/'config.json').read_text())
        man = json.loads((run/'manifest.json').read_text())
        fdir = Path(fibers)/Path(cfg['fibers_dir']).name
        gt, junc = load_gt(fdir, float(cfg['grid_scale']))
        rows = pd.read_csv(run/'rows.csv')
        tt = trace_table(data, src, gt, pts)
        # trace i of rows.csv must be line i of paths.jsonl (same fiber, seed and direction)
        meta = [(p['fiber_name'], p['t0'], p['sign']) for p in stream(run)]
        aligned = len(meta) == len(rows) and all(m[0] == r.fiber_name and abs(m[1]-r.t0) < 1e-6 and m[2] == r.sign
                                                 for m, r in zip(meta, rows.itertuples()))
        res['scrolls'][src] = dict(fibers_dir=str(fdir), grid_scale=float(cfg['grid_scale']), rows_aligned_with_paths=bool(aligned),
                                   a_hash=check_hash(rows, man, gt, fdir), b_distance=check_distance(run, rows, gt),
                                   c_arc=check_arc(tt), d_junctions=check_junctions(junc, gt), e_counts=check_counts(fdir, gt, rows, tt))
        print(f'{src} done ({time.time()-t1:.0f}s)', flush=True)
        del tt
    res['runtime_s'] = time.time()-t_start
    res['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    (out/'data_checks.json').write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    summary(res)


def summary(res):
    print('f. episodes match expected:', res['episodes']['match'])
    for c, v in res['episodes']['counts'].items():
        print('  ', c, v)
    for src, r in res['scrolls'].items():
        a, b, c, d, e = r['a_hash'], r['b_distance'], r['c_arc'], r['d_junctions'], r['e_counts']
        print(f'\n== {src}  aligned={r["rows_aligned_with_paths"]}')
        print(f' a: evaluated {a["n_evaluated"]} matched {a["n_matched"]} mismatched {a["mismatched"]} missing {a["missing_from_dir"]} '
              f'unloaded {a["in_dir_but_not_loaded"]} not_in_manifest {a["not_in_manifest"]}')
        print(f' b: n={b["n_vertices"]} median {b["median"]:.2g} p99 {b["p99"]:.2g} max {b["max"]:.3g} traces>0.01: {b["n_traces_max_gt_0p01"]}/{b["n_traces"]}')
        print(f' c: steps back>2: {c["share_steps_back_gt_2"]:.4g} of {c["n_steps"]}; first vertex within 2 of t0: {c["share_first_within_2_of_t0"]:.4g}; '
              f'kdtree vs stored: {c["kdtree_vs_stored"]}')
        print(f' d: {d}')
        print(f' e: {e}')
    print(f'\nruntime {res["runtime_s"]:.0f}s peak RSS {res["peak_rss_mb"]:.0f} MB')


# ---------------------------------------------------------------------------------------- stage 2a: covariates / tables
CM = 0.00192                                   # cm per trace-grid vx (1 vx = 19.2 um)
K_NEIGH, R_NEIGH, THIN, CAP = 256, 100.0, 2, 100.0   # neighbour query: k, radius (vx), candidate thinning, cap of distances
FAR_CHECK = 100.0                              # vertices farther than this from the GT are not asserted in trace_arcs
GT_COVS = ('bend', 'kink', 'cross_d', 'par_d', 'junc_d', 'end_d', 'cp_d')   # continuous covariates stored per GT point
CONT = ('bend', 'kink', 'cross_d', 'par_d', 'junc_d', 'seed_d', 'end_d', 'cp_d')   # summarised in sanity.json
CAPPED = ('cross_d', 'par_d', 'junc_d')
QS = (0.10, 1/3, 0.50, 2/3, 0.90)
TERMINAL = ('hard_switch', 'drift', 'ended')   # episode classes that end a trace's at-risk exposure


def pos(f, x):
    """P(x): fiber position at arc length x (linear interpolation of the points over s; clamped outside [s0, s1])."""
    return np.stack([np.interp(x, f.s, f.points[:, k]) for k in range(3)], 1)


def angle_deg(a, b):
    """Angle in degrees (0-180) between row vectors a and b; NaN where either has zero length."""
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    ok = (na > 0) & (nb > 0)
    c = np.einsum('...j,...j->...', a, b)/np.where(ok, na*nb, 1.0)
    return np.where(ok, np.degrees(np.arccos(np.clip(c, -1.0, 1.0))), np.nan)


def fiber_dir(f):
    """Local unit direction unit(P(min(s+4, s1)) - P(max(s-4, s0))) at every point of a fiber."""
    v = pos(f, np.minimum(f.s+4, f.s[-1])) - pos(f, np.maximum(f.s-4, f.s[0]))
    n = np.linalg.norm(v, axis=1)
    return np.where(n[:, None] > 0, v/np.where(n > 0, n, 1.0)[:, None], np.nan)


def bend_kink(f):
    """bend: angle between chords P(s)-P(s-16) and P(s+16)-P(s) (NaN within 16 vx of an end).
    kink: max over u = -8..8 of the angle between 4-vx chords P(s+u)-P(s+u-4) and P(s+u+4)-P(s+u) (NaN if no u qualifies)."""
    s, s0, s1 = f.s, f.s[0], f.s[-1]
    bend = np.where((s-16 >= s0) & (s+16 <= s1), angle_deg(f.points-pos(f, s-16), pos(f, s+16)-f.points), np.nan)
    kink = np.full(len(s), np.nan)
    for u in range(-8, 9):
        c = s+u
        ok = (c-4 >= s0) & (c+4 <= s1)
        kink = np.fmax(kink, np.where(ok, angle_deg(pos(f, c)-pos(f, c-4), pos(f, c+4)-pos(f, c)), np.nan))
    return bend, kink


def control_distance(f):
    """Arc-length distance from each point to the nearest human control point."""
    c = np.r_[-np.inf, np.sort(np.asarray(f.ctrl, float)), np.inf]
    i = np.searchsorted(c, f.s)
    return np.minimum(np.abs(f.s-c[i-1]), np.abs(c[i]-f.s))


def neighbour_setup(gt, dirs):
    """Candidate points of every loaded fiber of a scroll, thinned to every THIN-th point, with their directions."""
    names = list(gt)
    return dict(fid={n: i for i, n in enumerate(names)},
                pts=np.concatenate([gt[n].points[::THIN] for n in names]),
                dir=np.concatenate([dirs[n][::THIN] for n in names]),
                fiber=np.concatenate([np.full(len(gt[n].points[::THIN]), i, np.int32) for i, n in enumerate(names)]))


def neighbour_distances(setup, name, P, D, chunk=4096):
    """cross_d, par_d (cap CAP) and truncation flags for the points P (directions D) of fiber `name`.

    Candidates: points of the other fibers inside the fiber's bounding box expanded by R_NEIGH; the K_NEIGH nearest within
    R_NEIGH are classed by the unsigned angle between local directions (>= 45 deg crossing, < 45 deg parallel). A class that is
    absent gets CAP; it is flagged truncated when all K_NEIGH slots were filled (the class might exist just beyond)."""
    m = len(P)
    cross, par = np.full(m, CAP), np.full(m, CAP)
    tcross, tpar = np.zeros(m, bool), np.zeros(m, bool)
    sel = np.flatnonzero((setup['pts'] >= P.min(0)-R_NEIGH).all(1) & (setup['pts'] <= P.max(0)+R_NEIGH).all(1)
                         & (setup['fiber'] != setup['fid'][name]))
    if not len(sel):
        return cross, par, tcross, tpar
    tree = cKDTree(setup['pts'][sel])
    cdir = np.vstack([setup['dir'][sel], np.zeros((1, 3))])  # row tree.n = the 'missing neighbour' index
    for a in range(0, m, chunk):
        d, idx = tree.query(P[a:a+chunk], k=K_NEIGH, distance_upper_bound=R_NEIGH, workers=2)
        ang = np.degrees(np.arccos(np.minimum(np.abs(np.einsum('mj,mkj->mk', D[a:a+chunk], cdir[idx])), 1.0)))
        fin = np.isfinite(d)
        dx = np.where(fin & (ang >= 45), d, np.inf).min(1)
        dp = np.where(fin & (ang < 45), d, np.inf).min(1)
        full = fin[:, -1]
        cross[a:a+chunk], par[a:a+chunk] = np.where(np.isfinite(dx), dx, CAP), np.where(np.isfinite(dp), dp, CAP)
        tcross[a:a+chunk], tpar[a:a+chunk] = ~np.isfinite(dx) & full, ~np.isfinite(dp) & full
    return cross, par, tcross, tpar


def scroll_covariates(gt, junc, ev):
    """Table A for one scroll: covariates of every point of every evaluated fiber (list `ev` of names), as a dict of arrays
    ordered by fiber (index into ev) then point; `offsets[i]` is the first row of fiber i."""
    dirs = {n: fiber_dir(f) for n, f in gt.items()}
    setup = neighbour_setup(gt, dirs)
    jt = cKDTree(np.array([j['xyz'] for j in junc])) if junc else None
    parts = []
    for i, n in enumerate(ev):
        f = gt[n]
        bend, kink = bend_kink(f)
        cross, par, tc, tp = neighbour_distances(setup, n, f.points, dirs[n])
        junc_d = np.minimum(jt.query(f.points)[0], CAP) if jt is not None else np.full(len(f.s), CAP)
        parts.append(dict(fiber=np.full(len(f.s), i, np.int32), pt=np.arange(len(f.s), dtype=np.int32), s=f.s, bend=bend, kink=kink,
                          cross_d=cross, par_d=par, junc_d=junc_d, end_d=np.minimum(f.s-f.s[0], f.s[-1]-f.s),
                          cp_d=control_distance(f), trunc_cross=tc, trunc_par=tp, family=np.full(len(f.s), f.tag),
                          dir=dirs[n]))
    cov = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    for k in (*GT_COVS, 'dir'):
        cov[k] = cov[k].astype(np.float32)
    cov['fiber_names'] = np.array(ev)
    cov['offsets'] = np.r_[0, np.cumsum([len(gt[n].s) for n in ev])].astype(np.int64)
    return cov


def arc_to_idx(s, arc):
    """Index of the fiber point whose s is nearest to `arc` (ties go to the lower index); -1 where arc is not finite."""
    i = np.clip(np.searchsorted(s, arc), 1, len(s)-1)
    return np.where(np.isfinite(arc), np.where(arc-s[i-1] <= s[i]-arc, i-1, i), -1).astype(np.int32)


def cov_rows(cov, fiber, pt):
    """Row of the covariate table for (fiber index, point index)."""
    return cov['offsets'][np.asarray(fiber)] + np.asarray(pt)


def trace_arcs(data, src, rows, gt, pts, conf='v2'):
    """Villa's own matched arc (scoring.matched_profile, the window-restricted nearest vertex villa uses to define departures)
    for every scored-region vertex of every trace, and the fiber point whose s is nearest to it.

    Returns (list indexed by trace of dict(arc, pt), diagnostics). Asserts that the returned distance equals the stored one
    (within 0.002) on the scored-region vertices whose stored distance is <= FAR_CHECK vx, and that the mapped GT point lies at
    that distance from the vertex there. Vertices farther than that are only recorded (`far_exceptions`): villa's
    window-restricted match can differ from the stored value far from the fiber, where no table uses it."""
    run = run_dir(data, src, conf)
    out = []
    dg = dict(n_vertices=0, n_nonfinite=0, n_asserted=0, max_abs_d_vs_stored=0.0, max_abs_arc_vs_point_s=0.0,
              max_abs_mapped_point_dist_vs_stored=0.0, far_exceptions=[])
    for t, (r, p) in enumerate(zip(rows.itertuples(), stream(run))):
        Ls, ds = pts[(src, t)]
        n = len(Ls)
        P = np.asarray(p['path'], float).reshape(-1, 3)
        f = gt[r.fiber_name]
        if not n:
            out.append(dict(arc=np.zeros(0), pt=np.zeros(0, np.int32)))
            continue
        L = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        assert np.allclose(L[:n], Ls, atol=1e-9, rtol=0), (src, t)
        d, arc = V['scoring'].matched_profile(P, L, f.tf, r.t0, r.sign, S.TOL)
        assert len(d) == len(arc) == len(P), (src, t, len(d), len(P))
        d, arc = d[:n], arc[:n]
        fin = np.isfinite(d) & np.isfinite(ds)
        assert np.array_equal(np.isfinite(d), np.isfinite(ds)), (src, t, 'non-finite pattern differs')
        pt = arc_to_idx(f.s, arc)
        dg['n_vertices'] += n
        dg['n_nonfinite'] += int((~fin).sum())
        near = fin & (ds <= FAR_CHECK)
        dg['n_asserted'] += int(near.sum())
        if near.any():
            dg['max_abs_d_vs_stored'] = max(dg['max_abs_d_vs_stored'], float(np.abs(d[near]-ds[near]).max()))
            dg['max_abs_arc_vs_point_s'] = max(dg['max_abs_arc_vs_point_s'], float(np.abs(f.s[pt]-arc)[near].max()))
            dm = np.linalg.norm(P[:n]-f.points[pt], axis=1)
            dg['max_abs_mapped_point_dist_vs_stored'] = max(dg['max_abs_mapped_point_dist_vs_stored'], float(np.abs(dm[near]-ds[near]).max()))
        for j in np.flatnonzero(fin & (ds > FAR_CHECK) & (np.abs(d-ds) > 0.002)):
            dg['far_exceptions'].append(dict(trace=t, vertex=int(j), stored=float(ds[j]), villa=float(d[j])))
        assert dg['max_abs_d_vs_stored'] <= 0.002, (src, t, 'villa distance differs from stored', dg)
        assert dg['max_abs_mapped_point_dist_vs_stored'] <= 0.002, (src, t, 'mapped GT point is not at the matched distance', dg)
        out.append(dict(arc=np.asarray(arc, float), pt=pt))
    return out, dg


def scroll_tables(si, src, rows, ev, gt, cov, arcs, pts, rows_s, eps_s):
    """B, C, D for one scroll: (exposure dict of arrays, episodes DataFrame, traces DataFrame, float64 at-risk length)."""
    fidx = {n: i for i, n in enumerate(ev)}
    by_trace = {t: g for t, g in eps_s.groupby('trace', sort=False)}
    expo = {k: [] for k in ('trace', 'fiber', 'pt', 'length', 'seed_d')}
    tr_rows, ep_rows, at_risk = [], [], 0.0
    pw_on = rows_s.set_index('trace')['pw_on']
    for t, r in enumerate(rows.itertuples()):
        Ls, ds = pts[(src, t)]
        n = len(Ls)
        q = arcs[t]
        eps_t = by_trace.get(t)
        eps_l = list(eps_t.itertuples()) if eps_t is not None else []
        term = next((e for e in eps_l if e.cls in TERMINAL), None)
        limit = term.onset_idx if term is not None else n
        fi = fidx[r.fiber_name]
        ii = np.arange(1, n)[(ds[1:] <= S.TOL) & (np.arange(1, n) < limit)] if n > 1 else np.zeros(0, int)
        seg = Ls[ii]-Ls[ii-1] if len(ii) else np.zeros(0)
        assert seg.sum() <= pw_on[t] + 1e-6, (src, t, seg.sum(), pw_on[t])
        at_risk += float(seg.sum())
        if len(ii):
            assert (q['pt'][ii] >= 0).all(), (src, t)
            expo['trace'].append(np.full(len(ii), t, np.int32)); expo['fiber'].append(np.full(len(ii), fi, np.int32))
            expo['pt'].append(q['pt'][ii]); expo['length'].append(seg.astype(np.float32))
            expo['seed_d'].append(np.abs(q['arc'][ii]-r.t0).astype(np.float32))
            reach = float(q['arc'][ii[np.argmax((q['arc'][ii]-r.t0)*r.sign)]])
        else:
            reach = np.nan
        tr_rows.append(dict(src=src, trace=t, fiber_name=r.fiber_name, sign=r.sign, t0=r.t0, reach_arc=reach,
                            terminal=term.cls if term is not None else 'none',
                            terminal_dep_arc=(q['arc'][term.onset_idx-1] if term.onset_idx > 0 else r.t0) if term is not None else np.nan,
                            reason=r.reason))
        for e in eps_l:
            assert e.onset_idx == 0 or ds[e.onset_idx-1] <= S.TOL, (src, t, e.onset_idx)  # the vertex before an onset is on the fiber
            dep = float(q['arc'][e.onset_idx-1]) if e.onset_idx > 0 else float(r.t0)
            pt = int(arc_to_idx(gt[r.fiber_name].s, np.array([dep]))[0])
            ep_rows.append(dict(src=src, trace=t, fiber_name=r.fiber_name, family=r.family, cls=e.cls, onset_idx=e.onset_idx,
                                onset_len=e.onset_len, peak=e.peak, sign=r.sign, t0=r.t0, dep_arc=dep, fiber_idx=fi, gt_pt=pt,
                                seed_d=abs(dep-r.t0)))
    expo = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in expo.items()}
    ep = pd.DataFrame(ep_rows)
    valid = (ep['gt_pt'] >= 0).to_numpy()
    row = cov_rows(cov, ep['fiber_idx'], np.where(valid, ep['gt_pt'], 0))
    assert (cov['family'][row] == ep['family'].to_numpy())[valid].all()
    for k in GT_COVS:
        ep[k] = np.where(valid, cov[k][row], np.nan)
    ep['trunc_cross'], ep['trunc_par'] = cov['trunc_cross'][row] & valid, cov['trunc_par'][row] & valid
    for j, k in enumerate('xyz'):
        ep['dir_'+k] = np.where(valid, cov['dir'][row, j], np.nan)
    return expo, ep, pd.DataFrame(tr_rows), at_risk


def wquantiles(v, w, qs=QS):
    """Exposure-weighted quantiles of the finite values: the smallest value whose cumulative weight share is >= q."""
    ok = np.isfinite(v)
    v, w = v[ok], np.asarray(w[ok], float)
    if not len(v) or w.sum() <= 0:
        return [np.nan]*len(qs)
    o = np.argsort(v, kind='stable')
    v, c = v[o], np.cumsum(w[o])
    c /= c[-1]
    return [float(v[min(int(np.searchsorted(c, q)), len(v)-1)]) for q in qs]


def cov_summary(v, w, capped=False):
    """Exposure-weighted NaN share, quantiles (levels in %) and, for distances capped at 100 vx, the share at the cap."""
    v, w = np.asarray(v, float), np.asarray(w, float)
    nan = ~np.isfinite(v)
    out = dict(nan_share=float(w[nan].sum()/w.sum()), quantiles={f'{100*q:.1f}': x for q, x in zip(QS, wquantiles(v, w))})
    if capped:
        out['share_at_cap'] = float(w[~nan & (v >= CAP)].sum()/w.sum())
    return out


def sanity_numbers(tab, rows_all, ep_all, diag):
    """sanity.json content from the saved tables only (so the saved files are what is checked)."""
    ex = np.load(tab/'exposure.npz')
    res, pooled_v, pooled_w, pooled_flags, pooled_fam = {}, {k: [] for k in CONT}, [], {'trunc_cross': [], 'trunc_par': []}, []
    for si, src in enumerate(S.SOURCES):
        cov = np.load(tab/f'covariates_{src}.npz')
        m = ex['src'] == si
        w = ex['length'][m].astype(float)
        row = cov_rows(cov, ex['fiber'][m], ex['pt'][m])
        vals = {k: (ex['seed_d'][m] if k == 'seed_d' else cov[k][row]) for k in CONT}
        pw = float(rows_all[rows_all.source == src]['pw_on'].sum())
        cm, pwcm = w.sum()*CM, pw*CM
        ep = ep_all[ep_all.src == src]
        res[src] = dict(
            n_evaluated_fibers=len(cov['fiber_names']), n_gt_points_evaluated=int(len(cov['s'])), n_at_risk_segments=int(m.sum()),
            at_risk_cm=cm, pw_on_cm=pwcm, pw_on_minus_at_risk_cm=pwcm-cm, at_risk_share_of_pw_on=cm/pwcm,
            at_risk_cm_float64_unrounded=diag[src]['at_risk_vx_float64']*CM,
            episodes_by_cls={c: int((ep.cls == c).sum()) for c in EXPECT}, episodes_valid_gt_index=int((ep.gt_pt >= 0).sum()),
            episodes_total=int(len(ep)),
            covariates={k: cov_summary(vals[k], w, k in CAPPED) for k in CONT},
            trunc_share={k: float(w[cov[k][row]].sum()/w.sum()) for k in ('trunc_cross', 'trunc_par')},
            exposure_share_by_family={f: float(w[cov['family'][row] == f].sum()/w.sum()) for f in ('H', 'V')},
            n_nan_dir_evaluated_points=int(np.isnan(cov['dir']).any(1).sum()), checks=diag[src]['arc'])
        for k in CONT:
            pooled_v[k].append(vals[k])
        pooled_w.append(w)
        for k in pooled_flags:
            pooled_flags[k].append(cov[k][row])
        pooled_fam.append(cov['family'][row])
    w = np.concatenate(pooled_w)
    fam = np.concatenate(pooled_fam)
    res['pooled'] = dict(at_risk_cm=w.sum()*CM, pw_on_cm=float(rows_all['pw_on'].sum())*CM,
                         covariates={k: cov_summary(np.concatenate(pooled_v[k]), w, k in CAPPED) for k in CONT},
                         trunc_share={k: float(w[np.concatenate(v)].sum()/w.sum()) for k, v in pooled_flags.items()},
                         exposure_share_by_family={f: float(w[fam == f].sum()/w.sum()) for f in ('H', 'V')})
    got = {c: {s: int(((ep_all.src == s) & (ep_all.cls == c)).sum()) for s in S.SOURCES} for c in EXPECT}
    res['episodes_match_stage1'] = bool(got == EXPECT)
    res['episode_counts'] = got
    return res


def tables_main(data, fibers, villa, out):
    t_start = time.time()
    tab = Path(out)/'tables'; tab.mkdir(parents=True, exist_ok=True)
    init_villa(villa)
    S.ROOT = Path(data)
    rows_all, eps, pts = S.load('v2')
    print(f'stats.load ok ({time.time()-t_start:.0f}s); rounding ties: {len(S.TIES)}', flush=True)
    expo, eps_out, tr_out, diag = [], [], [], {}
    for si, src in enumerate(S.SOURCES):
        t1 = time.time()
        run = run_dir(data, src)
        cfg = json.loads((run/'config.json').read_text())
        gt, junc = load_gt(Path(fibers)/Path(cfg['fibers_dir']).name, float(cfg['grid_scale']))
        rows = pd.read_csv(run/'rows.csv')
        ev = list(dict.fromkeys(rows.fiber_name))
        assert all(n in gt for n in ev), [n for n in ev if n not in gt]
        cov = scroll_covariates(gt, junc, ev)
        np.savez(tab/f'covariates_{src}.npz', **cov)
        print(f'{src}: covariates {len(cov["s"])} points of {len(ev)} fibers ({time.time()-t1:.0f}s)', flush=True)
        arcs, dg = trace_arcs(data, src, rows, gt, pts)
        print(f'{src}: arcs ({time.time()-t1:.0f}s) {dg}', flush=True)
        e, ep, tr, at_risk = scroll_tables(si, src, rows, ev, gt, cov, arcs, pts, rows_all[rows_all.source == src],
                                           eps[eps.source == src])
        e['src'] = np.full(len(e['trace']), si, np.int8)
        expo.append(e); eps_out.append(ep); tr_out.append(tr)
        diag[src] = dict(arc=dg, at_risk_vx_float64=at_risk)
        print(f'{src} done ({time.time()-t1:.0f}s): {len(e["trace"])} at-risk segments, {len(ep)} episodes', flush=True)
        del gt, cov, arcs
    np.savez(tab/'exposure.npz', sources=np.array(S.SOURCES), **{k: np.concatenate([e[k] for e in expo]) for k in expo[0]})
    ep_all = pd.concat(eps_out, ignore_index=True)
    ep_all.to_csv(tab/'episodes.csv', index=False)
    pd.concat(tr_out, ignore_index=True).to_csv(tab/'traces.csv', index=False)
    res = sanity_numbers(tab, rows_all, ep_all, diag)
    res['command'] = ' '.join(sys.argv)
    res['n_rounding_ties'] = len(S.TIES)
    res['runtime_s'] = time.time()-t_start
    res['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    (tab/'sanity.json').write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print(json.dumps(res, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


# ------------------------------------------------------------------------- stage 2c: past-end labels, strict exposure (post-hoc)
END_TOL, END_EXTRA = 3.0, 6.0   # villa score_trace: endpoint_crossing(..., tol, reach=tol + DEFAULT_MAX_RECOVERY_DISTANCE, start=onset)


def past_end_flag(sc, P, Ls, arc, f, sign, onset_idx):
    """(past_end, end_tagged, crossing_len) of one non-returned episode of a trace, as villa's score_trace labels a departed trace.

    The end the trace travels toward is the last GT point (sign > 0) or the first. If it is tagged as a physical endpoint the
    length beyond it is an overrun, not unknown: past_end False. Otherwise past_end = the path crosses that end's plane within
    END_TOL + END_EXTRA of the endpoint after the onset (scoring.endpoint_crossing on the scored region P, Ls, arc)."""
    if bool(f.endpoint_stop[1 if sign > 0 else 0]):
        return False, True, np.nan
    c = sc.endpoint_crossing(P, Ls, arc, f.tf, sign, END_TOL, reach=END_TOL+END_EXTRA, start=onset_idx)
    return c is not None, False, np.nan if c is None else float(c)


def flags_main(data, fibers, villa, out):
    """Stage 2c, part 1: episode_flags.csv (columns src, trace, order, cls, onset_idx, end_d, past_end, end_tagged, crossing_len) for every
    hard_switch / drift / ended episode and for the first episode of every trace (any class; returned -> past_end False, end_tagged and
    crossing_len empty), and the strict exposure (segments with ds <= TOL before the onset of the trace's first episode of ANY class)."""
    t_start = time.time()
    out = Path(out); tab = out/'tables'; res = out/'results'; res.mkdir(parents=True, exist_ok=True)
    init_villa(villa)
    sc = V['scoring']
    assert sc.DEFAULT_MAX_RECOVERY_DISTANCE == END_EXTRA, sc.DEFAULT_MAX_RECOVERY_DISTANCE
    S.ROOT = Path(data)
    rows_all, eps2, pts = S.load('v2')
    ep_all = pd.read_csv(tab/'episodes.csv', float_precision='round_trip')
    assert len(ep_all) == len(eps2) and (ep_all.src.to_numpy() == eps2.source.to_numpy()).all() \
        and (ep_all.trace.to_numpy() == eps2.trace.to_numpy()).all() and (ep_all.cls.to_numpy() == eps2.cls.to_numpy()).all() \
        and (ep_all.onset_idx.to_numpy() == eps2.onset_idx.to_numpy()).all(), 'episodes.csv differs from stats.load'
    print(f'stats.load ok ({time.time()-t_start:.0f}s)', flush=True)
    flag_rows, expo, diag = [], [], {}
    for si, src in enumerate(S.SOURCES):
        t1 = time.time()
        run = run_dir(data, src)
        cfg = json.loads((run/'config.json').read_text())
        gt, _ = load_gt(Path(fibers)/Path(cfg['fibers_dir']).name, float(cfg['grid_scale']))
        rows = pd.read_csv(run/'rows.csv')
        ev = list(dict.fromkeys(rows.fiber_name))
        with np.load(tab/f'covariates_{src}.npz') as z:
            assert list(z['fiber_names']) == ev, src
        fidx = {n: i for i, n in enumerate(ev)}
        arcs, dg = trace_arcs(data, src, rows, gt, pts)
        print(f'{src}: arcs ({time.time()-t1:.0f}s)', flush=True)
        by_trace = {t: g for t, g in ep_all[ep_all.src == src].groupby('trace', sort=False)}
        e_seg = {k: [] for k in ('trace', 'fiber', 'pt', 'length', 'seed_d')}
        strict_vx, over, n_over, n_first = 0.0, -np.inf, 0, 0
        for t, (r, p) in enumerate(zip(rows.itertuples(), stream(run))):
            Ls, ds = pts[(src, t)]
            n = len(Ls)
            q = arcs[t]
            g = by_trace.get(t)
            if g is not None:
                n_first += 1
                P = np.asarray(p['path'], float).reshape(-1, 3)[:n]
                for k, e in enumerate(g.itertuples()):
                    if k > 0 and e.cls not in TERMINAL:
                        continue
                    pe, tag, cl = (False, np.nan, np.nan) if e.cls == 'returned' else \
                        past_end_flag(sc, P, Ls, q['arc'], gt[r.fiber_name], r.sign, int(e.onset_idx))
                    flag_rows.append(dict(src=src, trace=t, order=k, cls=e.cls, onset_idx=int(e.onset_idx), end_d=e.end_d,
                                          past_end=bool(pe), end_tagged=tag, crossing_len=cl))
            limit = int(g.iloc[0].onset_idx) if g is not None else n
            ii = np.arange(1, n)[(ds[1:] <= S.TOL) & (np.arange(1, n) < limit)] if n > 1 else np.zeros(0, int)
            seg = Ls[ii]-Ls[ii-1] if len(ii) else np.zeros(0)
            strict_vx += float(seg.sum())
            if seg.sum() > r.correct + 0.05:
                n_over += 1
            over = max(over, float(seg.sum()-r.correct))
            if len(ii):
                assert (q['pt'][ii] >= 0).all(), (src, t)
                e_seg['trace'].append(np.full(len(ii), t, np.int32)); e_seg['fiber'].append(np.full(len(ii), fidx[r.fiber_name], np.int32))
                e_seg['pt'].append(q['pt'][ii]); e_seg['length'].append(seg.astype(np.float64))
                e_seg['seed_d'].append(np.abs(q['arc'][ii]-r.t0).astype(np.float64))
        e_seg = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in e_seg.items()}
        e_seg['src'] = np.full(len(e_seg['trace']), si, np.int8)
        expo.append(e_seg)
        diag[src] = dict(strict_exposure_cm=strict_vx*CM, sum_rows_correct_cm=float(rows.correct.sum())*CM,
                         strict_exposure_vx=strict_vx, sum_rows_correct_vx=float(rows.correct.sum()),
                         max_trace_excess_over_correct_vx=over, traces_with_excess_gt_0p05_vx=n_over, n_segments=int(len(e_seg['trace'])),
                         traces_with_first_episode=n_first, traces_diverged_in_rows=int(rows.diverged.sum()), n_traces=len(rows),
                         trace_arcs_diagnostics=dg)
        print(f'{src} done ({time.time()-t1:.0f}s): strict {strict_vx*CM:.2f} cm vs sum correct {rows.correct.sum()*CM:.2f} cm', flush=True)
        del gt, arcs
    fl = pd.DataFrame(flag_rows)
    fl['end_tagged'] = fl['end_tagged'].astype('boolean')
    fl.to_csv(res/'episode_flags.csv', index=False)
    np.savez(res/'strict_exposure.npz', sources=np.array(S.SOURCES), **{k: np.concatenate([e[k] for e in expo]) for k in expo[0]})
    info = dict(command=' '.join(sys.argv), end_tol=END_TOL, end_reach=END_TOL+END_EXTRA, default_max_recovery_distance=float(sc.DEFAULT_MAX_RECOVERY_DISTANCE),
                scoring_py_sha256=hashlib.sha256(Path(sc.__file__).read_bytes()).hexdigest(), scrolls=diag, n_flag_rows=len(fl),
                runtime_s=time.time()-t_start, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    (res/'flags_2c.json').write_text(json.dumps(info, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print(json.dumps({s: {k: v for k, v in d.items() if k != 'trace_arcs_diagnostics'} for s, d in diag.items()}, indent=1))
    print(f'runtime {info["runtime_s"]:.0f}s peak RSS {info["peak_rss_mb"]:.0f} MB')


def crossdiag_main(data, fibers, villa, out):
    """Stage 2c diagnostic (post-hoc; feeds no table): for every hard_switch / drift / ended episode, where villa's end-plane crossing is found under
    other readings of the rule than the one in episode_flags.csv (which uses start = the episode's own onset).
      any_cross       endpoint_crossing(reach 9, start 0): a crossing anywhere in the scored region (the tagged end is not excluded)
      from_first      endpoint_crossing(reach 9, start = onset of the trace's FIRST episode): how score_trace calls it for a departed trace
      first_call      endpoint_crossing(reach 3, start 0, stop = onset of the trace's first episode): score_trace's first call
      now_*           the current villa score_trace on the full path (crossed_endpoint, diverged); rows_crossed: rows.csv crossed_endpoint."""
    t_start = time.time()
    out = Path(out); tab = out/'tables'; res = out/'results'
    init_villa(villa)
    sc = V['scoring']
    S.ROOT = Path(data)
    rows_all, eps2, pts = S.load('v2')
    ep_all = pd.read_csv(tab/'episodes.csv', float_precision='round_trip')
    flags = pd.read_csv(res/'episode_flags.csv', float_precision='round_trip')
    spec = {(a, b, c): d for a, b, c, d in zip(flags.src, flags.trace, flags.order, flags.past_end)}
    out_rows = []
    for si, src in enumerate(S.SOURCES):
        t1 = time.time()
        run = run_dir(data, src)
        cfg = json.loads((run/'config.json').read_text())
        gt, _ = load_gt(Path(fibers)/Path(cfg['fibers_dir']).name, float(cfg['grid_scale']))
        rows = pd.read_csv(run/'rows.csv')
        by_trace = {t: g for t, g in ep_all[(ep_all.src == src) & ep_all.cls.isin(TERMINAL)].groupby('trace', sort=False)}
        all_eps = {t: g for t, g in ep_all[ep_all.src == src].groupby('trace', sort=False)}
        for t, (r, p) in enumerate(zip(rows.itertuples(), stream(run))):
            g = by_trace.get(t)
            if g is None:
                continue
            f = gt[r.fiber_name]
            Ls, ds = pts[(src, t)]
            n = len(Ls)
            Pf = np.asarray(p['path'], float).reshape(-1, 3)
            L = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Pf, axis=0), axis=1))]
            d, arc = sc.matched_profile(Pf, L, f.tf, r.t0, r.sign, S.TOL)
            P, arc = Pf[:n], arc[:n]
            first_on = int(all_eps[t].iloc[0].onset_idx)
            now = sc.score_trace(Pf, f.tf, r.t0, r.sign)
            fc = sc.endpoint_crossing(P, Ls, arc, f.tf, r.sign, END_TOL, reach=END_TOL, start=0, stop=first_on)
            ac = sc.endpoint_crossing(P, Ls, arc, f.tf, r.sign, END_TOL, reach=END_TOL+END_EXTRA, start=0)
            ff = sc.endpoint_crossing(P, Ls, arc, f.tf, r.sign, END_TOL, reach=END_TOL+END_EXTRA, start=first_on)
            for e in g.itertuples():
                k = int(all_eps[t].index.get_loc(e.Index))
                out_rows.append(dict(src=src, trace=t, order=k, cls=e.cls, onset_idx=int(e.onset_idx), onset_len=e.onset_len, end_d=e.end_d,
                                     end_tagged=bool(f.endpoint_stop[1 if r.sign > 0 else 0]), past_end_spec=bool(spec[(src, t, k)]),
                                     any_cross=ac is not None, any_cross_len=np.nan if ac is None else ac,
                                     from_first=ff is not None, from_first_len=np.nan if ff is None else ff,
                                     first_call=fc is not None, first_call_len=np.nan if fc is None else fc,
                                     first_episode_onset_idx=first_on, now_crossed_endpoint=bool(now['crossed_endpoint']), now_diverged=bool(now['diverged']),
                                     now_unknown=float(now['unknown']), rows_crossed=bool(r.crossed_endpoint), rows_endpoint_known=bool(r.endpoint_known),
                                     rows_correct=float(r.correct), now_correct=float(now['correct'])))
        print(f'{src} done ({time.time()-t1:.0f}s)', flush=True)
    df = pd.DataFrame(out_rows)
    df.to_csv(res/'episode_crossing_diag.csv', index=False)
    print(f'{len(df)} rows; runtime {time.time()-t_start:.0f}s peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB')


def corrected_main(data, fibers, villa, out):
    """Stage 2d (post-hoc, decided after stage 2c): score only where a human label exists.

    For every trace with at least one episode: cut_len = endpoint_crossing(P, Ls, arc, fiber, sign, 3.0, reach=9.0, start=onset_idx of the
    trace's first episode of any class), skipped when the end the trace travels toward is tagged physical. If found, the scored region is
    cut to the vertices with Ls <= cut_len, the episodes are recomputed by stats.episodes on the cut (Ls, ds), and the at-risk and strict
    exposure keep only segments ending at those vertices. Traces without a cut or without episodes are unchanged."""
    t_start = time.time()
    out = Path(out); tab = out/'tables'; tab2 = out/'tables_2d'; res = out/'results'
    tab2.mkdir(parents=True, exist_ok=True); res.mkdir(parents=True, exist_ok=True)
    init_villa(villa)
    sc = V['scoring']
    assert sc.DEFAULT_MAX_RECOVERY_DISTANCE == END_EXTRA
    S.ROOT = Path(data)
    rows_all, eps2, pts = S.load('v2')
    ep_old = pd.read_csv(tab/'episodes.csv', float_precision='round_trip')
    assert len(ep_old) == len(eps2) and (ep_old.trace.to_numpy() == eps2.trace.to_numpy()).all() and (ep_old.cls.to_numpy() == eps2.cls.to_numpy()).all() \
        and (ep_old.onset_idx.to_numpy() == eps2.onset_idx.to_numpy()).all(), 'episodes.csv differs from stats.load'
    old_ex = np.load(tab/'exposure.npz')
    old_strict = np.load(res/'strict_exposure.npz')
    print(f'stats.load ok ({time.time()-t_start:.0f}s)', flush=True)
    expo, ep_new, tr_new, strict, trans, diag = [], [], [], [], [], {}
    for si, src in enumerate(S.SOURCES):
        t1 = time.time()
        run = run_dir(data, src)
        cfg = json.loads((run/'config.json').read_text())
        gt, _ = load_gt(Path(fibers)/Path(cfg['fibers_dir']).name, float(cfg['grid_scale']))
        rows = pd.read_csv(run/'rows.csv')
        ev = list(dict.fromkeys(rows.fiber_name))
        cov = dict(np.load(tab/f'covariates_{src}.npz'))
        assert list(cov['fiber_names']) == ev, src
        arcs, dg = trace_arcs(data, src, rows, gt, pts)
        print(f'{src}: covariates, arcs ({time.time()-t1:.0f}s)', flush=True)
        by_old = {t: g for t, g in ep_old[ep_old.src == src].groupby('trace', sort=False)}
        pts_new, eps_rows, cut_of = {}, [], {}
        n_cut = n_first_removed = n_not_subset = 0
        for t, (r, p) in enumerate(zip(rows.itertuples(), stream(run))):
            Ls, ds = pts[(src, t)]
            n = len(Ls)
            pts_new[(src, t)] = (Ls, ds)
            g = by_old.get(t)
            if g is None:
                continue
            first_on = int(g.iloc[0].onset_idx)
            f = gt[r.fiber_name]
            tagged = bool(f.endpoint_stop[1 if r.sign > 0 else 0])
            cut = None if tagged else sc.endpoint_crossing(np.asarray(p['path'], float).reshape(-1, 3)[:n], Ls, arcs[t]['arc'], f.tf, r.sign,
                                                           END_TOL, reach=END_TOL+END_EXTRA, start=first_on)
            if cut is None:
                for e in g.itertuples():
                    eps_rows.append(dict(src=src, trace=t, cls=e.cls, onset_idx=int(e.onset_idx), onset_len=e.onset_len, peak=e.peak, orig_cls=e.cls))
                    trans.append(dict(src=src, trace=t, onset_idx=int(e.onset_idx), orig_cls=e.cls, cls=e.cls, reason='no cut', end_d=e.end_d, cut_len=np.nan))
                continue
            n_cut += 1
            cut_of[t] = float(cut)
            nt = int((Ls <= cut).sum())
            pts_new[(src, t)] = (Ls[:nt], ds[:nt])
            new = S.episodes(Ls[:nt], ds[:nt]) if nt > 1 else []
            newd = {int(e['onset_idx']): e for e in new}
            if not set(newd) <= set(int(x) for x in g.onset_idx):
                n_not_subset += 1
            n_first_removed += int(first_on not in newd)
            for e in g.itertuples():
                ne = newd.get(int(e.onset_idx))
                if ne is None:
                    trans.append(dict(src=src, trace=t, onset_idx=int(e.onset_idx), orig_cls=e.cls, cls='removed',
                                      reason='onset after cut' if e.onset_idx >= nt else 'run cut short', end_d=e.end_d, cut_len=float(cut)))
                else:
                    eps_rows.append(dict(src=src, trace=t, cls=ne['cls'], onset_idx=int(ne['onset_idx']), onset_len=ne['onset_len'], peak=ne['peak'], orig_cls=e.cls))
                    trans.append(dict(src=src, trace=t, onset_idx=int(e.onset_idx), orig_cls=e.cls, cls=ne['cls'], reason='cut', end_d=e.end_d, cut_len=float(cut)))
        assert n_not_subset == 0, (src, n_not_subset)
        eps_df = pd.DataFrame(eps_rows)
        e, ep, tr, at_risk = scroll_tables(si, src, rows, ev, gt, cov, arcs, pts_new, rows_all[rows_all.source == src], eps_df)
        assert len(ep) == len(eps_df) and (ep.trace.to_numpy() == eps_df.trace.to_numpy()).all() and (ep.onset_idx.to_numpy() == eps_df.onset_idx.to_numpy()).all()
        assert (ep.cls.to_numpy() == eps_df.cls.to_numpy()).all()
        ep['orig_cls'] = eps_df['orig_cls'].to_numpy()
        ep['cut_len'] = ep['trace'].map(cut_of).to_numpy()
        e['src'] = np.full(len(e['trace']), si, np.int8)
        # the unchanged traces must reproduce the stage-2a exposure exactly
        m_old = old_ex['src'] == si
        old_len = np.bincount(old_ex['trace'][m_old], weights=old_ex['length'][m_old].astype(float), minlength=len(rows))
        new_len = np.bincount(e['trace'], weights=e['length'].astype(float), minlength=len(rows))
        uncut = np.array([t not in cut_of for t in range(len(rows))])
        chg = np.abs(old_len-new_len) > 1e-6
        # strict exposure on the cut region: segments with ds <= TOL before the first episode of any class
        s_seg = {k: [] for k in ('trace', 'fiber', 'pt', 'length', 'seed_d')}
        first_by_trace = {int(t): int(g.iloc[0].onset_idx) for t, g in ep.groupby('trace', sort=False)}
        for t, r in enumerate(rows.itertuples()):
            Ls, ds = pts_new[(src, t)]
            n = len(Ls)
            limit = first_by_trace.get(t, n)
            ii = np.arange(1, n)[(ds[1:] <= S.TOL) & (np.arange(1, n) < limit)] if n > 1 else np.zeros(0, int)
            if len(ii):
                q = arcs[t]
                s_seg['trace'].append(np.full(len(ii), t, np.int32)); s_seg['fiber'].append(np.full(len(ii), ev.index(r.fiber_name), np.int32))
                s_seg['pt'].append(q['pt'][ii]); s_seg['length'].append((Ls[ii]-Ls[ii-1]).astype(np.float64))
                s_seg['seed_d'].append(np.abs(q['arc'][ii]-r.t0).astype(np.float64))
        s_seg = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in s_seg.items()}
        s_seg['src'] = np.full(len(s_seg['trace']), si, np.int8)
        m_s = old_strict['src'] == si
        old_s = np.bincount(old_strict['trace'][m_s], weights=old_strict['length'][m_s], minlength=len(rows))
        new_s = np.bincount(s_seg['trace'], weights=s_seg['length'], minlength=len(rows))
        n_first_old = int(len(by_old))
        expo.append(e); ep_new.append(ep); tr_new.append(tr); strict.append(s_seg)
        diag[src] = dict(traces=len(rows), traces_with_episodes=n_first_old, traces_cut=n_cut, traces_cut_first_episode_removed=n_first_removed,
                         traces_with_first_episode_after=len(first_by_trace), at_risk_cm_old=float(old_len.sum())*CM, at_risk_cm_new=float(new_len.sum())*CM,
                         at_risk_uncut_traces_changed=int((chg & uncut).sum()), at_risk_cut_traces_increased=int(((new_len-old_len) > 1e-6).sum()),
                         strict_cm_old=float(old_s.sum())*CM, strict_cm_new=float(new_s.sum())*CM, strict_traces_changed=int((np.abs(old_s-new_s) > 1e-6).sum()),
                         strict_max_abs_change_vx=float(np.abs(old_s-new_s).max()), trace_arcs_diagnostics=dg)
        assert diag[src]['at_risk_uncut_traces_changed'] == 0, (src, diag[src])   # traces without a cut reproduce the stage-2a exposure exactly
        print(f'{src} done ({time.time()-t1:.0f}s): {diag[src]}', flush=True)
        del gt, cov, arcs
    ep_all = pd.concat(ep_new, ignore_index=True)
    ep_all.to_csv(tab2/'episodes.csv', index=False)
    ep_all.to_csv(res/'episodes_corrected.csv', index=False)
    pd.concat(tr_new, ignore_index=True).to_csv(tab2/'traces.csv', index=False)
    np.savez(tab2/'exposure.npz', sources=np.array(S.SOURCES), **{k: np.concatenate([e[k] for e in expo]) for k in expo[0]})
    np.savez(tab2/'strict_exposure.npz', sources=np.array(S.SOURCES), **{k: np.concatenate([e[k] for e in strict]) for k in strict[0]})
    for f in [*(f'covariates_{s}.npz' for s in S.SOURCES), 'sanity.json']:
        if not (tab2/f).exists():
            os.symlink(os.path.join('..', 'tables', f), tab2/f)
    tdf = pd.DataFrame(trans)
    tdf.to_csv(res/'episode_transitions_2d.csv', index=False)
    info = dict(command=' '.join(sys.argv), rule='post-hoc stage 2d: score only where a human label exists', end_tol=END_TOL, end_reach=END_TOL+END_EXTRA,
                scoring_py_sha256=hashlib.sha256(Path(sc.__file__).read_bytes()).hexdigest(), scrolls=diag, n_episodes_old=len(ep_old), n_episodes_new=len(ep_all),
                runtime_s=time.time()-t_start, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    (res/'corrected_2d.json').write_text(json.dumps(info, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print(f'{len(ep_old)} -> {len(ep_all)} episodes; runtime {info["runtime_s"]:.0f}s peak RSS {info["peak_rss_mb"]:.0f} MB')


S6_WINDOW, S6_RUN, S6_TOL = 200.0, 16.0, 3.0   # S6 landing: path length after the onset, run length (vx of path), distance to the other fiber (vx)


def acute_angle(a, b):
    """Unsigned angle in degrees (0-90) between direction vectors a and b (rows or single vectors); NaN where either has zero length."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    ok = (na > 0) & (nb > 0)
    c = np.abs(np.einsum('...j,...j->...', a, b))/np.where(ok, na*nb, 1.0)
    return np.where(ok, np.degrees(np.arccos(np.minimum(c, 1.0))), np.nan)


def s6_landing(P, L, o_len, own, tree, P_all, fid):
    """First other fiber G that the path stays within S6_TOL vx of (nearest GT point of G) over a run of consecutive vertices spanning
    >= S6_RUN vx of path length, among the vertices with path length in (o_len, o_len + S6_WINDOW]. First = earliest run start, ties by
    the smaller distance at the start. Returns None or dict(g, vertex (index into the path), point (index into P_all), dist, run_len)."""
    win = np.flatnonzero((L > o_len) & (L <= o_len+S6_WINDOW))
    if not len(win):
        return None
    nb = tree.query_ball_point(P[win], r=S6_TOL)
    lens = np.fromiter((len(x) for x in nb), int, len(nb))
    if lens.sum() == 0:
        return None
    flat = np.concatenate([np.asarray(x, int) for x in nb if len(x)])
    jj = np.repeat(np.arange(len(win)), lens)
    gg = fid[flat]
    ok = gg != own
    flat, jj, gg = flat[ok], jj[ok], gg[ok]
    if not len(flat):
        return None
    d = np.linalg.norm(P_all[flat]-P[win][jj], axis=1)
    o = np.lexsort((d, jj, gg))
    flat, jj, gg, d = flat[o], jj[o], gg[o], d[o]
    first = np.r_[True, (gg[1:] != gg[:-1]) | (jj[1:] != jj[:-1])]
    flat, jj, gg, d = flat[first], jj[first], gg[first], d[first]   # one row per (fiber, vertex): that fiber's nearest point
    new_run = np.r_[True, (gg[1:] != gg[:-1]) | (jj[1:] != jj[:-1]+1)]
    starts = np.flatnonzero(new_run)
    ends = np.r_[starts[1:], len(jj)]-1
    span = L[win[jj[ends]]]-L[win[jj[starts]]]
    good = np.flatnonzero(span >= S6_RUN)
    if not len(good):
        return None
    b = good[np.lexsort((d[starts[good]], jj[starts[good]]))[0]]
    s_ = starts[b]
    return dict(g=int(gg[s_]), vertex=int(win[jj[s_]]), point=int(flat[s_]), dist=float(d[s_]), run_len=float(span[b]))


def s6_main(data, fibers, villa, out):
    """Stage 2e, S6 (pre-registered, now on the corrected hard switches; the gap and its rank are post-hoc): what a real switch looks like.

    For each corrected hard switch (tables_2d/episodes.csv) the FULL saved path is used. Other fibers = every loaded fiber of the scroll except
    the trace's own. Also saves the trace with a removed hard switch (onset after the cut) with the longest path after the cut, for the figure."""
    t_start = time.time()
    out = Path(out); tab2 = out/'tables_2d'; res = out/'results'
    init_villa(villa)
    ep = pd.read_csv(tab2/'episodes.csv', float_precision='round_trip')
    ep['order'] = ep.groupby(['src', 'trace']).cumcount()
    hs = ep[ep.cls == 'hard_switch'].reset_index(drop=True)
    tr2 = pd.read_csv(res/'episode_transitions_2d.csv', float_precision='round_trip')
    rem = tr2[(tr2.orig_cls == 'hard_switch') & (tr2.cls == 'removed') & (tr2.reason == 'onset after cut')]
    code_dir = Path(__file__).resolve().parent.parent
    out_rows, best = [], None
    for si, src in enumerate(S.SOURCES):
        t1 = time.time()
        run = run_dir(data, src)
        cfg = json.loads((run/'config.json').read_text())
        gt, _ = load_gt(Path(fibers)/Path(cfg['fibers_dir']).name, float(cfg['grid_scale']))
        names = list(gt)
        fid_of = {n: i for i, n in enumerate(names)}
        sizes = np.array([len(gt[n].points) for n in names])
        fstart = np.r_[0, np.cumsum(sizes)[:-1]]
        P_all = np.concatenate([gt[n].points for n in names])
        fid = np.repeat(np.arange(len(names)), sizes)
        dirs_all = np.concatenate([fiber_dir(gt[n]) for n in names])
        tag = np.array([gt[n].tag for n in names])
        tree = cKDTree(P_all)
        dep = pd.read_csv(code_dir/'runs'/'v2'/src/'analysis'/'departures.csv', float_precision='round_trip').set_index('trace')
        mine = hs[hs.src == src]
        by_trace = {t: g for t, g in mine.groupby('trace')}
        rem_s = {int(r.trace): float(r.cut_len) for r in rem[rem.src == src].itertuples()}
        for t, p in enumerate(stream(run)):
            if t not in by_trace and t not in rem_s:
                continue
            P = np.asarray(p['path'], float).reshape(-1, 3)
            L = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            if t in rem_s and (best is None or L[-1]-rem_s[t] > best['after_cut_len']):
                f_own = gt[p['fiber_name']]
                best = dict(src=src, trace=t, fiber_name=p['fiber_name'], after_cut_len=float(L[-1]-rem_s[t]), cut_len=rem_s[t], path_len=float(L[-1]),
                            path=P, lengths=L, stored_ds=np.asarray(p['gt_distance'], float), gt_points=f_own.points, gt_s=f_own.s, sign=float(p['sign']),
                            t0=float(p['t0']), endpoint_stop=np.array(f_own.endpoint_stop, bool),
                            onset_idx=int(tr2[(tr2.src == src) & (tr2.trace == t) & (tr2.orig_cls == 'hard_switch')].onset_idx.iloc[0]))
            for e in (by_trace[t].itertuples() if t in by_trace else []):
                own = fid_of[e.fiber_name]
                depxyz = gt[e.fiber_name].points[int(e.gt_pt)]
                owndir = np.array([e.dir_x, e.dir_y, e.dir_z], float)
                row = dict(src=src, trace=t, fiber_name=e.fiber_name, onset_idx=int(e.onset_idx), onset_len=float(e.onset_len), is_first=bool(e.order == 0),
                           family=e.family, end_d=float(e.end_d), seed_d=float(e.seed_d), path_len=float(L[-1]), orig_cls=e.orig_cls, cut_len=float(e.cut_len) if np.isfinite(e.cut_len) else np.nan)
                land = s6_landing(P, L, e.onset_len, own, tree, P_all, fid)
                row.update(landed=land is not None, G=np.nan, G_family=np.nan, same_family=np.nan, landing_angle_deg=np.nan, landing_crossing=np.nan,
                           landing_len=np.nan, landing_run_len=np.nan, gap_vx=np.nan, gap_rank=np.nan)
                dall = np.linalg.norm(P_all-depxyz, axis=1)
                mins = np.minimum.reduceat(dall, fstart)
                others = np.arange(len(names)) != own
                row.update(nearest_other_dist=float(mins[others].min()), nearest_other=names[int(np.flatnonzero(others)[np.argmin(mins[others])])])
                if land is not None:
                    g = land['g']
                    ang = float(acute_angle(dirs_all[land['point']], owndir))
                    row.update(G=names[g], G_family=tag[g], same_family=bool(tag[g] == e.family), landing_angle_deg=ang, landing_crossing=bool(ang >= 45),
                               landing_len=float(L[land['vertex']]-e.onset_len), landing_run_len=land['run_len'], gap_vx=float(mins[g]),
                               gap_rank=int(1+(mins[others] < mins[g]).sum()))
                a, b = e.onset_len+16.0, e.onset_len+48.0
                if L[-1] >= b:
                    ca = np.array([np.interp(a, L, P[:, k]) for k in range(3)]); cb = np.array([np.interp(b, L, P[:, k]) for k in range(3)])
                    row['turn_angle_deg'] = float(acute_angle(cb-ca, owndir))
                else:
                    row['turn_angle_deg'] = np.nan
                row['turn_bin'] = np.nan if np.isnan(row['turn_angle_deg']) else ('0-15' if row['turn_angle_deg'] < 15 else '15-45' if row['turn_angle_deg'] < 45 else '45-90')
                row.update(direction=np.nan, departure_len=np.nan, departure_match=np.nan)
                if row['is_first'] and t in dep.index:
                    row.update(direction=dep.loc[t, 'direction'], departure_len=float(dep.loc[t, 'departure_len']),
                               departure_match=bool(abs(float(dep.loc[t, 'departure_len'])-e.onset_len) < 1.0))
                out_rows.append(row)
        print(f'{src} done ({time.time()-t1:.0f}s): {len(by_trace)} switches', flush=True)
        del P_all, dirs_all, tree, gt
    df = pd.DataFrame(out_rows)
    assert len(df) == len(hs), (len(df), len(hs))
    df.to_csv(res/'s6_switches.csv', index=False)
    np.savez(res/'past_end_example_data.npz', **{k: (np.array(v) if not isinstance(v, str) else np.array(v)) for k, v in best.items()})
    print(f'{len(df)} switches; example {best["src"]} trace {best["trace"]} after-cut length {best["after_cut_len"]:.0f} vx; runtime {time.time()-t_start:.0f}s '
          f'peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB')


if __name__ == '__main__':
    flags = {a for a in sys.argv[1:] if a.startswith('--')}
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(args) != 4 or not flags or flags-{'--checks', '--tables', '--flags', '--crossdiag', '--corrected', '--s6'}:
        sys.exit(__doc__)
    if '--checks' in flags:
        checks_main(*args)
    if '--tables' in flags:
        tables_main(*args)
    if '--flags' in flags:
        flags_main(*args)
    if '--crossdiag' in flags:
        crossdiag_main(*args)
    if '--corrected' in flags:
        corrected_main(*args)
    if '--s6' in flags:
        s6_main(*args)
