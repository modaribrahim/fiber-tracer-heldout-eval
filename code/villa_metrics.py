"""villa main metrics (geometric_outcomes at 1.5 and 3 vx, also limited to the path that has GT under PR #2143) on saved traces.
Usage: python -I villa_metrics.py MAIN_SRC PATCHED_SRC FIBERS RUN OUT_CSV  (MAIN_SRC: villa main 8aa507c vesuvius/src; PATCHED_SRC: PR #2143 branch)"""
import sys, types, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
src, patched_src, gt, run, out = map(Path, sys.argv[1:6])
sys.path.insert(0, str(src))
for name, sub in [('vesuvius', 'vesuvius'), ('vesuvius.neural_tracing', 'vesuvius/neural_tracing')]:
    m = types.ModuleType(name); m.__path__ = [str(src/sub)]; sys.modules[name] = m  # skip heavy package __init__
class _Stub(types.ModuleType):  # CT-reading deps imported by scoring's imports but unused when scoring
    __path__ = []
    def __getattr__(self, k): return _Stub(k) if not k.startswith('__') else super().__getattribute__(k)
    def __call__(self, *a, **k): return self
    def __mro_entries__(self, bases): return (object,)
while True:
    try:
        from vesuvius.neural_tracing.tracers.fibers.evaluation import scoring as main; break
    except ModuleNotFoundError as e:
        if e.name.split('.')[0] in ('vesuvius', 'vc3d_fiber_format'): raise
        print('stub', e.name); sys.modules[e.name] = _Stub(e.name)
from vesuvius.neural_tracing.tracers.shared.annotations.vc3d import load_fibers
pfile = patched_src/'vesuvius/neural_tracing/tracers/fibers/evaluation/scoring.py'
while True:
    try:
        spec = importlib.util.spec_from_file_location('scoring_patched', pfile)
        patched = importlib.util.module_from_spec(spec); sys.modules['scoring_patched'] = patched
        spec.loader.exec_module(patched); break
    except ModuleNotFoundError as e:
        if e.name.split('.')[0] in ('vesuvius', 'vc3d_fiber_format'): raise
        print('stub', e.name); sys.modules[e.name] = _Stub(e.name)
print('main scoring:', main.__file__, '| patched scoring:', patched.__file__, flush=True)
assert hasattr(main, 'geometric_outcomes') and not hasattr(main, 'endpoint_crossing')
cfg = json.loads((run/'config.json').read_text())
fibers = {Path(f.name).name: f for f in load_fibers(str(gt), grid_scale=cfg['grid_scale'])}
rows = pd.read_csv(run/'rows.csv')
KEEP = ('length', 'avail', 'followed', 'coverage', 'correct', 'offtrack', 'unknown', 'diverged', 'reached_end',
        'endpoint_known', 'err', 'precision', 'endpoint_overrun')
res = []
for k, line in enumerate(open(run/'paths.jsonl')):
    p = json.loads(line); r = rows.iloc[k]; assert r.fiber_name == p['fiber_name']
    f = fibers[p['fiber_name']]; path = np.asarray(p['path'], np.float64); t0, sign = p['t0'], p['sign']
    m = main.score_trace(path, f, t0, sign)
    row = dict(k=k, fiber_name=p['fiber_name'], sign=sign, **{key: m[key] for key in KEEP})
    pp = patched.score_trace(path, f, t0, sign)
    row.update(p_correct=pp['correct'], p_offtrack=pp['offtrack'], p_unknown=pp['unknown'])
    for tol, suf in ((1.5, '_1p5'), (3.0, '_3')):
        g = main.geometric_outcomes(path, f, t0, sign, tolerance=tol)
        row.update({key+suf: v for key, v in g.items()})
    arc, dist = main.distance_profile(path, f, t0, sign)
    row['finite_len'] = float(np.diff(arc, prepend=0.)[np.isfinite(dist)].sum())
    seg = np.diff(arc, prepend=0.); fin = np.isfinite(dist)
    kl = row['p_correct'] + row['p_offtrack']; row['known_len'] = kl
    for tol, suf in ((1.5, '1p5'), (3.0, '3')):
        w = fin & (dist <= tol)
        full = float(seg[w].sum())
        assert abs(full - row['geometric_agreement_'+suf]) < 1e-6, (k, tol, full, row['geometric_agreement_'+suf])
        row['within_known_'+suf] = float(seg[w & (arc <= kl)].sum())
    res.append(row)
    if k % 50 == 0: print('trace', k, flush=True)
pd.DataFrame(res).to_csv(out, index=False)
print('done', len(res), flush=True)
