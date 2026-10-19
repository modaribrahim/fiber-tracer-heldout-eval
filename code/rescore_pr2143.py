"""Score saved traces with villa's score_trace before and after PR #2143. Usage: rescore_pr2143.py SRC GT RUN OUT
SRC: PR #2143 branch vesuvius/src. Needs scoring_before.py (villa main scoring.py before the PR) next to this file."""
import sys, types, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
src, gt, run, out = map(Path, sys.argv[1:5])
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
        from vesuvius.neural_tracing.tracers.fibers.evaluation import scoring as after; break
    except ModuleNotFoundError as e:
        if e.name.split('.')[0] in ('vesuvius', 'vc3d_fiber_format'): raise
        print('stub', e.name); sys.modules[e.name] = _Stub(e.name)
from vesuvius.neural_tracing.tracers.shared.annotations.vc3d import load_fibers
spec = importlib.util.spec_from_file_location('scoring_before', Path(__file__).with_name('scoring_before.py'))
before = importlib.util.module_from_spec(spec); spec.loader.exec_module(before)
cfg = json.loads((run/'config.json').read_text())
fibers = {Path(f.name).name: f for f in load_fibers(str(gt), grid_scale=cfg['grid_scale'])}
rows = pd.read_csv(run/'rows.csv')
res = []
for k, line in enumerate(open(run/'paths.jsonl')):
    p = json.loads(line); r = rows.iloc[k]; assert r.fiber_name == p['fiber_name']
    f = fibers[p['fiber_name']]; path = np.asarray(p['path'], np.float64)
    b = before.score_trace(path, f, p['t0'], p['sign']); a = after.score_trace(path, f, p['t0'], p['sign'])
    res.append(dict(k=k, fiber=p['fiber_name'], sign=p['sign'], diverged=b['diverged'], endpoint_known=b['endpoint_known'],
                    saved_correct=r.correct, saved_offtrack=r.offtrack,
                    b_correct=b['correct'], b_wrong=b['offtrack'], b_unknown=b['unknown'],
                    a_correct=a['correct'], a_wrong=a['offtrack'], a_unknown=a['unknown'], length=b['length']))
df = pd.DataFrame(res); df.to_csv(out, index=False)
print('traces', len(df), '| before matches saved rows.csv:',
      bool(np.allclose(df.b_correct, df.saved_correct) and np.allclose(df.b_wrong, df.saved_offtrack)))
print('correct length unchanged:', bool(np.allclose(df.a_correct, df.b_correct)))
ch = df[~np.isclose(df.a_wrong, df.b_wrong)]
print('traces changed:', len(ch), 'of', int(df.diverged.sum()), 'diverged; all diverged & untagged end:',
      bool(ch.diverged.all() and (~ch.endpoint_known).all()))
for t in ('b', 'a'):
    c, w = df[f'{t}_correct'].sum(), df[f'{t}_wrong'].sum()
    print({'b': 'before', 'a': 'after '}[t], f'strict precision {c/(c+w):.3f}  wrong {w:.0f} vx  unknown {df[f"{t}_unknown"].sum():.0f} vx')
