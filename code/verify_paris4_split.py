"""Check the Paris4 control fibers against the checkpoint's own training split (CPU, no torch, no villa import).

Reads the metadata of last.pt with a restricted unpickler (tensors are skipped, nothing else is executed), recomputes
villa's fiber_hash order (shared/annotations/splits.py::heldout_ids: sha256("7349:<name>"), fraction 0.1) over the
checkpoint's fiber_manifest, and reports which of the 62 control fibers are held out for EVERY legacy reserved set
consistent with dataset_provenance (563 of the manifest's fibers were trained on).

  python code/verify_paris4_split.py last.pt eval_fibers_names.txt rows.csv > results/verification/paris4_split_check.json
"""
import codecs
import collections
import hashlib
import json
import pickle
import sys
import zipfile

import numpy as np
import pandas as pd


def checkpoint_metadata(path):
    import numpy._core.multiarray as ma
    allowed = {('collections', 'OrderedDict'): collections.OrderedDict, ('numpy._core.multiarray', '_reconstruct'): ma._reconstruct,
               ('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype, ('numpy._core.multiarray', 'scalar'): ma.scalar,
               ('_codecs', 'encode'): codecs.encode}

    class Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) in allowed:
                return allowed[(module, name)]
            if module.startswith('numpy.dtypes'):
                import numpy.dtypes
                return getattr(numpy.dtypes, name)
            if module.startswith('torch'):
                return lambda *a, **k: None  # tensor rebuilds: weights are not needed
            raise pickle.UnpicklingError(f'blocked {module}.{name}')

        def persistent_load(self, pid):
            return None
    z = zipfile.ZipFile(path)
    return Restricted(z.open(next(n for n in z.namelist() if n.endswith('data.pkl')))).load()


def main():
    ck = checkpoint_metadata(sys.argv[1])
    eval_names = set(open(sys.argv[2]).read().split())
    used = set(pd.read_csv(sys.argv[3]).fiber_name)
    manifest = [f['name'] for f in ck['fiber_manifest']]
    paris4 = next(s for s in ck['dataset_config']['sources'] if s['kind'] == 'paris4')
    trained = next(s['fibers'] for s in ck['dataset_provenance'] if s['kind'] == 'paris4')
    seed, frac = int(paris4['validation']['seed']), float(paris4['validation']['fraction'])
    key = lambda k: hashlib.sha256(f'{seed}:{k}'.encode()).digest()
    order = sorted(manifest, key=key)
    ours = sorted(eval_names, key=key)[:62]
    assert set(ours) == used, 'control set differs from the traced fibers'
    # reserved R + hash-held-out round(frac*(N-R)) + fold-back exclusions F = N - trained, F >= 0
    n = len(manifest)
    r_max = max(r for r in range(n) if r + round(frac*(n - r)) <= n - trained)
    min_count = round(frac*(n - r_max))
    rank = {k: order.index(k) for k in ours}
    print(json.dumps(dict(manifest_fibers=n, trained=trained, seed=seed, fraction=frac, rmax=r_max, min_count=min_count,
                          guaranteed=sorted(k for k in ours if rank[k] < min_count),
                          at_risk=sorted(k for k in ours if rank[k] >= min_count), ranks=rank), indent=1))


if __name__ == '__main__':
    main()
