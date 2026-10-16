"""Forward/backward consistency checks on finished forward traces (no ground truth used to decide).

full:      from each forward trace's end, trace back; consistent if the back path returns within --tol of the seed.
segmented: every --every voxels of forward arclength, trace back --back voxels from that point; the check passes if
           the back path stays within --tol of the forward path. A trace is cut at its first failing check.
Scoring then uses the forward run's ground-truth labels (correct / offtrack lengths) to measure what each rule keeps.
Writes fwdback_full.csv, fwdback_segments.csv and fwdback_summary.json into the forward run directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from vesuvius.neural_tracing.tracers.fibers.frames.learned import trace_family_kwargs
from vesuvius.neural_tracing.tracers.fibers.tracing.runtime import configure_volume, resolve_checkpoint, use_gpu_crops
from vesuvius.neural_tracing.tracers.fibers.tracing.trace import FiberTracer, TraceParams
from vesuvius.neural_tracing.tracers.models.checkpoint import load_checkpoint
from vesuvius.neural_tracing.tracers.shared.crop_sampling.normalization import prepare_normalization
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolume
from vesuvius.neural_tracing.tracers.shared.policy import checkpoint_policy


def arclength(p):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def back_heading(p, i, k=8):
    """Unit direction pointing backwards along the forward path at index i."""
    j = max(0, i - k)
    d = p[j] - p[i]
    n = np.linalg.norm(d)
    return d / n if n > 0 else None


def run_batch(tracer, seeds, headings, families, batch):
    paths = []
    for b in range(0, len(seeds), batch):
        kw = trace_family_kwargs(tracer, families[b:b+batch])
        ps, _ = tracer.trace(np.stack(seeds[b:b+batch]), np.stack(headings[b:b+batch]), **kw)
        paths += [np.asarray(x) for x in ps]
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True, help='forward run directory from heldout_eval.py')
    ap.add_argument('--checkpoint', default='scrollprize/fiber_single_small')
    ap.add_argument('--tol', type=float, default=3.0, help='trace-grid voxels, as the team scorer')
    ap.add_argument('--every', type=float, default=128.0)
    ap.add_argument('--back', type=float, default=192.0)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--gpu-chunks', type=int, default=1024)
    ap.add_argument('--cache-dir', default='/workspace/cache')
    args = ap.parse_args()
    run = Path(args.run)
    cfg = json.loads((run/'config.json').read_text())
    rows = pd.read_csv(run/'rows.csv')
    fwd = [json.loads(line) for line in open(run/'paths.jsonl')]

    checkpoint = resolve_checkpoint(args.checkpoint)
    model, crop, n_hist, spec, ck = load_checkpoint(checkpoint, 'cuda')
    configure_volume(SimpleNamespace(ct=cfg['ct'], cache_dir=args.cache_dir), spec, model.cfg, checkpoint)
    prepare_normalization(run/'fwdback_norm', [spec], known=ck['ct_normalization'])
    policy = checkpoint_policy(ck, model.cfg, confidence=cfg['operating_policy']['confidence'])
    vol = CTVolume(spec, cache_bytes=16 << 30)

    # Full round trip.
    seeds, heads, fams, owners = [], [], [], []
    for t, (r, f) in enumerate(zip(rows.itertuples(), fwd)):
        p = np.asarray(f['path'], float)
        h = back_heading(p, len(p)-1) if len(p) > 8 else None
        if h is not None:
            seeds.append(p[-1]); heads.append(h); fams.append(r.family); owners.append(t)
    tracer = FiberTracer(model, vol, crop, n_hist, TraceParams.from_policy(policy, max_len=float(cfg['max_len'])+64), device='cuda')
    use_gpu_crops(tracer, vol, 'cuda', args.gpu_chunks)
    try:
        backs = run_batch(tracer, seeds, heads, fams, args.batch)
    finally:
        tracer.close()
    full = []
    for t, b in zip(owners, backs):
        p = np.asarray(fwd[t]['path'], float)
        d_seed = float(np.min(np.linalg.norm(b - p[0], axis=1))) if len(b) else np.inf
        dist_fwd = cKDTree(b).query(p)[0] if len(b) else np.full(len(p), np.inf)
        full.append(dict(trace=t, returned=d_seed <= args.tol, back_len=float(arclength(b)[-1]) if len(b) > 1 else 0.0,
                         seed_miss=d_seed, frac_fwd_covered=float(np.mean(dist_fwd <= args.tol))))
    full = pd.DataFrame(full)
    full.to_csv(run/'fwdback_full.csv', index=False)

    # Segmented checks.
    seeds, heads, fams, meta = [], [], [], []
    for t, (r, f) in enumerate(zip(rows.itertuples(), fwd)):
        p = np.asarray(f['path'], float)
        s = arclength(p)
        for L in np.arange(args.every, s[-1] + 1e-6, args.every):
            i = int(np.searchsorted(s, L))
            h = back_heading(p, min(i, len(p)-1))
            if h is not None:
                seeds.append(p[min(i, len(p)-1)]); heads.append(h); fams.append(r.family)
                meta.append(dict(trace=t, at_len=float(s[min(i, len(p)-1)]), at_idx=min(i, len(p)-1)))
    tracer = FiberTracer(model, vol, crop, n_hist, TraceParams.from_policy(policy, max_len=args.back), device='cuda')
    use_gpu_crops(tracer, vol, 'cuda', args.gpu_chunks)
    try:
        backs = run_batch(tracer, seeds, heads, fams, args.batch)
    finally:
        tracer.close()
    segs, back_paths = [], []
    for m, b in zip(meta, backs):
        p = np.asarray(fwd[m['trace']]['path'], float)
        s = arclength(p)
        window = p[(s <= m['at_len']) & (s >= m['at_len'] - args.back)]
        if len(b) < 2:
            segs.append(dict(m, ok=True, back_len=0.0, compared_len=0.0, max_dev=np.nan, frac_on=np.nan))
            back_paths.append(np.asarray(b))
            continue
        bl = arclength(b)
        # Only the stretch that has a forward path to compare against: never past the forward seed.
        span = min(args.back, m['at_len'] - 4.0)
        dev = cKDTree(window).query(b[bl <= span])[0]
        segs.append(dict(m, back_len=float(bl[-1]), compared_len=float(min(bl[-1], span)), max_dev=float(dev.max()),
                         frac_on=float(np.mean(dev <= args.tol)),
                         ok=bool(np.mean(dev <= args.tol) >= 0.9 and bl[-1] >= span * 0.5)))
        back_paths.append(np.round(b, 2))
    segs = pd.DataFrame(segs)
    segs.to_csv(run/'fwdback_segments.csv', index=False)
    np.savez_compressed(run/'fwdback_segment_paths.npz', **{f's{i}': p for i, p in enumerate(back_paths)})

    # Score both rules against the forward run's ground truth.
    correct, wrong = rows['correct'].to_numpy(float), rows['offtrack'].to_numpy(float)
    ret = np.zeros(len(rows), bool)
    ret[full['trace'].to_numpy()] = full['returned'].to_numpy()
    cut = np.full(len(rows), np.inf)
    for t, g in segs.groupby('trace'):
        bad = g[~g['ok']]
        if len(bad):
            cut[t] = max(0.0, float(bad['at_len'].min()) - args.every)  # keep only up to the last passing check
    seg_c = np.minimum(cut, correct).sum()
    seg_w = np.maximum(0, np.minimum(cut, correct + wrong) - correct).sum()
    summary = dict(source=cfg['name'], confidence=cfg['operating_policy']['confidence'], traces=len(rows),
                   baseline_correct=float(correct.sum()), baseline_wrong=float(wrong.sum()),
                   full_returned=int(ret.sum()), full_correct_kept=float(correct[ret].sum()), full_wrong_kept=float(wrong[ret].sum()),
                   seg_checks=len(segs), seg_failed=int((~segs['ok']).sum()) if len(segs) else 0,
                   seg_correct_kept=float(seg_c), seg_wrong_kept=float(seg_w))
    (run/'fwdback_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
