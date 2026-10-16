"""Held-out evaluation of a released fiber follower on human-annotated fibers of any scroll.

Uses villa's own seed builder (validation_manifest), tracer (FiberTracer) and scoring
(evaluation.scoring.evaluate / trace_events), so numbers are comparable with the team's
protocol. Writes, per run directory:
  config.json     exact inputs: checkpoint revision, CT, policy, seeds digest, villa commit
  manifest.json   the frozen seeds (monitor / calibration / final)
  rows.csv        one row per directed seed with the team's metrics plus departure location
  paths.jsonl     every traced path (trace-grid coordinates) for later analysis
  summary.json    team summary per split and overall
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vesuvius.neural_tracing.tracers.fibers.data.datasets import fiber_manifest, validation_manifest
from vesuvius.neural_tracing.tracers.fibers.data.seeds import make_seeds
from vesuvius.neural_tracing.tracers.fibers.evaluation.scoring import evaluate, summarize_outcomes, trace_events
from vesuvius.neural_tracing.tracers.fibers.tracing.runtime import configure_volume, resolve_checkpoint, use_gpu_crops
from vesuvius.neural_tracing.tracers.fibers.tracing.trace import FiberTracer, TraceParams
from vesuvius.neural_tracing.tracers.models.checkpoint import load_checkpoint
from vesuvius.neural_tracing.tracers.shared.annotations.vc3d import load_fibers
from vesuvius.neural_tracing.tracers.shared.crop_sampling.normalization import prepare_normalization
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolume
from vesuvius.neural_tracing.tracers.shared.io import jsonable
from vesuvius.neural_tracing.tracers.shared.policy import checkpoint_policy

SPLITS = ('monitor', 'calibration', 'final')
ROW_KEYS = ('followed', 'avail', 'coverage', 'correct', 'offtrack', 'unknown', 'endpoint_overrun', 'scored_length',
            'length', 'precision', 'verified_fraction', 'reached_end', 'crossed_endpoint', 'endpoint_known',
            'diverged', 'err', 'reason', 't0', 'sign')


class DecisionLogger:
    """on_decision observer: records every decision's inputs and proposals, never alters the trace.

    Per decision: trace key, decision index, arclength travelled, head position and crop frame,
    all predicted points (world, trace grid), the cumulative-min confidence curve and the commit count.
    """
    def __init__(self):
        self.records, self.diag_keys = [], None

    def start_batch(self, fibers, chunk):
        self.chunk, self.counter = chunk, {}

    def __call__(self, i, state):
        seed = self.chunk[i]
        key = (int(seed['fiber']), float(seed['t']), float(seed['sign']))
        k = self.counter.get(i, 0)
        self.counter[i] = k + 1
        world = state['pos'] + np.asarray(state['points']) @ np.asarray(state['frame']).T
        diag = state.get('ct_frame_diagnostics')
        if self.diag_keys is None and isinstance(diag, dict):
            self.diag_keys = sorted(k2 for k2, v in diag.items() if np.ndim(v) == 0 and isinstance(v, (int, float, np.number, bool)))
        self.records.append(dict(
            key=key, decision=k, travelled=float(state['travelled']), pos=np.asarray(state['pos'], np.float32),
            frame=np.asarray(state['frame'], np.float32), world=np.asarray(world, np.float32),
            confidence=np.asarray(state['confidence'], np.float32), n_commit=int(state['n_commit']),
            would_stop=bool(state['would_stop']), recovery_blocked=bool(state['recovery_blocked']),
            diag=[float(diag[k2]) for k2 in self.diag_keys] if isinstance(diag, dict) and self.diag_keys else []))
        return None

    def save(self, path, keymap):
        if not self.records:
            return
        r = self.records
        np.savez_compressed(
            path, trace=np.array([keymap[x['key']] for x in r], np.int32), decision=np.array([x['decision'] for x in r], np.int32),
            travelled=np.array([x['travelled'] for x in r], np.float32), pos=np.stack([x['pos'] for x in r]),
            frame=np.stack([x['frame'] for x in r]), world=np.stack([x['world'] for x in r]),
            confidence=np.stack([x['confidence'] for x in r]), n_commit=np.array([x['n_commit'] for x in r], np.int16),
            would_stop=np.array([x['would_stop'] for x in r]), recovery_blocked=np.array([x['recovery_blocked'] for x in r]),
            diag=np.array([x['diag'] for x in r], np.float32), diag_keys=np.array(self.diag_keys or [], dtype=str))


def annotation_meta(fiber_dir):
    meta = {}
    for f in Path(fiber_dir).glob('*.json'):
        j = json.loads(f.read_text())
        meta[f.name] = dict(username=j.get('username'), version=j.get('version'),
                            reviewed='reviewed' in (j.get('tags') or []),
                            hv_tag=(j.get('hv_classification') or {}).get('automatic_tag'))
    return meta


def villa_commit():
    try:
        return subprocess.check_output(['git', '-C', '/workspace/villa', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name', required=True, help='source name, e.g. PHerc1451')
    ap.add_argument('--fibers', required=True, help='directory of VC3D fiber JSONs')
    ap.add_argument('--ct', required=True, help='CT OME-Zarr URL the fibers are annotated on')
    ap.add_argument('--out', required=True)
    ap.add_argument('--checkpoint', default='scrollprize/fiber_single_small')
    ap.add_argument('--confidence', type=float, default=0.5)
    ap.add_argument('--n-commit', type=int)
    ap.add_argument('--gate', choices=('full', 'prefix'))
    ap.add_argument('--max-len', type=float, default=2000.0, help='per-direction limit, trace-grid voxels')
    ap.add_argument('--per-split', type=int, default=40, help='fibers seeded per split (one position, both directions)')
    ap.add_argument('--splits', nargs='+', default=list(SPLITS), choices=SPLITS)
    ap.add_argument('--manifest', help='reuse a frozen manifest.json instead of building one')
    ap.add_argument('--unseen-only', action='store_true', help="drop fibers listed in the checkpoint's training manifest")
    ap.add_argument('--heldout-hash', type=int,
                    help="keep only the first N fibers in the trainer's fiber_hash order (seed 7349): provably held out "
                         "of training whatever its legacy reserved set was")
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--gpu-chunks', type=int, default=4096)
    ap.add_argument('--cache-dir', default='/workspace/cache')
    ap.add_argument('--limit', type=int, help='trace only the first N directed seeds per split (smoke tests)')
    ap.add_argument('--manifest-only', action='store_true', help='freeze manifest.json and config.json, then exit')
    ap.add_argument('--per-fiber', type=int,
                    help='seed EVERY fiber at this many annotated positions (both directions), all in the final split, '
                         'instead of the capped validation_manifest')
    ap.add_argument('--log-decisions', action='store_true', help='write decisions_<split>.npz (per-decision proposals)')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    checkpoint = resolve_checkpoint(args.checkpoint)
    model, crop, n_hist, spec, ck = load_checkpoint(checkpoint, 'cuda')
    configure_volume(SimpleNamespace(ct=args.ct, cache_dir=args.cache_dir), spec, model.cfg, checkpoint)
    prepare_normalization(out, [spec], known=ck['ct_normalization'])

    fibers = load_fibers(args.fibers, grid_scale=spec.grid_scale)
    trained = {f['name'] for f in ck.get('fiber_manifest', [])}
    if args.unseen_only:
        fibers = [f for f in fibers if f.name not in trained]
    if args.heldout_hash:
        order = sorted((f.name for f in fibers), key=lambda key: hashlib.sha256(f'7349:{key}'.encode()).digest())
        keep = set(order[:args.heldout_hash])
        fibers = [f for f in fibers if f.name in keep]
    overlap = sum(f.name in trained for f in fibers)
    meta = annotation_meta(args.fibers)
    print(f'{args.name}: {len(fibers)} fibers ({overlap} also in the training manifest)', flush=True)

    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
    elif args.per_fiber:
        seed_volume = CTVolume(spec, cache_bytes=256 << 20)
        seeds = []
        for fi, f in enumerate(fibers):
            seeds += [dict(s, fiber=fi) for s in make_seeds([f], seed_volume, per_fiber=args.per_fiber, seed=7349+fi)]
        manifest = dict(version=3, fibers=fiber_manifest(fibers), volume=spec.to_dict(), per_fiber=args.per_fiber,
                        monitor=[], calibration=[], final=seeds, final_fibers=list(range(len(fibers))))
        manifest = json.loads(json.dumps(manifest, default=jsonable))
        manifest['sha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        (out/'manifest.json').write_text(json.dumps(manifest, default=jsonable))
    else:
        manifest = validation_manifest(fibers, spec, seed=7349, monitor_count=args.per_split)
        (out/'manifest.json').write_text(json.dumps(manifest, default=jsonable))
    for split in SPLITS:
        for seed in manifest[split]:
            seed['pos'], seed['heading'] = np.asarray(seed['pos'], float), np.asarray(seed['heading'], float)

    policy = checkpoint_policy(ck, model.cfg, confidence=args.confidence, n_commit=args.n_commit, gate=args.gate)
    config = dict(name=args.name, fibers_dir=args.fibers, n_fibers=len(fibers), training_overlap=overlap,
                  ct=spec.ct_zarr, ct_level=spec.ct_level, grid_scale=spec.grid_scale, checkpoint=str(checkpoint),
                  checkpoint_step=ck.get('step'), operating_policy=policy.to_dict(), max_len=args.max_len,
                  splits=args.splits, per_split=args.per_split, per_fiber=args.per_fiber, manifest_sha256=manifest.get('sha256'),
                  villa_commit=villa_commit(), unseen_only=args.unseen_only, limit=args.limit)
    (out/'config.json').write_text(json.dumps(config, indent=1, default=jsonable))
    if args.manifest_only:
        print(json.dumps({s: len(manifest[s]) for s in SPLITS} | {'seconds': round(time.time()-started, 1)}), flush=True)
        return

    vol = CTVolume(spec, cache_bytes=16 << 30)
    tracer = FiberTracer(model, vol, crop, n_hist, TraceParams.from_policy(policy, max_len=args.max_len), device='cuda')
    use_gpu_crops(tracer, vol, 'cuda', args.gpu_chunks)

    rows_out, paths_out = [], []
    try:
        for split in args.splits:
            seeds = manifest[split][:args.limit] if args.limit else manifest[split]
            if not seeds:
                continue
            t_split = time.time()
            traced = {}
            logger = DecisionLogger() if args.log_decisions else None
            rows, _ = evaluate(tracer, fibers, seeds, batch=args.batch, history_audit=logger,
                               on_trace=lambda s, p, r: traced.__setitem__((s['fiber'], s['t'], s['sign']), np.asarray(p)))
            if logger is not None:
                keymap = {(int(r['fiber']), float(r['t0']), float(r['sign'])): len(paths_out)+n for n, r in enumerate(rows)}
                logger.save(out/f'decisions_{split}.npz', keymap)
            for row in rows:
                path = traced[(row['fiber'], row['t0'], row['sign'])]
                f = fibers[row['fiber']]
                lengths, d, arc, end, crossing = trace_events(path, f, row['t0'], row['sign'])
                dep = path[end] if row['diverged'] and end < len(path) else None
                m = meta.get(f.name, {})
                rows_out.append(dict(
                    source=args.name, split=split, fiber_name=f.name, family=f.tag, annotator=m.get('username'),
                    reviewed=m.get('reviewed'), fiber_length=float(f.length), in_training=f.name in trained,
                    seed_x=float(path[0][0]*spec.grid_scale), seed_y=float(path[0][1]*spec.grid_scale),
                    seed_z=float(path[0][2]*spec.grid_scale),
                    departure_arclength=float(lengths[end]) if dep is not None else None,
                    dep_x=float(dep[0]*spec.grid_scale) if dep is not None else None,
                    dep_y=float(dep[1]*spec.grid_scale) if dep is not None else None,
                    dep_z=float(dep[2]*spec.grid_scale) if dep is not None else None,
                    **{k: (float(row[k]) if isinstance(row[k], (int, float, np.floating, np.bool_)) and not isinstance(row[k], bool) else row[k]) for k in ROW_KEYS}))
                paths_out.append(dict(source=args.name, split=split, fiber_name=f.name, t0=row['t0'], sign=row['sign'],
                                      reason=row['reason'], path=np.round(path, 3).tolist(),
                                      gt_distance=np.round(d, 3).tolist(), departure_index=int(end) if dep is not None else None))
            print(json.dumps(dict(split=split, n=len(rows), seconds=round(time.time()-t_split, 1),
                                  traced_voxels=round(sum(float(r['length']) for r in rows), 1))), flush=True)
    finally:
        tracer.close()

    with open(out/'rows.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    with open(out/'paths.jsonl', 'w') as fh:
        for p in paths_out:
            fh.write(json.dumps(p) + '\n')
    groups = {'all': rows_out}
    for r in rows_out:
        groups.setdefault(r['split'], []).append(r)
    summary = {k: summarize_outcomes(v) for k, v in groups.items()}
    summary['seconds'] = round(time.time() - started, 1)
    (out/'summary.json').write_text(json.dumps(summary, indent=1, default=jsonable))
    a = summary['all']
    print(json.dumps({k: a.get(k) for k in ('n', 'length_precision', 'length_weighted_coverage', 'correct_length',
                                            'wrong_length', 'divergence_count')} | {'seconds': summary['seconds']},
                     default=jsonable), flush=True)


if __name__ == '__main__':
    main()
