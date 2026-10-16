"""Warm the tracer's persistent CT chunk cache around every seeded GT fiber, in parallel.

Chunks are written by villa's own RemoteChunkedArray (decoded, uncompressed, atomic),
so the tracer reads them exactly as if it had fetched them itself.
Footprint: GT points within --reach trace-grid voxels of arclength from each seed,
dilated by --margin CT voxels (the crop half-extent).
"""
import argparse
import concurrent.futures as cf
import json
import time
from pathlib import Path

import numpy as np

from vesuvius.neural_tracing.tracers.shared.annotations.vc3d import load_fibers
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import RemoteChunkedArray


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True, help='run directory holding manifest.json and config.json')
    ap.add_argument('--reach', type=float, default=1000.0)
    ap.add_argument('--margin', type=float, default=96.0, help='CT voxels around each GT point')
    ap.add_argument('--threads', type=int, default=48)
    ap.add_argument('--cache-dir', default='/workspace/cache')
    args = ap.parse_args()

    run = Path(args.run)
    config = json.loads((run/'config.json').read_text())
    manifest = json.loads((run/'manifest.json').read_text())
    grid, ct_scale = float(config['grid_scale']), 4.0  # trace voxel -> base voxels; CT level-2 voxel -> base voxels
    fibers = load_fibers(config['fibers_dir'], grid_scale=grid)
    names = {f['name']: i for i, f in enumerate(manifest['fibers'])} if isinstance(manifest['fibers'], list) else {}
    by_name = {f.name: f for f in fibers}
    arr = RemoteChunkedArray(config['ct'], config['ct_level'], args.cache_dir, 1 << 30)
    shape, chunks = np.asarray(arr.shape), np.asarray(arr.chunks)

    keys = set()
    for split in ('monitor', 'calibration', 'final'):
        for seed in manifest[split]:
            entry = manifest['fibers'][seed['fiber']]
            f = by_name[entry['name']] if isinstance(entry, dict) else fibers[seed['fiber']]
            ahead = (f.s - seed['t']) * float(seed['sign'])  # each seed traces one direction
            near = (ahead >= -64.0) & (ahead <= args.reach)
            pts = f.points[near] * grid / ct_scale  # trace-grid xyz -> CT level-2 xyz
            zyx = pts[:, ::-1]
            for offset in np.array(np.meshgrid(*[[-args.margin, 0, args.margin]]*3)).reshape(3, -1).T:
                q = np.clip(np.floor((zyx + offset) / chunks).astype(int), 0, (shape - 1) // chunks)
                keys.update(map(tuple, np.unique(q, axis=0)))
    cached = sum((Path(arr.path)/arr.sep.join(map(str, k))).exists() for k in keys) if hasattr(arr, 'sep') else 0
    print(f'{config["name"]}: {len(keys)} chunks in footprint ({len(keys)*2/1024:.1f} GiB), {cached} already cached', flush=True)

    t0, done = time.time(), 0
    with cf.ThreadPoolExecutor(args.threads) as pool:
        for _ in pool.map(lambda k: arr._load(k), sorted(keys)):
            done += 1
            if done % 500 == 0:
                print(f'  {done}/{len(keys)} chunks, {done*2/(time.time()-t0):.0f} MiB/s', flush=True)
    print(f'{config["name"]}: prefetched {len(keys)} chunks in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
