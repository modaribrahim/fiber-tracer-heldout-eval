#!/usr/bin/env bash
# Disk guard for run v2: frees CT caches of finished sources when space runs low; never touches run outputs.
cd /workspace; . venv/bin/activate
cdir() { python -c "from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import RemoteChunkedArray as R; print(R.cache_path('$1', 2, '/workspace/cache'))"; }
C1451=$(cdir s3://vesuvius-challenge-open-data/PHerc1451/volumes/20260319101107-2.399um-0.2m-78keV-masked.zarr)
C0332=$(cdir s3://vesuvius-challenge-open-data/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr)
while ! grep -q ALL_DONE logs/v2_status.log; do
  free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  if [ "$free" -lt 40 ] && [ -d "$C1451" ] && grep -q "done PHerc1451" logs/v2_status.log; then
    echo "$(date -u +%T) diskguard: free ${free}G, deleting PHerc1451 CT cache" >> logs/v2_status.log; rm -rf "$C1451"
  elif [ "$free" -lt 40 ] && [ -d "$C0332" ] && grep -q "done PHerc0332" logs/v2_status.log; then
    echo "$(date -u +%T) diskguard: free ${free}G, deleting PHerc0332 CT cache" >> logs/v2_status.log; rm -rf "$C0332"
  elif [ "$free" -lt 12 ]; then
    echo "$(date -u +%T) diskguard: WARNING free ${free}G and nothing left to delete" >> logs/v2_status.log
  fi
  sleep 30
done
