#!/usr/bin/env bash
# v2 scale-up: every annotated fiber, K start points x 2 directions, conf 0.5 (CLI default), same checkpoint and villa
# commit as v1. Per source, in order: seeds -> CT prefetch -> trace -> fwd/back -> image features. The next source's
# prefetch overlaps the current GPU work only when >= 80 GB disk is free; completed sources' CT caches are deleted
# (largest first) whenever free disk drops below 60 GB before a prefetch.
ulimit -n 524288
cd /workspace && . venv/bin/activate
export HF_HOME=/workspace/hf LC_ALL=C.UTF-8
declare -A FIB=([paris4_heldout]=fibers/paris4_eval [PHerc1451]=fibers/PHerc1451 [PHerc0332]=fibers/PHerc0332 [PHerc0139]=fibers/PHerc0139)
declare -A CT=([paris4_heldout]=s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
               [PHerc0139]=s3://vesuvius-challenge-open-data/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr
               [PHerc1451]=s3://vesuvius-challenge-open-data/PHerc1451/volumes/20260319101107-2.399um-0.2m-78keV-masked.zarr
               [PHerc0332]=s3://vesuvius-challenge-open-data/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr)
declare -A K=([paris4_heldout]=4 [PHerc1451]=4 [PHerc0332]=4 [PHerc0139]=2)
declare -A EXTRA=([paris4_heldout]="--heldout-hash 62")
ORDER=(paris4_heldout PHerc1451 PHerc0332 PHerc0139)
log() { echo "$(date -u +%T) $*" >> logs/v2_status.log; }
args() { echo --name $1 --fibers ${FIB[$1]} --ct ${CT[$1]} --out runs/v2/$1 --per-fiber ${K[$1]} ${EXTRA[$1]}; }
free_gb() { df -BG --output=avail /workspace | tail -1 | tr -dc 0-9; }
cache_dir() { python -c "from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import RemoteChunkedArray as R; print(R.cache_path('$1', 2, '/workspace/cache'))"; }
make_room() {
  for s in "${DONE[@]}"; do
    [ "$(free_gb)" -ge 60 ] && return
    d=$(cache_dir ${CT[$s]}); log "free $(free_gb)G < 60G: deleting CT cache of $s ($d)"; rm -rf "$d"
  done
}
prepare() {   # seeds + CT prefetch for one source; marks .ready or .failed
  local s=$1
  [ -s runs/v2/$s/manifest.json ] || python code/heldout_eval.py $(args $s) --manifest-only > logs/v2_manifest_$s.log 2>&1 || { touch runs/v2/$s.failed; return; }
  python code/prefetch_ct.py --run runs/v2/$s --threads 64 > logs/v2_prefetch_$s.log 2>&1
  touch runs/v2/$s.ready
}
mkdir -p runs/v2 logs; DONE=()
log "start (free $(free_gb)G)"
prepare ${ORDER[0]}
for n in 0 1 2 3; do
  s=${ORDER[$n]}; next=${ORDER[$((n+1))]:-}
  while [ ! -e runs/v2/$s.ready ] && [ ! -e runs/v2/$s.failed ]; do
    [ -n "${PREP:-}" ] && ! kill -0 $PREP 2>/dev/null && [ ! -e runs/v2/$s.ready ] && [ ! -e runs/v2/$s.failed ] && { make_room; prepare $s; }
    sleep 10
  done
  [ -e runs/v2/$s.failed ] && { log "$s seeds FAILED"; continue; }
  PREP=
  if [ -n "$next" ] && [ "$(free_gb)" -ge 80 ]; then prepare $next & PREP=$!; log "prefetching $next in parallel (free $(free_gb)G)"; fi
  log "trace $s"
  python code/heldout_eval.py $(args $s) --manifest runs/v2/$s/manifest.json --batch 128 --log-decisions > logs/v2_trace_$s.log 2>&1 || log "$s trace FAILED"
  log "fwdback $s"
  python code/fwdback.py --run runs/v2/$s --batch 256 > logs/v2_fwdback_$s.log 2>&1 || log "$s fwdback FAILED"
  log "image features $s"
  python code/image_features.py --run runs/v2/$s --out runs/v2/$s/img > logs/v2_img_$s.log 2>&1 || log "$s image features FAILED"
  log "done $s (free $(free_gb)G)"
  DONE+=($s)
  if [ -n "$next" ] && [ -z "$PREP" ]; then make_room; prepare $next; fi
done
wait
log "ALL_DONE"
