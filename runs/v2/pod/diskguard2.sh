#!/usr/bin/env bash
# Disk guard v2 for PHerc0139: when free < 20G, evict the OLDEST cached CT chunk files (fetched for earlier, finished
# batches) until free >= 35G. villa's RemoteChunkedArray re-downloads any missing chunk on demand, so this costs
# bandwidth only, never correctness. Only digit-named chunk files are touched (not .zarray/.zattrs/locks/temp files).
C=/workspace/cache/s3/vesuvius-challenge-open-data/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr/2
cd /workspace
while ! grep -q ALL_DONE logs/v2_status.log; do
  free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  if [ "$free" -lt 20 ]; then
    n=$(( (35 - free) * 512 ))   # 2 MiB chunks
    find "$C" -mindepth 3 -maxdepth 3 -type f -regextype posix-extended -regex '.*/2/[0-9]+/[0-9]+/[0-9]+' -printf '%T@ %p\n' \
      | sort -n | head -n $n | cut -d' ' -f2- | xargs -r rm -f
    echo "$(date -u +%T) diskguard2: free ${free}G -> evicted $n oldest 0139 chunks, now $(df -BG --output=avail / | tail -1 | tr -dc 0-9)G" >> logs/v2_status.log
  fi
  sleep 20
done
