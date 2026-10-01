#!/bin/sh
# Same runs as run_final2.sh for one seed, skipping (room, tag) results already on disk.
# Usage: sh scripts/run_final2_seed.sh <modality> <lr> <steps> <seed>
mod=$1; lr=$2; st=$3; s=$4
for h in E01 E02 E03 E04; do
  ls results/stream_${mod}_${h}_s${s}_K100_F2_*.json >/dev/null 2>&1 || \
    python scripts/run_stream.py --modality $mod --hold $h --seed $s --label_every 100 \
      --methods source norm smooth rigid affine affine+smooth supft --kw lr=$lr steps=$st --tag _F2
  ls results/stream_${mod}_${h}_s${s}_K1000_B2_*.json >/dev/null 2>&1 || \
    python scripts/run_stream.py --modality $mod --hold $h --seed $s --label_every 20 50 200 500 1000 \
      --methods affine+smooth supft --kw lr=$lr steps=$st --tag _B2
done
