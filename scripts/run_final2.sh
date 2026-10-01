#!/bin/sh
# F2 (headline, K=100, all methods) and B2 (label budget) at the tuned supervised settings from E3b
# (scripts/sweep_supft.py). Usage: sh scripts/run_final2.sh <modality> <lr> <steps>
mod=$1; lr=$2; st=$3
for s in 0 1 2; do for h in E01 E02 E03 E04; do
  python scripts/run_stream.py --modality $mod --hold $h --seed $s --label_every 100 \
    --methods source norm smooth rigid affine affine+smooth supft --kw lr=$lr steps=$st --tag _F2
  python scripts/run_stream.py --modality $mod --hold $h --seed $s --label_every 20 50 200 500 1000 \
    --methods affine+smooth supft --kw lr=$lr steps=$st --tag _B2
done; done
