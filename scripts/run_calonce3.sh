#!/bin/sh
# E5c: fit once on the first 240 labels (the room's first three subjects) with the fit window covering all 240
# (window=240; E5b used the default window of 64, i.e. effectively the third subject alone), then freeze.
for mod in radar wifi; do for s in 0 1 2; do for h in E01 E02 E03 E04; do
  python scripts/run_stream.py --modality $mod --hold $h --seed $s --label_every 100 \
    --methods shift rigid affine affine+smooth --kw freeze_after=240 window=240 --tag _CAL240
done; done; done
