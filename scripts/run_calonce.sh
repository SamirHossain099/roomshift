#!/bin/sh
# E5: calibrate once per room. Fit on the first F labelled frames (K=100; all from the room's first subject for
# F <= 64), freeze, score the other nine subjects from per_subject. F=0 = continuous updating, for comparison.
for mod in radar wifi; do for s in 0 1 2; do for h in E01 E02 E03 E04; do for F in 0 16 64; do
  python scripts/run_stream.py --modality $mod --hold $h --seed $s --label_every 100 \
    --methods source rigid affine affine+smooth --kw freeze_after=$F --tag _CAL$F
done; done; done; done
