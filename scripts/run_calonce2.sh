#!/bin/sh
# E5b: translation-only correction (shift, 3 parameters) for the calibrate-once test, and a longer calibration
# (F=240 labels = the room's first three subjects) for every correction.
for mod in radar wifi; do for s in 0 1 2; do for h in E01 E02 E03 E04; do
  for F in 0 64; do
    python scripts/run_stream.py --modality $mod --hold $h --seed $s --label_every 100 --methods shift --kw freeze_after=$F --tag _CAL$F
  done
  python scripts/run_stream.py --modality $mod --hold $h --seed $s --label_every 100 \
    --methods shift rigid affine affine+smooth --kw freeze_after=240 --tag _CAL240
done; done; done
