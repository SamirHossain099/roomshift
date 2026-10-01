#!/bin/sh
# LF3: label-free methods, all rooms, 3 seeds, recorded order (_LF3) and the shuffled-frame diagnostic (_IID3).
for mod in radar wifi; do for s in 0 1 2; do for h in E01 E02 E03 E04; do
  python scripts/run_stream.py --modality $mod --hold $h --seed $s --label_every 0 --methods source norm tent smooth --tag _LF3
  python scripts/run_stream.py --modality $mod --hold $h --seed $s --label_every 0 --methods source norm tent --iid --tag _IID3
done; done; done
