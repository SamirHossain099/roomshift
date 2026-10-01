# roomshift

Code and results for **"Where the Person Stands: Cross-Room Adaptation of mmWave Radar and Wi-Fi Pose Estimation"**
(Samir Hossain, Texas Tech University; manuscript submitted).

Archived at Zenodo: [10.5281/zenodo.23092426](https://doi.org/10.5281/zenodo.23092426) (concept DOI, always the latest release).

3-D human pose estimated from mmWave radar or Wi-Fi CSI degrades in a new room. On the four rooms of
[MM-Fi](https://ntu-aiot-lab.github.io/mm-fi), with each room held out in turn and streamed frame by frame, this
repository measures what changes (mostly where the person is, partly the sensor placement), why label-free
test-time adaptation of normalization statistics fails for absolute pose (the statistics carry the person's
position), and how a gradient-free 12-parameter output correction from sparse labels compares with tuned online
fine-tuning across label budgets, including a correction fitted once per room.

## Contents

| Path | What |
|---|---|
| `src/mmfi/` | data windows, models (PointNet radar, CNN Wi-Fi), adaptation methods, metrics |
| `scripts/preprocess.py`, `train_eval.py` | MM-Fi to per-room arrays; leave-one-room-out training (checkpoints are not released) |
| `scripts/run_stream.py` | the online stream protocol (predict, then adapt; one label every K frames) |
| `scripts/sweep_supft.py` | tuning grid for the fine-tuning baseline |
| `scripts/run_*.sh` | the exact runs behind the paper (final, label budget, calibrate once, label-free) |
| `scripts/confound.py`, `mechanism.py`, `bias.py`, `predeploy.py`, `latency.py` | the analyses of Sections 4 to 7 |
| `scripts/headline.py`, `manuscript_numbers.py` | aggregation into `results/*.json` and every number quoted in the paper |
| `src/figures.py` | the figures, from `results/` only |
| `results/` | per-run results of the final runs and the aggregates |
| `tests/` | tests that pin each claim of the paper to `results/`, and the packaging guard |

## Reproducing

MM-Fi is available from its authors; it is not redistributed here, nor are models trained on it.

```bash
pip install -r requirements.txt
python scripts/preprocess.py                      # set MMFI_ROOT to the extracted MMFi_Dataset folder
python scripts/train_eval.py --modality radar --seeds 0 1 2
python scripts/train_eval.py --modality wifi --seeds 0 1 2
sh scripts/run_final2.sh radar 3e-6 4 && sh scripts/run_final2.sh wifi 1e-3 64
sh scripts/run_labelfree.sh && sh scripts/run_calonce.sh && sh scripts/run_calonce2.sh && sh scripts/run_calonce3.sh
python scripts/headline.py
python -m pytest tests -q
```

The tests run on the released `results/` without the dataset. `results/fig_mechanism.json` (per-frame poses for
Figure 1) is not released, so `src/figures.py` regenerates Figure 1 only after `scripts/fig_mechanism_data.py`.

## Citation

See `CITATION.cff`. Please also cite MM-Fi (Yang et al., NeurIPS 2023 Datasets and Benchmarks).

## Licence

MIT, for the code and the result files in this repository.
