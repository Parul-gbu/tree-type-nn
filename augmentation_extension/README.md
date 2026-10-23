# Augmentation extension: tree-type network with NN nodes

Adds data augmentation, ensembles, bias/variance and loss-landscape analysis to the earlier tree-of-MLP-heads project (Cats vs Dogs, 32x32).
Drop this folder into the root of the main repo: the scripts import `models.py` and `tree.py` from the parent directory, so run them from the repo root
(`python3 augmentation_extension/exp_q1.py` after `cd`-ing so that `data_cache/catsdogs_X.npy` and `results2/` resolve, or copy the scripts next to `tree.py`).

| script | purpose |
|---|---|
| `aug_lib.py` | augmentation (flip, +/-4 px translation, +/-20% brightness/contrast), end-to-end trunk training, K-view feature extraction |
| `exp_q1.py` | Q1: plain vs augmented trunk x tree on clean vs 4 augmented views (3 seeds) |
| `exp_q1b.py` | Q1: bigger trees that can interpolate |
| `exp_dd.py`, `exp_dd_extra.py` | Q1: head-width sweep with 20% label noise, K=1 vs K=6 views |
| `exp_bv.py`, `analyze_bv.py` | Q2 + Q3: bias/variance (trunk retrained per replicate) and majority-vote ensembles |
| `exp_land.py` | Q1 epoch-wise curves + Q4: Hessian sharpness, 2D slices, interpolation |
| `run_ext.sh` | runs everything in order (~1 hour on one CPU core) |
| `results/` | JSON metrics, figures, logs, trunk weights |
| `docs/report_augmentation.pdf` | the brief report |

Caveats: one dataset, 3 seeds (Q1), 5 replicates (bias/variance), single trunk for sweeps and landscape; see the report.
