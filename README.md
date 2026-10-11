# Tree-type network with neural-network nodes (Cats vs Dogs, CIFAR-10)

A tree of tiny MLP heads on top of a shared, frozen CNN trunk. Each node is trained on its parent's data; two children
correct the parent's two error types (predicted 0 but should be 1 -> child A, predicted 1 but should be 0 -> child B),
and the parent's net input is shifted by w_A (>= 0) and w_B (<= 0) chosen so every corrected sample crosses the margin.
CIFAR-10 uses one tree per class (one-vs-rest).

## Layout
| file | purpose |
|---|---|
| `models.py` | `Trunk` (2 conv layers + BN + global average pool + projection, 13.6k params at 24/48/48), `NodeHead` (tiny MLP), multi-level feature helper |
| `tree.py` | `TreeNode` (recursive error-correcting tree, node budget), `OneVsRestTree`, `train_head` |
| `train_trunk.py` | trunk training (binary pretext for Cats-vs-Dogs, 10-way softmax for CIFAR) |
| `run_main.py` | main experiment: trunk + tree on both datasets, root-only baselines, multi-level-feature ablation |
| `double_descent.py`, `double_descent_refined.py` | head-width sweeps with label noise (refined: 20% noise, 6 seeds) and a whole-tree sweep |
| `receptive_fields.py`, `rf_control.py` | gradient-saliency maps per node, and a control for the small-sample averaging artefact |
| `size_min.py`, `size_fair.py` | size grids; `size_fair.py` uses a matched optimisation budget and pruning (use this one) |
| `results/` | JSON metrics, figures, logs, trunk weights |

## Reproduce
Edit the absolute paths at the top of `prepare_data.py` (it expects extracted Cats-vs-Dogs and Kaggle CIFAR-10 PNGs plus `trainLabels.csv`), then `./run_all.sh`.
Data are resized to 32x32; CIFAR uses 400 images per class (4,000 total). Seeds are fixed, but a single trunk seed was used.

## Caveats
- Test sets are small (476 Cats-vs-Dogs, 800 CIFAR), so differences of ~2-4 points are within noise.
- `size_min.py` (first grid) trained nodes for only 150 epochs and is kept for transparency; its tree-vs-single-head comparison is not fair. `size_fair.py` fixes that.
- No double descent was observed (see report).
