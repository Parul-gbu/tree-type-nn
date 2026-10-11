"""Fair size comparison: tree nodes get the SAME optimisation budget (1500 epochs, lr 0.01) as the single-head baseline.
Also greedy pruning of tree nodes that do not change training accuracy."""
import numpy as np, torch, json, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_trunk import train_trunk, extract_features
from tree import TreeNode
from size_min_lib import prune, count_nodes, tree_params

d = np.load('results/catsdogs_split.npz'); Xtr, Xte, ytr, yte = d['Xtr'], d['Xte'], d['ytr'], d['yte']
rows = []
for cfg in [(12, 24, 24), (24, 48, 48)]:
    if cfg == (24, 48, 48): trunk = torch.load('results/catsdogs_trunk.pt', weights_only=False)
    else: trunk = train_trunk(Xtr, ytr, Xte, yte, c1=cfg[0], c2=cfg[1], feat_dim=cfg[2], epochs=60, lr=3e-3, verbose=False)
    ftr, fte = extract_features(trunk, Xtr), extract_features(trunk, Xte)
    for h in [2, 4, 8, 16, 32]:
        t0 = time.time(); torch.manual_seed(0)
        t = TreeNode(hidden=h, max_depth=10, min_samples=1, epochs=1500, lr=0.01, budget=[120]).fit(ftr, ytr)
        tr, te = float((t.predict(ftr) == ytr).mean()), float((t.predict(fte) == yte).mean())
        r = dict(trunk=cfg, trunk_params=trunk.n_params(), head=h, nodes=t.n_nodes_, tree_params=t.total_params(),
                 total=trunk.n_params() + t.total_params(), train=tr, test=te, secs=time.time() - t0)
        if tr >= 0.99:
            prune(t, t, ftr, ytr, tr)
            r.update(pruned_nodes=count_nodes(t), pruned_tree_params=tree_params(t), pruned_total=trunk.n_params() + tree_params(t),
                     pruned_train=float((t.predict(ftr) == ytr).mean()), pruned_test=float((t.predict(fte) == yte).mean()))
        rows.append(r); print(r, flush=True)
json.dump(rows, open('results/size_fair.json', 'w'), indent=1)
print('DONE')
