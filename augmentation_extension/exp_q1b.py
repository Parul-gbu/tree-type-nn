"""Q1 (overfit + augmentation): bigger trees (h=128, depth 12, up to 400 extra nodes, 300 epochs/node) so the tree can interpolate.
Compare on the augmentation-trained trunks: tree on clean features vs tree on K=4 augmented views."""
import numpy as np, torch, json, time
from aug_lib import to_t, view_features
from tree import TreeNode
d = np.load('results2/split.npz'); Xtr_t, Xte_t, ytr, yte = to_t(d['Xtr']), to_t(d['Xte']), d['ytr'], d['yte']
fte_cache = {}; R = []; K = 4
BIG = dict(hidden=128, max_depth=12, min_samples=1, epochs=300, lr=0.05)
for seed in [0, 1]:
    trunk = torch.load(f'results2/q1_1_{seed}.pt', weights_only=False)['trunk']
    f1, _ = view_features(trunk, Xtr_t, 1); fk, _ = view_features(trunk, Xtr_t, K, seed); fte, _ = view_features(trunk, Xte_t, 1)
    for name, f, yy in [('clean', f1, ytr), ('aug4', fk, np.tile(ytr, K))]:
        t0 = time.time(); torch.manual_seed(seed); t = TreeNode(budget=[400], **BIG).fit(f, yy)
        r = dict(seed=seed, data=name, train_fit=float((t.predict(f) == yy).mean()), train_orig=float((t.predict(f1) == ytr).mean()),
                 test=float((t.predict(fte) == yte).mean()), nodes=t.n_nodes_, params=t.total_params(), secs=time.time() - t0)
        R.append(r); print(json.dumps(r), flush=True); json.dump(R, open('results2/q1b.json', 'w'), indent=1)
print('DONE')
