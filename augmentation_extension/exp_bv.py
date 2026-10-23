"""Q2/Q3 engine: for n in NS and B replicates (different training subsets + seeds), train trunk (record test logits across epochs),
grow a tree, store test outputs. Run with and without augmentation. Also trees on a shared trunk (n=1600) for ensembles."""
import numpy as np, torch, pickle, time
from aug_lib import to_t, train_e2e, view_features
from tree import TreeNode
d = np.load('results2/split.npz'); Xtr_t, Xte_t, ytr, yte = to_t(d['Xtr']), to_t(d['Xte']), d['ytr'], d['yte']
E = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40]; NS = [200, 400, 800, 1600]; B = 5
TREE = dict(hidden=64, max_depth=8, min_samples=1, epochs=150, lr=0.05)
res = dict(E=E, NS=NS, B=B, yte=yte, runs={}, shared={})
for aug in [False, True]:
    K = 3 if aug else 1
    for n in NS:
        L, N, TA, ND, PR = [], [], [], [], []
        for b in range(B):
            t0 = time.time(); idx = np.random.default_rng(100 + b).choice(len(ytr), n, replace=False)
            trunk, head, rec = train_e2e(Xtr_t[idx], ytr[idx], epochs=40, aug=aug, seed=b, Xte_t=Xte_t, record=E)
            L.append(np.stack([rec[e] for e in E]))
            f, _ = view_features(trunk, Xtr_t[idx], K, b); fte, _ = view_features(trunk, Xte_t, 1); yy = np.tile(ytr[idx], K)
            torch.manual_seed(b); t = TreeNode(budget=[200], **TREE).fit(f, yy)
            N.append(t.total_net(fte)); TA.append(float((t.predict(f[:n]) == ytr[idx]).mean())); ND.append(t.n_nodes_); PR.append(t.total_params())
            print(f'aug={aug} n={n} b={b} trunk+tree {time.time()-t0:.0f}s nodes={t.n_nodes_} tree_test={(N[-1]>=0).astype(int).__eq__(yte).mean():.3f}', flush=True)
            if n == NS[-1] and b == 0: keep = (f, yy, fte, K)
        res['runs'][(aug, n)] = dict(logits=np.stack(L), tree_net=np.stack(N), train_acc=TA, nodes=ND, params=PR)
        pickle.dump(res, open('results2/bv.pkl', 'wb'))
    # shared-trunk ensemble: 5 trees on one trunk, different bootstrap resamples of the images (all K views kept) and seeds
    f, yy, fte, K_ = keep; n = NS[-1]; nets = []
    for m in range(5):
        bs = np.random.default_rng(900 + m).choice(n, n, replace=True); rows = np.concatenate([bs + k * n for k in range(K_)])
        torch.manual_seed(m); t = TreeNode(budget=[200], **TREE).fit(f[rows], yy[rows]); nets.append(t.total_net(fte)); print('shared', aug, m, flush=True)
    res['shared'][aug] = np.stack(nets); pickle.dump(res, open('results2/bv.pkl', 'wb'))
print('DONE')
