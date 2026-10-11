"""Minimise network size (Cats vs Dogs): trunk width x head width grid, greedy tree pruning, and
comparison against a single wide head (no tree)."""
import numpy as np, torch, torch.nn as nn, json, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_trunk import train_trunk, extract_features
from tree import TreeNode
from models import NodeHead

d = np.load('results/catsdogs_split.npz'); Xtr, Xte, ytr, yte = d['Xtr'], d['Xte'], d['ytr'], d['yte']

def count_nodes(n):
    return 1 + (count_nodes(n.child_A) if n.child_A else 0) + (count_nodes(n.child_B) if n.child_B else 0)

def prune(node, root, feats, y, base_acc):
    """Greedy bottom-up pruning: drop a child subtree if train accuracy of the whole tree does not fall."""
    for attr in ['child_A', 'child_B']:
        ch = getattr(node, attr)
        if ch is None: continue
        prune(ch, root, feats, y, base_acc)
        setattr(node, attr, None)
        if (root.predict(feats) == y).mean() < base_acc:
            setattr(node, attr, ch)
    return node

def tree_params(n):  # recompute after pruning
    return n.head.n_params() + (tree_params(n.child_A) if n.child_A else 0) + (tree_params(n.child_B) if n.child_B else 0)

rows = []
TRUNKS = [(4, 8, 8), (8, 16, 16), (12, 24, 24), (16, 32, 32), (24, 48, 48)]
HEADS = [2, 4, 8, 16, 32]
for (c1, c2, fd) in TRUNKS:
    t0 = time.time()
    trunk = train_trunk(Xtr, ytr, Xte, yte, c1=c1, c2=c2, feat_dim=fd, epochs=60, lr=3e-3, verbose=False)
    ftr, fte = extract_features(trunk, Xtr), extract_features(trunk, Xte)
    print(f'trunk {(c1, c2, fd)} params={trunk.n_params()} ({time.time()-t0:.0f}s)', flush=True)
    for h in HEADS:
        torch.manual_seed(0)
        t = TreeNode(hidden=h, max_depth=10, min_samples=1, epochs=150, lr=0.05, budget=[250]).fit(ftr, ytr)
        tr, te = float((t.predict(ftr) == ytr).mean()), float((t.predict(fte) == yte).mean())
        r = dict(trunk=(c1, c2, fd), trunk_params=trunk.n_params(), head=h, nodes=t.n_nodes_, tree_params=t.total_params(),
                 total_params=trunk.n_params() + t.total_params(), train=tr, test=te)
        # pruning
        if tr >= 0.99:
            prune(t, t, ftr, ytr, tr)
            r.update(pruned_nodes=count_nodes(t), pruned_tree_params=tree_params(t),
                     pruned_total=trunk.n_params() + tree_params(t),
                     pruned_train=float((t.predict(ftr) == ytr).mean()), pruned_test=float((t.predict(fte) == yte).mean()))
        rows.append(r); print(' ', r, flush=True)

# single wide head (no tree) on the biggest trunk, for comparison
trunk = train_trunk(Xtr, ytr, Xte, yte, c1=24, c2=48, feat_dim=48, epochs=60, lr=3e-3, verbose=False)
ftr, fte = extract_features(trunk, Xtr), extract_features(trunk, Xte)
single = []
for h in [16, 32, 64, 128, 256, 512, 1024]:
    torch.manual_seed(0); head = NodeHead(48, h); opt = torch.optim.Adam(head.parameters(), lr=0.01); lf = nn.BCEWithLogitsLoss()
    yt = torch.as_tensor(ytr, dtype=torch.float32)
    for ep in range(1500):
        opt.zero_grad(); lf(head(ftr), yt).backward(); opt.step()
    with torch.no_grad():
        r = dict(head=h, params=head.n_params(), train=float(((head(ftr) >= 0).numpy() == ytr).mean()), test=float(((head(fte) >= 0).numpy() == yte).mean()))
    single.append(r); print(' single', r, flush=True)
json.dump(dict(grid=rows, single=single), open('results/size_min.json', 'w'), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for r in rows:
    ax[0].scatter(r['total_params'], r['train'], c='C0', s=18)
    ax[0].scatter(r['total_params'], r['test'], c='C1', s=18)
ax[0].axhline(0.99, color='gray', ls='--'); ax[0].set_xscale('log'); ax[0].set_xlabel('total params (trunk + tree)'); ax[0].set_ylabel('accuracy')
ax[0].set_title('Size grid: blue = train, orange = test'); 
ok = [r for r in rows if 'pruned_total' in r]
if not ok: ok = []
ax[1].scatter([r['total_params'] for r in ok], [r['pruned_total'] for r in ok], c='C2')
if ok:
    lim = [min(r['pruned_total'] for r in ok) * 0.8, max(r['total_params'] for r in ok) * 1.2]
    ax[1].plot(lim, lim, 'k--', lw=0.8); ax[1].set_xscale('log'); ax[1].set_yscale('log')
ax[1].set_xlabel('total params before pruning'); ax[1].set_ylabel('after pruning (same train acc)'); ax[1].set_title('Greedy pruning of tree nodes')
plt.tight_layout(); plt.savefig('results/size_min.png', dpi=130); print('DONE', flush=True)
