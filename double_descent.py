"""Double-descent sweep: width of a node head (frozen trunk features, label noise) + tree-level sweep."""
import numpy as np, torch, torch.nn as nn, json, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_trunk import extract_features
from models import NodeHead
from tree import TreeNode

NOISE = 0.15
WIDTHS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 96, 128, 256, 512, 1024, 2048]

def sweep(ftr, ytr_noisy, fte, yte, widths, seeds=(0, 1), epochs=1500, lr=0.01):
    out = []
    ytr_t = torch.as_tensor(ytr_noisy, dtype=torch.float32)
    for h in widths:
        trs, tes, ps = [], [], None
        for sd in seeds:
            torch.manual_seed(sd)
            head = NodeHead(ftr.shape[1], hidden=h)
            opt = torch.optim.Adam(head.parameters(), lr=lr)
            lf = nn.BCEWithLogitsLoss()
            for ep in range(epochs):
                opt.zero_grad(); loss = lf(head(ftr), ytr_t); loss.backward(); opt.step()
            with torch.no_grad():
                trs.append(float(((head(ftr) >= 0).numpy() != ytr_noisy).mean()))
                tes.append(float(((head(fte) >= 0).numpy() != yte).mean()))
            ps = head.n_params()
        out.append(dict(width=h, params=ps, train_err=float(np.mean(trs)), test_err=float(np.mean(tes))))
        print(f"  h={h:5d} params={ps:7d} train_err(noisy)={np.mean(trs):.3f} test_err={np.mean(tes):.3f}", flush=True)
    return out

def noisy(y, p, seed=0):
    rng = np.random.default_rng(seed); flip = rng.random(len(y)) < p
    return np.where(flip, 1 - y, y)

R = {}
# --- Cats vs Dogs ---
d = np.load('results/catsdogs_split.npz'); trunk = torch.load('results/catsdogs_trunk.pt', weights_only=False)
ftr, fte = extract_features(trunk, d['Xtr']), extract_features(trunk, d['Xte'])
yn = noisy(d['ytr'], NOISE)
print('Cats-vs-Dogs head sweep, N =', len(yn), flush=True)
R['catsdogs_head'] = dict(N=len(yn), noise=NOISE, curve=sweep(ftr, yn, fte, d['yte'], WIDTHS))

# --- CIFAR cat vs dog (classes 3 vs 5) ---
c = np.load('results/cifar_split.npz'); trunk = torch.load('results/cifar_trunk.pt', weights_only=False)
def sub(X, y):
    m = (y == 3) | (y == 5); return X[m], (y[m] == 5).astype(int)
Xa, ya = sub(c['Xtr'], c['ytr']); Xb, yb = sub(c['Xte'], c['yte'])
ftr, fte = extract_features(trunk, Xa), extract_features(trunk, Xb)
yn = noisy(ya, NOISE)
print('CIFAR cat-vs-dog head sweep, N =', len(yn), flush=True)
R['cifar_head'] = dict(N=len(yn), noise=NOISE, curve=sweep(ftr, yn, fte, yb, WIDTHS))

# --- tree-level sweep on Cats vs Dogs (no label noise): total params vs error ---
trunk = torch.load('results/catsdogs_trunk.pt', weights_only=False)
ftr, fte = extract_features(trunk, d['Xtr']), extract_features(trunk, d['Xte'])
tree_curve = []
print('tree-level width sweep', flush=True)
for h in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    torch.manual_seed(0)
    t = TreeNode(hidden=h, max_depth=6, min_samples=1, epochs=150, lr=0.05, budget=[120]).fit(ftr, d['ytr'])
    r = dict(width=h, params=t.total_params(), nodes=t.n_nodes_,
             train_err=float((t.predict(ftr) != d['ytr']).mean()), test_err=float((t.predict(fte) != d['yte']).mean()))
    tree_curve.append(r); print(' ', r, flush=True)
R['tree_sweep'] = tree_curve
json.dump(R, open('results/double_descent.json', 'w'), indent=1)

# --- plots ---
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for a, key, title in [(ax[0], 'catsdogs_head', 'Cats vs Dogs: root-node head width'), (ax[1], 'cifar_head', 'CIFAR-10 cat vs dog: head width')]:
    cv = R[key]['curve']; p = [r['params'] for r in cv]
    a.plot(p, [r['train_err'] for r in cv], 'o-', label='train error (noisy labels)')
    a.plot(p, [r['test_err'] for r in cv], 's-', label='test error (clean labels)')
    a.axvline(R[key]['N'], color='gray', ls='--', label=f"#params = N = {R[key]['N']}")
    a.set_xscale('log'); a.set_xlabel('head parameters'); a.set_ylabel('error'); a.set_title(title); a.legend(fontsize=8)
cv = R['tree_sweep']; p = [r['params'] for r in cv]
ax[2].plot(p, [r['train_err'] for r in cv], 'o-', label='train error'); ax[2].plot(p, [r['test_err'] for r in cv], 's-', label='test error')
ax[2].set_xscale('log'); ax[2].set_xlabel('total tree parameters'); ax[2].set_title('Whole tree (Cats vs Dogs), sweep of head width'); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig('results/double_descent.png', dpi=130)
print('DONE', flush=True)
