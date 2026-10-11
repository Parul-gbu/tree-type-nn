"""Refined double-descent check on Cats-vs-Dogs root head: 6 seeds, 20% label noise, dense widths near the
interpolation threshold (params ~ N), longer training, mean +- std over seeds."""
import numpy as np, torch, torch.nn as nn, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_trunk import extract_features
from models import NodeHead

NOISE, SEEDS, EPOCHS = 0.20, range(6), 2500
WIDTHS = [4, 8, 16, 24, 32, 36, 40, 44, 48, 56, 64, 96, 128, 256, 1024]
d = np.load('results/catsdogs_split.npz'); trunk = torch.load('results/catsdogs_trunk.pt', weights_only=False)
ftr, fte = extract_features(trunk, d['Xtr']), extract_features(trunk, d['Xte']); yte = d['yte']
rng = np.random.default_rng(123); yn = np.where(rng.random(len(d['ytr'])) < NOISE, 1 - d['ytr'], d['ytr'])
yt = torch.as_tensor(yn, dtype=torch.float32)
out = []
for h in WIDTHS:
    trs, tes = [], []
    for sd in SEEDS:
        torch.manual_seed(sd); head = NodeHead(48, h); opt = torch.optim.Adam(head.parameters(), lr=0.01); lf = nn.BCEWithLogitsLoss()
        for ep in range(EPOCHS):
            opt.zero_grad(); lf(head(ftr), yt).backward(); opt.step()
        with torch.no_grad():
            trs.append(float(((head(ftr) >= 0).numpy() != yn).mean())); tes.append(float(((head(fte) >= 0).numpy() != yte).mean()))
    r = dict(width=h, params=head.n_params(), train_mean=np.mean(trs), train_std=np.std(trs), test_mean=np.mean(tes), test_std=np.std(tes))
    out.append(r); print(r, flush=True)
json.dump(dict(N=len(yn), noise=NOISE, seeds=len(list(SEEDS)), curve=out), open('results/dd_refined.json', 'w'), indent=1)
p = [r['params'] for r in out]
fig, ax = plt.subplots(figsize=(7, 4.2))
for k, lab in [('train', 'train error (noisy labels)'), ('test', 'test error (clean labels)')]:
    m = np.array([r[k + '_mean'] for r in out]); s = np.array([r[k + '_std'] for r in out])
    ax.errorbar(p, m, yerr=s, marker='o', capsize=2, label=lab)
ax.axvline(len(yn), color='gray', ls='--', label=f'#params = N = {len(yn)}')
ax.set_xscale('log'); ax.set_xlabel('head parameters'); ax.set_ylabel('error'); ax.set_title(f'Cats vs Dogs root head, {int(NOISE*100)}% label noise, 6 seeds (mean ± std)')
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig('results/dd_refined.png', dpi=130); print('DONE')
