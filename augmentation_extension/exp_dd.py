"""Q1 (double descent): head-width sweep with 20% label noise, with and without augmented views of each training image (labels shared across views)."""
import numpy as np, torch, torch.nn as nn, json
from aug_lib import to_t, view_features
from models import NodeHead
d = np.load('results2/split.npz'); Xtr_t, Xte_t, ytr, yte = to_t(d['Xtr']), to_t(d['Xte']), d['ytr'], d['yte']
trunk = torch.load('results2/q1_1_0.pt', weights_only=False)['trunk']       # augmentation-trained trunk, seed 0 (frozen)
rng = np.random.default_rng(123); yn = np.where(rng.random(len(ytr)) < 0.20, 1 - ytr, ytr)
fte, _ = view_features(trunk, Xte_t, 1)
WIDTHS, SEEDS, EPOCHS = [4, 8, 16, 32, 64, 128, 256, 512, 1024], (0, 1), 1000
out = {}
for K in [1, 6]:
    F, idx = view_features(trunk, Xtr_t, K, 0); yk = torch.as_tensor(np.tile(yn, K), dtype=torch.float32); ykn = yk.numpy()
    curve = []
    for h in WIDTHS:
        trs, tes, tro = [], [], []
        for sd in SEEDS:
            torch.manual_seed(sd); head = NodeHead(F.shape[1], h); opt = torch.optim.Adam(head.parameters(), lr=0.01); lf = nn.BCEWithLogitsLoss()
            for ep in range(EPOCHS):
                opt.zero_grad(); lf(head(F), yk).backward(); opt.step()
            with torch.no_grad():
                pr = (head(F) >= 0).numpy()
                trs.append(float((pr != ykn).mean())); tro.append(float((pr[:len(yn)] != yn).mean())); tes.append(float(((head(fte) >= 0).numpy() != yte).mean()))
        r = dict(K=K, width=h, params=head.n_params(), train_err=np.mean(trs), train_err_orig=np.mean(tro), test_err=np.mean(tes), test_std=np.std(tes))
        curve.append(r); print(r, flush=True)
    out[f'K{K}'] = dict(N_eff=len(yn) * K, curve=curve)
    json.dump(out, open('results2/dd.json', 'w'), indent=1, default=float)
print('DONE')
