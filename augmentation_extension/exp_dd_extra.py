"""Finish the DD sweep: wider heads. K=6 at 1024 (2 seeds) and 2048 (1 seed); K=1 at 2048 and 4096 (1 seed each)."""
import numpy as np, torch, torch.nn as nn, json
from aug_lib import to_t, view_features
from models import NodeHead
d = np.load('results2/split.npz'); Xtr_t, Xte_t, ytr, yte = to_t(d['Xtr']), to_t(d['Xte']), d['ytr'], d['yte']
trunk = torch.load('results2/q1_1_0.pt', weights_only=False)['trunk']
rng = np.random.default_rng(123); yn = np.where(rng.random(len(ytr)) < 0.20, 1 - ytr, ytr); fte, _ = view_features(trunk, Xte_t, 1)
out = []
for K, h, seeds in [(1, 2048, (0,)), (1, 4096, (0,)), (6, 1024, (0, 1)), (6, 2048, (0,))]:
    F, _ = view_features(trunk, Xtr_t, K, 0); yk = torch.as_tensor(np.tile(yn, K), dtype=torch.float32); ykn = yk.numpy(); trs, tes, tro = [], [], []
    for sd in seeds:
        torch.manual_seed(sd); head = NodeHead(F.shape[1], h); opt = torch.optim.Adam(head.parameters(), lr=0.01); lf = nn.BCEWithLogitsLoss()
        for ep in range(1000): opt.zero_grad(); lf(head(F), yk).backward(); opt.step()
        with torch.no_grad():
            pr = (head(F) >= 0).numpy(); trs.append(float((pr != ykn).mean())); tro.append(float((pr[:len(yn)] != yn).mean())); tes.append(float(((head(fte) >= 0).numpy() != yte).mean()))
    r = dict(K=K, width=h, params=head.n_params(), train_err=float(np.mean(trs)), train_err_orig=float(np.mean(tro)), test_err=float(np.mean(tes)), seeds=len(seeds))
    out.append(r); print(json.dumps(r), flush=True); json.dump(out, open('results2/dd_extra.json', 'w'), indent=1)
print('DONE')
