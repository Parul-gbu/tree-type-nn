"""Q1 (epoch-wise) + Q4 (loss landscape) on node heads over FROZEN augmentation-trained trunk features.
Settings: K in {1 (clean only), 6 (6 views/image)} x label noise in {0, 20%}.
Part A: wide head (h=256), 4000 epochs: train/test error+loss every 40 epochs, top Hessian eigenvalue at checkpoints.
Part B: h=64, 3000 epochs: top Hessian eigenvalue, filter-normalised 2D slices, linear interpolation between two seeds."""
import numpy as np, torch, torch.nn as nn, json, time
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from aug_lib import to_t, view_features
from models import NodeHead

d = np.load('results2/split.npz'); Xtr_t, Xte_t, ytr, yte = to_t(d['Xtr']), to_t(d['Xte']), d['ytr'], d['yte']
trunk = torch.load('results2/q1_1_0.pt', weights_only=False)['trunk']
fte, _ = view_features(trunk, Xte_t, 1); f1, _ = view_features(trunk, Xtr_t, 1)
rng = np.random.default_rng(123); flip = rng.random(len(ytr)) < 0.20
lf = nn.BCEWithLogitsLoss()
VIEWS = {1: f1, 6: view_features(trunk, Xtr_t, 6, 0)[0]}
def labels(noise): return np.where(flip, 1 - ytr, ytr) if noise else ytr
def T(y): return torch.as_tensor(y, dtype=torch.float32)

def loss_of(head, F, y): return lf(head(F), y)
def err(head, F, y):
    with torch.no_grad(): return float(((head(F) >= 0).numpy() != y).mean())
def top_eig(head, F, y, iters=25):
    ps = [p for p in head.parameters()]; v = [torch.randn_like(p) for p in ps]
    nrm = lambda vs: torch.sqrt(sum((x ** 2).sum() for x in vs)); n0 = nrm(v); v = [x / n0 for x in v]; lam = 0.0
    for _ in range(iters):
        g = torch.autograd.grad(loss_of(head, F, y), ps, create_graph=True)
        hv = torch.autograd.grad(sum((gi * vi).sum() for gi, vi in zip(g, v)), ps)
        lam = float(sum((h * x).sum() for h, x in zip(hv, v))); n = nrm(hv) + 1e-12; v = [h / n for h in hv]
    return lam

def train(h, F, y, epochs, seed, lr=0.01, record=None, checkpoints=()):
    torch.manual_seed(seed); head = NodeHead(F.shape[1], h); opt = torch.optim.Adam(head.parameters(), lr=lr); yt = T(y); out = []; eig = {}
    ynp = y
    for ep in range(1, epochs + 1):
        opt.zero_grad(); loss_of(head, F, yt).backward(); opt.step()
        if record and ep % record == 0:
            with torch.no_grad(): tl = float(lf(head(fte), T(yte)))
            out.append(dict(epoch=ep, train_err=err(head, F, ynp), test_err=err(head, fte, yte), train_loss=float(loss_of(head, F, yt)), test_loss=tl))
        if ep in checkpoints: eig[ep] = top_eig(head, F, yt)
    return head, out, eig

R = {'A': {}, 'B': {}}
# ---------------- Part A ----------------
for K in [1, 6]:
    for noise in [0, 1]:
        t0 = time.time(); F = VIEWS[K]; y = np.tile(labels(noise), K)
        _, curve, eig = train(256, F, y, 4000, 0, record=40, checkpoints=(40, 200, 1000, 4000))
        R['A'][f'K{K}_noise{noise}'] = dict(curve=curve, eig=eig); print('A', K, noise, f'{time.time()-t0:.0f}s', eig, flush=True)
json.dump(R, open('results2/land.json', 'w'), indent=1)

# ---------------- Part B ----------------
heads = {}
for K in [1, 6]:
    for noise in [0, 1]:
        F = VIEWS[K]; y = np.tile(labels(noise), K); key = f'K{K}_noise{noise}'
        hs, eigs = [], []
        for sd in [0, 1]:
            hd, _, e = train(64, F, y, 3000, sd, checkpoints=(3000,)); hs.append(hd); eigs.append(e[3000])
        heads[key] = hs
        yt = T(y); yte_t = T(yte)
        R['B'][key] = dict(eig_train=eigs, eig_test=[top_eig(h, fte, yte_t) for h in hs],
                           final_train_loss=float(loss_of(hs[0], F, yt)), final_test_loss=float(loss_of(hs[0], fte, yte_t)),
                           train_err=err(hs[0], F, y), test_err=err(hs[0], fte, yte)); print('B', key, R['B'][key], flush=True)
json.dump(R, open('results2/land.json', 'w'), indent=1)

def flat(h): return torch.cat([p.detach().flatten() for p in h.parameters()])
def setflat(h, v):
    i = 0
    for p in h.parameters(): n = p.numel(); p.data.copy_(v[i:i + n].view_as(p)); i += n
def fn_direction(h, gen):
    """filter-normalised random direction: each weight row gets a Gaussian direction rescaled to that row's norm; biases get zero."""
    parts = []
    for p in h.parameters():
        if p.dim() == 2:
            dd = torch.randn(p.shape, generator=gen); dd = dd / dd.norm(dim=1, keepdim=True) * p.detach().norm(dim=1, keepdim=True); parts.append(dd.flatten())
        else: parts.append(torch.zeros(p.numel()))
    return torch.cat(parts)

# 2D slices (train loss measured on the CLEAN training images/true-or-noisy labels each head was trained with, test loss on clean test)
G = 21; al = np.linspace(-1, 1, G); S = {}
for key, hs in heads.items():
    K, noise = int(key[1]), int(key[-1]); h = hs[0]; w0 = flat(h).clone(); gen = torch.Generator().manual_seed(7)
    d1, d2 = fn_direction(h, gen), fn_direction(h, gen); ytr_eval = T(labels(noise)); yte_t = T(yte)
    TR = np.zeros((G, G)); TE = np.zeros((G, G))
    with torch.no_grad():
        for i, a in enumerate(al):
            for j, b in enumerate(al):
                setflat(h, w0 + a * d1 + b * d2); TR[i, j] = float(loss_of(h, f1, ytr_eval)); TE[i, j] = float(loss_of(h, fte, yte_t))
    setflat(h, w0); S[key] = (TR, TE)
    c = G // 2; R['B'][key]['slice_train_center'] = float(TR[c, c]); R['B'][key]['slice_train_edge_mean'] = float(np.mean([TR[0, c], TR[-1, c], TR[c, 0], TR[c, -1]]))
    R['B'][key]['slice_test_center'] = float(TE[c, c]); R['B'][key]['slice_test_edge_mean'] = float(np.mean([TE[0, c], TE[-1, c], TE[c, 0], TE[c, -1]]))
fig, ax = plt.subplots(2, 4, figsize=(15, 6.8)); keys = list(S)
for c, key in enumerate(keys):
    for r, nm in enumerate(['train loss', 'test loss']):
        M = S[key][r]; im = ax[r, c].contourf(al, al, np.log10(M + 1e-4), 20, cmap='viridis'); ax[r, c].set_title(f'{key}: log10 {nm}', fontsize=9); plt.colorbar(im, ax=ax[r, c], fraction=0.046)
plt.tight_layout(); plt.savefig('results2/land_slices.png', dpi=120); plt.close()

# interpolation between the two independently trained solutions (same setting)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8)); al1 = np.linspace(-0.25, 1.25, 61)
for key, hs in heads.items():
    K, noise = int(key[1]), int(key[-1]); a, b = flat(hs[0]).clone(), flat(hs[1]).clone(); tmp = NodeHead(f1.shape[1], 64); ytr_eval = T(labels(noise)); yte_t = T(yte); trl, tel = [], []
    with torch.no_grad():
        for t in al1: setflat(tmp, (1 - t) * a + t * b); trl.append(float(loss_of(tmp, f1, ytr_eval))); tel.append(float(loss_of(tmp, fte, yte_t)))
    i0, i1 = int(np.argmin(abs(al1 - 0))), int(np.argmin(abs(al1 - 1))); lin = (1 - al1[i0:i1 + 1]) * trl[i0] + al1[i0:i1 + 1] * trl[i1]
    R['B'][key]['interp_barrier_train'] = float(max(np.array(trl[i0:i1 + 1]) - lin)); R['B'][key]['interp_barrier_test'] = float(max(np.array(tel[i0:i1 + 1]) - ((1 - al1[i0:i1 + 1]) * tel[i0] + al1[i0:i1 + 1] * tel[i1])))
    ax[0].plot(al1, trl, label=key); ax[1].plot(al1, tel, label=key)
for a_, t in zip(ax, ['train loss', 'test loss']): a_.set_yscale('log'); a_.set_xlabel('interpolation coefficient between two independent solutions'); a_.set_title(t); a_.legend(fontsize=7)
plt.tight_layout(); plt.savefig('results2/land_interp.png', dpi=120); plt.close()

# epoch-wise plot
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for key, v in R['A'].items():
    e = [r['epoch'] for r in v['curve']]; ax[0].plot(e, [r['test_err'] for r in v['curve']], label=key); ax[1].plot(e, [r['train_err'] for r in v['curve']], label=key)
ax[0].set_title('test error vs epoch (h=256)'); ax[1].set_title('train error vs epoch (on its own, possibly noisy, labels)')
for a_ in ax: a_.set_xscale('log'); a_.set_xlabel('epoch'); a_.legend(fontsize=7)
plt.tight_layout(); plt.savefig('results2/land_epochwise.png', dpi=120); plt.close()
json.dump(R, open('results2/land.json', 'w'), indent=1); print('DONE')
