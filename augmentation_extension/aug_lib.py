"""Augmentation + end-to-end trunk training + augmented-view feature extraction (v3 experiments)."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from models import Trunk, NodeHead

def to_t(X):
    return torch.as_tensor(X, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

def augment(x, gen=None):
    """x: (B,3,32,32) in [0,1]. Random horizontal flip, random translation (reflect-pad 4 + crop), brightness/contrast jitter (+-20%)."""
    B = x.shape[0]
    flip = torch.rand(B, generator=gen) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    xp = F.pad(x, (4, 4, 4, 4), mode='reflect')
    i = torch.randint(0, 9, (B,), generator=gen); j = torch.randint(0, 9, (B,), generator=gen)
    out = torch.stack([xp[b, :, i[b]:i[b] + 32, j[b]:j[b] + 32] for b in range(B)])
    br = 1 + (torch.rand(B, 1, 1, 1, generator=gen) - 0.5) * 0.4
    ct = 1 + (torch.rand(B, 1, 1, 1, generator=gen) - 0.5) * 0.4
    m = out.mean(dim=(1, 2, 3), keepdim=True)
    return (((out - m) * ct + m) * br).clamp(0, 1)

def train_e2e(Xtr_t, ytr, c1=16, c2=32, fd=32, epochs=60, lr=3e-3, batch=64, aug=False, seed=0,
              Xte_t=None, record=None, head_hidden=16):
    """Train trunk + pretext head end-to-end. If `record` (list of epochs) and Xte_t are given, test logits are stored at those epochs."""
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(1000 + seed)
    trunk = Trunk(c1, c2, fd); head = NodeHead(fd, head_hidden)
    opt = torch.optim.Adam(list(trunk.parameters()) + list(head.parameters()), lr=lr)
    lf = nn.BCEWithLogitsLoss(); y = torch.as_tensor(ytr, dtype=torch.float32); n = len(Xtr_t)
    rec = {}
    for ep in range(1, epochs + 1):
        trunk.train(); head.train(); perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xtr_t[idx]
            if aug: xb = augment(xb, gen)
            opt.zero_grad(); lf(head(trunk(xb)), y[idx]).backward(); opt.step()
        if record is not None and ep in record:
            trunk.eval(); head.eval()
            with torch.no_grad(): rec[ep] = head(trunk(Xte_t)).numpy().copy()
    trunk.eval(); head.eval()
    for p in list(trunk.parameters()) + list(head.parameters()): p.requires_grad_(False)
    return trunk, head, rec

def logits(trunk, head, X_t):
    with torch.no_grad(): return head(trunk(X_t)).numpy()

def view_features(trunk, X_t, K, seed=0, chunk=1024):
    """Features of K views per image: view 0 = original, views 1..K-1 = random augmentations. Returns (K*N, d), index of source image."""
    gen = torch.Generator().manual_seed(5000 + seed); N = len(X_t); out = []
    with torch.no_grad():
        for k in range(K):
            fs = []
            for i in range(0, N, chunk):
                xb = X_t[i:i + chunk]
                if k > 0: xb = augment(xb, gen)
                fs.append(trunk(xb))
            out.append(torch.cat(fs))
    return torch.cat(out), np.tile(np.arange(N), K)
