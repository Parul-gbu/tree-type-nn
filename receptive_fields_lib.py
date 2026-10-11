"""Receptive-field / saliency visualisation of tree nodes (gradient of node logit w.r.t. input pixels)."""
import numpy as np, torch, pickle, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def saliency(trunk, head, X, max_n=150):
    x = torch.as_tensor(X[:max_n], dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    x.requires_grad_(True)
    head(trunk(x)).sum().backward()
    g = x.grad.abs().sum(1).mean(0).numpy()      # (32,32) mean |d logit / d pixel|
    return g / (g.max() + 1e-12)

def stats(g):
    p = g / g.sum()
    ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(p.size))      # 1 = perfectly uniform, 0 = single pixel
    centre = float(p[8:24, 8:24].sum())                                 # mass in central 16x16 (25% of the area)
    ys, xs = np.mgrid[0:32, 0:32]
    return dict(entropy=ent, centre_mass=centre, cy=float((p * ys).sum()), cx=float((p * xs).sum()))

def bfs_nodes(root, max_nodes=6):
    out, q = [], [(root, 'root', 0)]
    while q and len(out) < max_nodes:
        n, name, d = q.pop(0); out.append((n, name, d))
        if n.child_A is not None: q.append((n.child_A, name + '.A', d + 1))
        if n.child_B is not None: q.append((n.child_B, name + '.B', d + 1))
    return out

def firing(trunk, node, X):
    x = torch.as_tensor(X, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    with torch.no_grad(): m = (node.head(trunk(x)) >= 0).numpy()
    return X[m] if m.sum() >= 5 else X

