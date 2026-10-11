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

R = {}
fig_rows = []
# ---- Cats vs Dogs tree ----
d = np.load('results/catsdogs_split.npz'); trunk = torch.load('results/catsdogs_trunk.pt', weights_only=False)
tree = pickle.load(open('results/catsdogs_tree.pkl', 'rb'))
nodes = bfs_nodes(tree, 6); row = []
for n, name, dep in nodes:
    Xf = firing(trunk, n, d['Xtr']); g = saliency(trunk, n.head, Xf)
    s = stats(g); s.update(depth=dep, name=name, n_fire=int(len(Xf))); R.setdefault('catsdogs', []).append(s)
    row.append((f"CD {name}\nfires on {len(Xf)}", Xf.mean(0).astype(np.uint8), g))
fig_rows.append(row)

# ---- CIFAR: root head of four class trees ----
c = np.load('results/cifar_split.npz'); trunk_c = torch.load('results/cifar_trunk.pt', weights_only=False)
ovr = pickle.load(open('results/cifar_ovr.pkl', 'rb'))
names = open('data_cache/cifar_classes.txt').read().split('\n'); row = []
for k in [0, 3, 7, 8]:
    n = ovr.trees[k]
    Xf = firing(trunk_c, n, c['Xtr']); g = saliency(trunk_c, n.head, Xf)
    s = stats(g); s.update(depth=0, name=names[k] + ' root', n_fire=int(len(Xf))); R.setdefault('cifar', []).append(s)
    row.append((f"{names[k]} root\nfires on {len(Xf)}", Xf.mean(0).astype(np.uint8), g))
# deepest node of the cat tree
deep = bfs_nodes(ovr.trees[3], 8)[-1]
Xf = firing(trunk_c, deep[0], c['Xtr']); g = saliency(trunk_c, deep[0].head, Xf)
s = stats(g); s.update(depth=deep[2], name='cat ' + deep[1], n_fire=int(len(Xf))); R['cifar'].append(s)
row.append((f"cat {deep[1]} (deep)\nfires on {len(Xf)}", Xf.mean(0).astype(np.uint8), g))
fig_rows.append(row)

for key, ttl in [('catsdogs', 'Cats vs Dogs'), ('cifar', 'CIFAR-10')]:
    print(ttl); [print('  ', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}) for s in R[key]]

ncol = max(len(r) for r in fig_rows)
fig, ax = plt.subplots(4, ncol, figsize=(2.3 * ncol, 9.4))
for ri, row in enumerate(fig_rows):
    for ci in range(ncol):
        a0, a1 = ax[2 * ri, ci], ax[2 * ri + 1, ci]
        if ci < len(row):
            t, img, g = row[ci]
            a0.imshow(img); a0.set_title(t, fontsize=7); a1.imshow(g, cmap='inferno')
        a0.axis('off'); a1.axis('off')
ax[0, 0].text(-12, 16, 'mean image\n(firing set)', fontsize=7, rotation=90, va='center'); 
plt.suptitle('Node receptive fields: mean firing image (top) and gradient saliency (bottom)\nrows 1-2: Cats vs Dogs tree (BFS order)   rows 3-4: CIFAR-10 class trees', fontsize=9)
plt.tight_layout(); plt.savefig('results/receptive_fields.png', dpi=130); plt.close()

# ---- trunk first-layer filters + theoretical RF ----
w = trunk.conv1.weight.detach().numpy()
fig, ax = plt.subplots(3, 8, figsize=(8, 3.4))
for i, a in enumerate(ax.flat):
    f = w[i].transpose(1, 2, 0); f = (f - f.min()) / (f.max() - f.min() + 1e-9)
    a.imshow(f); a.axis('off')
plt.suptitle('Cats-vs-Dogs trunk: 24 first-layer 3x3 filters', fontsize=9); plt.tight_layout()
plt.savefig('results/trunk_filters.png', dpi=130); plt.close()
R['trunk_rf'] = dict(conv1=3, after_pool1=4, conv2=8, after_pool2=10, note='global average pool makes node-level RF the whole 32x32 image')
json.dump(R, open('results/receptive_fields.json', 'w'), indent=1)
print('DONE')
