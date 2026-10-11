"""Control for the receptive-field observation: is a deep node's speckled saliency a property of the node,
or just an artefact of averaging over only ~8-10 images? Compare against the ROOT averaged over equally few images."""
import numpy as np, torch, pickle, json
from receptive_fields_lib import saliency, bfs_nodes, firing

def speckle(g):
    """fraction of saliency-map energy in the high-pass residual (map minus 3x3 box blur)"""
    gp = np.pad(g, 1, mode='edge'); blur = sum(gp[i:i + 32, j:j + 32] for i in range(3) for j in range(3)) / 9
    hp = g - blur; return float((hp ** 2).sum() / ((g - g.mean()) ** 2).sum())

d = np.load('results/catsdogs_split.npz'); trunk = torch.load('results/catsdogs_trunk.pt', weights_only=False)
tree = pickle.load(open('results/catsdogs_tree.pkl', 'rb')); rng = np.random.default_rng(0)
nodes = bfs_nodes(tree, 6); root = nodes[0][0]
Xf_root = firing(trunk, root, d['Xtr']); out = {}
for n in [8, 10, 40, 150, 900]:
    vals = [speckle(saliency(trunk, root.head, Xf_root[rng.choice(len(Xf_root), min(n, len(Xf_root)), replace=False)], max_n=n)) for _ in range(20)]
    out[f'root_n{n}'] = (float(np.mean(vals)), float(np.std(vals))); print(f'root, averaged over {n:4d} images: speckle = {np.mean(vals):.3f} +- {np.std(vals):.3f}')
for node, name, dep in nodes[1:]:
    Xf = firing(trunk, node, d['Xtr']); g = saliency(trunk, node.head, Xf, max_n=150)
    out[name] = (speckle(g), len(Xf)); print(f'{name:10s} (n={len(Xf):4d}): speckle = {speckle(g):.3f}')
json.dump(out, open('results/rf_control.json', 'w'), indent=1)
