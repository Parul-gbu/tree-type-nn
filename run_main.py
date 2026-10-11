"""Main experiment: trunk + tree-of-MLP-heads on Cats-vs-Dogs and CIFAR-10. Saves trunks, trees, metrics."""
import numpy as np, torch, time, json, pickle, os
from sklearn.model_selection import train_test_split
from train_trunk import train_trunk, train_trunk_multiclass, extract_features
from tree import TreeNode, OneVsRestTree, train_head
from models import forward_multi

os.makedirs('results', exist_ok=True)
R = {}

# ---------------- Cats vs Dogs ----------------
X = np.load('data_cache/catsdogs_X.npy'); y = np.load('data_cache/catsdogs_y.npy')
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
np.savez('results/catsdogs_split.npz', Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte)
t0 = time.time()
trunk = train_trunk(Xtr, ytr, Xte, yte, c1=24, c2=48, feat_dim=48, epochs=60, lr=3e-3, verbose=False)
torch.save(trunk, 'results/catsdogs_trunk.pt')
ftr, fte = extract_features(trunk, Xtr), extract_features(trunk, Xte)
print('cd trunk done', time.time() - t0, flush=True)

# root-only baseline
root_only = TreeNode(hidden=64, max_depth=0, epochs=150, lr=0.05).fit(ftr, ytr)
tree = TreeNode(hidden=64, max_depth=8, min_samples=1, epochs=150, lr=0.05, budget=[300]).fit(ftr, ytr)
pickle.dump(tree, open('results/catsdogs_tree.pkl', 'wb'))
R['catsdogs'] = dict(
    trunk_params=trunk.n_params(),
    root_only_train=float((root_only.predict(ftr) == ytr).mean()), root_only_test=float((root_only.predict(fte) == yte).mean()),
    tree_train=float((tree.predict(ftr) == ytr).mean()), tree_test=float((tree.predict(fte) == yte).mean()),
    tree_nodes=tree.n_nodes_, tree_params=tree.total_params())
print('cd', R['catsdogs'], flush=True)

# innovation ablation: multi-level (hypercolumn) features for the heads
with torch.no_grad():
    mtr = forward_multi(trunk, torch.as_tensor(Xtr, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0)
    mte = forward_multi(trunk, torch.as_tensor(Xte, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0)
mt = TreeNode(hidden=64, max_depth=8, min_samples=1, epochs=150, lr=0.05, budget=[300]).fit(mtr, ytr)
mroot = TreeNode(hidden=64, max_depth=0, epochs=150, lr=0.05).fit(mtr, ytr)
R['catsdogs_multilevel'] = dict(
    root_only_train=float((mroot.predict(mtr) == ytr).mean()), root_only_test=float((mroot.predict(mte) == yte).mean()),
    tree_train=float((mt.predict(mtr) == ytr).mean()), tree_test=float((mt.predict(mte) == yte).mean()),
    tree_nodes=mt.n_nodes_, tree_params=mt.total_params())
print('cd multilevel', R['catsdogs_multilevel'], flush=True)

# ---------------- CIFAR-10 ----------------
X = np.load('data_cache/cifar_X.npy'); y = np.load('data_cache/cifar_y.npy')
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
np.savez('results/cifar_split.npz', Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte)
trunk = train_trunk_multiclass(Xtr, ytr, Xte, yte, n_classes=10, c1=24, c2=48, feat_dim=48, epochs=40, lr=3e-3, verbose=False)
torch.save(trunk, 'results/cifar_trunk.pt')
ftr, fte = extract_features(trunk, Xtr), extract_features(trunk, Xte)
flat = OneVsRestTree(n_classes=10, hidden=96, max_depth=0, epochs=200, lr=0.05).fit(ftr, ytr, verbose=False)
ovr = OneVsRestTree(n_classes=10, hidden=96, max_depth=9, min_samples=1, epochs=200, lr=0.05, budget=[60]).fit(ftr, ytr, verbose=False)
pickle.dump(ovr, open('results/cifar_ovr.pkl', 'wb'))
R['cifar'] = dict(
    trunk_params=trunk.n_params(),
    root_only_train=float((flat.predict(ftr) == ytr).mean()), root_only_test=float((flat.predict(fte) == yte).mean()),
    tree_train=float((ovr.predict(ftr) == ytr).mean()), tree_test=float((ovr.predict(fte) == yte).mean()),
    tree_nodes=int(sum(t.n_nodes_ for t in ovr.trees.values())), tree_params=ovr.total_params())
print('cifar', R['cifar'], flush=True)
json.dump(R, open('results/main_results.json', 'w'), indent=1)
print('DONE', flush=True)
