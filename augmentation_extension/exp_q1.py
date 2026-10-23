"""Q1: effect of augmentation (trunk and/or tree) on an overfit tree. 3 seeds, trunk (16,32,32), full training set."""
import numpy as np, torch, json, time
from sklearn.model_selection import train_test_split
from aug_lib import to_t, train_e2e, logits, view_features
from tree import TreeNode

X = np.load('data_cache/catsdogs_X.npy'); y = np.load('data_cache/catsdogs_y.npy')
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
np.savez('results2/split.npz', Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte)
Xtr_t, Xte_t = to_t(Xtr), to_t(Xte)
K, EPOCHS = 4, 60
TREE = dict(hidden=64, max_depth=8, min_samples=1, epochs=150, lr=0.05)
def grow(f, yy):
    torch.manual_seed(0); return TreeNode(budget=[200], **TREE).fit(f, yy)
def acc(t, f, yy): return float((t.predict(f) == yy).mean())
R = []
for seed in range(3):
    for aug in [False, True]:
        t0 = time.time()
        trunk, head, _ = train_e2e(Xtr_t, ytr, epochs=EPOCHS, aug=aug, seed=seed)
        torch.save({'trunk': trunk, 'head': head}, f'results2/q1_{int(aug)}_{seed}.pt')
        f1, _ = view_features(trunk, Xtr_t, 1); f4, _ = view_features(trunk, Xtr_t, K, seed); fte, _ = view_features(trunk, Xte_t, 1)
        r = dict(seed=seed, aug_trunk=aug, trunk_secs=time.time() - t0,
                 trunk_head_train=float(((logits(trunk, head, Xtr_t) >= 0) == ytr).mean()), trunk_head_test=float(((logits(trunk, head, Xte_t) >= 0) == yte).mean()))
        for name, f, yy in [('tree_orig', f1, ytr), ('tree_aug', f4, np.tile(ytr, K))]:
            t0 = time.time(); t = grow(f, yy)
            r[name] = dict(train_orig=acc(t, f1, ytr), train_fit=acc(t, f, yy), test=acc(t, fte, yte), nodes=t.n_nodes_, params=t.total_params(), secs=time.time() - t0)
        R.append(r); print(json.dumps(r), flush=True)
        json.dump(R, open('results2/q1.json', 'w'), indent=1)
print('DONE')
