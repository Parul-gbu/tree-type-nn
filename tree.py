"""Tree-type network, v2: same C1-C4 error-correction idea as the LP/SVM version,
but every node is now a tiny neural net (NodeHead) sitting on shared CNN trunk features."""
import numpy as np
import torch
import torch.nn as nn
from models import NodeHead


def train_head(feats, y, hidden=16, epochs=300, lr=0.05, weight_decay=0.0,
                verbose=False, early_stop_acc=1.0):
    """Train one node's tiny MLP head on (frozen) trunk features until it fits
    the given binary labels (or hits `epochs`). feats: (N, feat_dim) tensor. y: (N,) in {0,1}."""
    feat_dim = feats.shape[1]
    head = NodeHead(feat_dim, hidden=hidden)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    yt = torch.as_tensor(y, dtype=torch.float32)
    for ep in range(epochs):
        opt.zero_grad()
        logits = head(feats)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            with torch.no_grad():
                acc = ((logits >= 0).float() == yt).float().mean().item()
            if verbose:
                print(f"    ep{ep}: loss={loss.item():.4f} acc={acc:.3f}")
            if acc >= early_stop_acc:
                break
    return head


class TreeNode:
    """A node holds a NodeHead trained on the parent's (masked) feature subset.
    Children A/B correct for C3/C4 errors, exactly like the earlier LP/SVM tree,
    just with the LP/SVM swapped for a tiny trained neural net."""
    def __init__(self, hidden=16, max_depth=4, min_samples=10, depth=0, epochs=300, lr=0.05, budget=None):
        self.hidden = hidden
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.depth = depth
        self.epochs = epochs
        self.lr = lr
        self.budget = budget  # shared [remaining_nodes] list, caps total tree size
        self.head = None
        self.wA = 0.0
        self.wB = 0.0
        self.child_A = None
        self.child_B = None
        self.n_nodes_ = 1

    def fit(self, feats, y, verbose=False):
        self.head = train_head(feats, y, hidden=self.hidden, epochs=self.epochs, lr=self.lr, verbose=verbose)
        with torch.no_grad():
            net = self.head(feats).numpy()
        pred = (net >= 0).astype(int)

        C1 = (y == 0) & (pred == 0)
        C2 = (y == 1) & (pred == 1)
        C3 = (y == 1) & (pred == 0)
        C4 = (y == 0) & (pred == 1)

        if self.depth >= self.max_depth:
            return self

        if C3.sum() >= self.min_samples:
            mask_A = ~C2
            yA = C3[mask_A].astype(int)
            if len(np.unique(yA)) > 1 and self._can_grow():
                self.child_A = TreeNode(self.hidden, self.max_depth, self.min_samples,
                                         self.depth + 1, self.epochs, self.lr, self.budget)
                self.child_A.fit(feats[mask_A], yA, verbose=verbose)
                self.n_nodes_ += self.child_A.n_nodes_
            self.wA = max(1e-3, float(np.max(1 - net[C3])))
        elif C3.sum() > 0:
            self.wA = max(1e-3, float(np.max(1 - net[C3])))

        if C4.sum() >= self.min_samples:
            mask_B = ~C1
            yB = C4[mask_B].astype(int)
            if len(np.unique(yB)) > 1 and self._can_grow():
                self.child_B = TreeNode(self.hidden, self.max_depth, self.min_samples,
                                         self.depth + 1, self.epochs, self.lr, self.budget)
                self.child_B.fit(feats[mask_B], yB, verbose=verbose)
                self.n_nodes_ += self.child_B.n_nodes_
            self.wB = min(-1e-3, float(np.min(-1 - net[C4])))
        elif C4.sum() > 0:
            self.wB = min(-1e-3, float(np.min(-1 - net[C4])))

        return self

    def _can_grow(self):
        if self.budget is None:
            return True
        if self.budget[0] <= 0:
            return False
        self.budget[0] -= 1
        return True

    def total_net(self, feats):
        with torch.no_grad():
            net = self.head(feats).numpy()
        if self.child_A is not None:
            net = net + self.wA * (self.child_A.total_net(feats) >= 0).astype(float)
        if self.child_B is not None:
            net = net + self.wB * (self.child_B.total_net(feats) >= 0).astype(float)
        return net

    def predict(self, feats):
        return (self.total_net(feats) >= 0).astype(int)

    def total_params(self):
        n = self.head.n_params()
        if self.child_A is not None:
            n += self.child_A.total_params()
        if self.child_B is not None:
            n += self.child_B.total_params()
        return n


class OneVsRestTree:
    """Multiclass wrapper: one TreeNode per class (same scheme as the earlier assignment)."""
    def __init__(self, n_classes, **tree_kwargs):
        self.n_classes = n_classes
        self.tree_kwargs = tree_kwargs
        self.trees = {}

    def fit(self, feats, y, verbose=True):
        for c in range(self.n_classes):
            yc = (y == c).astype(int)
            node = TreeNode(**self.tree_kwargs)
            node.fit(feats, yc, verbose=False)
            self.trees[c] = node
            if verbose:
                print(f"  class {c}: {node.n_nodes_} nodes, {node.total_params()} params")
        return self

    def decision_matrix(self, feats):
        return np.stack([self.trees[c].total_net(feats) for c in range(self.n_classes)], axis=1)

    def predict(self, feats):
        return self.decision_matrix(feats).argmax(axis=1)

    def total_params(self):
        return sum(t.total_params() for t in self.trees.values())
