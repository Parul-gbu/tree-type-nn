"""Train the shared CNN trunk jointly with a root classifier head, then freeze it.
Trunk is trained ONCE per dataset and reused (frozen) by every node in the tree --
this is the 'features common across all nodes' approach from the assignment hint."""
import numpy as np
import torch
import torch.nn as nn
from models import Trunk, NodeHead


def train_trunk(X_train, y_train, X_test, y_test, c1=16, c2=32, feat_dim=32,
                 epochs=40, lr=1e-3, batch_size=64, verbose=True, seed=0):
    torch.manual_seed(seed)
    trunk = Trunk(c1=c1, c2=c2, feat_dim=feat_dim)
    head = NodeHead(feat_dim, hidden=16)
    opt = torch.optim.Adam(list(trunk.parameters()) + list(head.parameters()), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    Xtr = torch.as_tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    ytr = torch.as_tensor(y_train, dtype=torch.float32)
    Xte = torch.as_tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    yte = torch.as_tensor(y_test, dtype=torch.float32)

    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = head(trunk(Xtr[idx]))
            loss = loss_fn(logits, ytr[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            with torch.no_grad():
                tr_acc = ((head(trunk(Xtr)) >= 0).float() == ytr).float().mean().item()
                te_acc = ((head(trunk(Xte)) >= 0).float() == yte).float().mean().item()
            print(f"  trunk ep{ep}: loss={total_loss/n:.4f} train_acc={tr_acc:.3f} test_acc={te_acc:.3f}")

    trunk.eval()
    for p in trunk.parameters():
        p.requires_grad_(False)
    return trunk


def train_trunk_multiclass(X_train, y_train, X_test, y_test, n_classes, c1=16, c2=32, feat_dim=32,
                            epochs=40, lr=1e-3, batch_size=64, verbose=True, seed=0):
    """Same idea as train_trunk, but the pretext task used to shape the shared trunk
    is full multiclass softmax classification (used for CIFAR-10's 10 classes)."""
    torch.manual_seed(seed)
    trunk = Trunk(c1=c1, c2=c2, feat_dim=feat_dim)
    clf_head = nn.Linear(feat_dim, n_classes)
    opt = torch.optim.Adam(list(trunk.parameters()) + list(clf_head.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    Xtr = torch.as_tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    ytr = torch.as_tensor(y_train, dtype=torch.long)
    Xte = torch.as_tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    yte = torch.as_tensor(y_test, dtype=torch.long)

    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = clf_head(trunk(Xtr[idx]))
            loss = loss_fn(logits, ytr[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            with torch.no_grad():
                tr_acc = (clf_head(trunk(Xtr)).argmax(1) == ytr).float().mean().item()
                te_acc = (clf_head(trunk(Xte)).argmax(1) == yte).float().mean().item()
            print(f"  trunk ep{ep}: loss={total_loss/n:.4f} train_acc={tr_acc:.3f} test_acc={te_acc:.3f}")

    trunk.eval()
    for p in trunk.parameters():
        p.requires_grad_(False)
    return trunk


def extract_features(trunk, X):
    Xt = torch.as_tensor(X, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    with torch.no_grad():
        feats = trunk(Xt)
    return feats
