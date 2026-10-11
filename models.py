"""Small CNN trunk (shared feature extractor) + tiny MLP heads (tree nodes)."""
import torch
import torch.nn as nn


class Trunk(nn.Module):
    """Small CNN feature extractor, shared across all tree nodes.
    32x32x3 -> global-average-pooled feature vector of size feat_dim.
    """
    def __init__(self, c1=16, c2=32, feat_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, c1, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.pool = nn.MaxPool2d(2)
        self.proj = nn.Linear(c2, feat_dim)
        self.feat_dim = feat_dim

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)                 # 16x16
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)                 # 8x8
        x = x.mean(dim=[2, 3])           # global average pool -> (B, c2)
        x = self.proj(x)                 # -> (B, feat_dim)
        return x

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class NodeHead(nn.Module):
    """Tiny MLP sitting on trunk features. This IS the 'small neural network'
    at each tree node. hidden=0 means a single linear layer (logistic regression)."""
    def __init__(self, feat_dim, hidden=16):
        super().__init__()
        self.hidden = hidden
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(feat_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.net = nn.Linear(feat_dim, 1)

    def forward(self, feats):
        return self.net(feats).squeeze(-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def forward_multi(trunk, x):
    """Multi-level ('hypercolumn') features: GAP of conv1, GAP of conv2, and the projected feature.
    Lets nodes pick low-level texture/colour cues as well as the final embedding."""
    import torch
    a = torch.relu(trunk.bn1(trunk.conv1(x)))
    f1 = a.mean(dim=[2, 3])
    b = torch.relu(trunk.bn2(trunk.conv2(trunk.pool(a))))
    f2 = b.mean(dim=[2, 3])
    f3 = trunk.proj(trunk.pool(b).mean(dim=[2, 3]))
    return torch.cat([f1, f2, f3], dim=1)
