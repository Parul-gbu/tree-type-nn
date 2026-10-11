def count_nodes(n):
    return 1 + (count_nodes(n.child_A) if n.child_A else 0) + (count_nodes(n.child_B) if n.child_B else 0)

def prune(node, root, feats, y, base_acc):
    """Greedy bottom-up pruning: drop a child subtree if train accuracy of the whole tree does not fall."""
    for attr in ['child_A', 'child_B']:
        ch = getattr(node, attr)
        if ch is None: continue
        prune(ch, root, feats, y, base_acc)
        setattr(node, attr, None)
        if (root.predict(feats) == y).mean() < base_acc:
            setattr(node, attr, ch)
    return node

def tree_params(n):  # recompute after pruning
    return n.head.n_params() + (tree_params(n.child_A) if n.child_A else 0) + (tree_params(n.child_B) if n.child_B else 0)

