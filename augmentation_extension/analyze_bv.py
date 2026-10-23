"""Q2 + Q3 analysis from results2/bv.pkl: Domingos 0-1 bias/variance (noise-free labels), ensembles with majority vote."""
import numpy as np, pickle, json, itertools
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
res = pickle.load(open('results2/bv.pkl', 'rb')); y = res['yte']; E, NS = res['E'], res['NS']

def decomp(P):
    """P: (B, Ntest) 0/1 predictions from B independent replicates. Returns error, bias, variance, var on unbiased pts, var on biased pts."""
    main = (P.mean(0) >= 0.5).astype(int); bias_pt = (main != y).astype(float); var_pt = (P != main).mean(0)
    err = float((P != y).mean()); ub = bias_pt == 0
    return dict(error=err, bias=float(bias_pt.mean()), variance=float(var_pt.mean()), var_unbiased=float(var_pt[ub].sum() / len(y)), var_biased=float(var_pt[~ub].sum() / len(y)))

def vote(nets):  # nets: (M, Ntest) real-valued net inputs; majority vote of hard labels (odd M -> no ties)
    return ((nets >= 0).mean(0) >= 0.5).astype(int)

out = {'vs_n': {}, 'vs_epoch': {}, 'ensemble': {}}
for aug in [False, True]:
    for n in NS:
        r = res['runs'].get((aug, n))
        if r is None: continue
        P = (r['tree_net'] >= 0).astype(int); dd = decomp(P); dd['train_acc_mean'] = float(np.mean(r['train_acc'])); dd['nodes_mean'] = float(np.mean(r['nodes']))
        Pf = (r['logits'][:, -1, :] >= 0).astype(int); dd['trunkhead'] = decomp(Pf)
        out['vs_n'][f'aug{int(aug)}_n{n}'] = dd
        out['vs_epoch'][f'aug{int(aug)}_n{n}'] = [dict(epoch=e, **decomp((r['logits'][:, i, :] >= 0).astype(int))) for i, e in enumerate(E)]
        # independent-trunk ensemble (each replicate = different subset, trunk seed, tree): all odd-size combos of the B members
        B = P.shape[0]; ens = {}
        for M in [1, 3, 5]:
            if M > B: continue
            accs = [float((vote(r['tree_net'][list(c)]) == y).mean()) for c in itertools.combinations(range(B), M)]; ens[M] = (float(np.mean(accs)), float(np.std(accs)))
        out['ensemble'][f'indep_aug{int(aug)}_n{n}'] = ens
for aug, nets in res['shared'].items():
    out['ensemble'][f'shared_aug{int(aug)}'] = {M: float(np.mean([(vote(nets[list(c)]) == y).mean() for c in itertools.combinations(range(len(nets)), M)])) for M in [1, 3, 5] if M <= len(nets)}
json.dump(out, open('results2/bv_summary.json', 'w'), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for aug, c in [(False, 'C0'), (True, 'C1')]:
    ks = [k for k in out['vs_n'] if k.startswith(f'aug{int(aug)}')]; ns = [int(k.split('_n')[1]) for k in ks]; lab = 'aug' if aug else 'no aug'
    for a, key in zip(ax, ['error', 'bias', 'variance']): a.plot(ns, [out['vs_n'][k][key] for k in ks], 'o-', c=c, label=lab)
for a, t in zip(ax, ['test error (single tree)', 'bias (0-1 loss)', 'variance (0-1 loss)']): a.set_xscale('log'); a.set_xlabel('training images n'); a.set_title(t); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig('results2/bv_vs_n.png', dpi=120); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8)); n = NS[-1]
for aug, c in [(False, 'C0'), (True, 'C1')]:
    k = f'aug{int(aug)}_n{n}'
    if k not in out['vs_epoch']: continue
    for a, key in zip(ax, ['error', 'bias', 'variance']): a.plot(E, [r[key] for r in out['vs_epoch'][k]], 'o-', c=c, label='aug' if aug else 'no aug')
for a, t in zip(ax, [f'test error vs trunk epoch (n={n})', 'bias', 'variance']): a.set_xscale('log'); a.set_xlabel('trunk training epoch'); a.set_title(t); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig('results2/bv_vs_epoch.png', dpi=120); plt.close()
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for key, v in out['ensemble'].items():
    if key.startswith('shared'): ax.plot(list(v), list(v.values()), 's--', label=key)
    elif key.endswith(f'n{NS[-1]}'): ax.errorbar(list(v), [x[0] for x in v.values()], yerr=[x[1] for x in v.values()], marker='o', capsize=2, label=key)
ax.set_xlabel('number of trees M (majority vote)'); ax.set_ylabel('test accuracy'); ax.set_xticks([1, 3, 5]); ax.legend(fontsize=7); plt.tight_layout(); plt.savefig('results2/ensemble.png', dpi=120)
print(json.dumps(out['vs_n'], indent=1)[:1800]); print(json.dumps(out['ensemble'], indent=1))
