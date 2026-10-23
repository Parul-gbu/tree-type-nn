from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from PIL import Image as PI
ss = getSampleStyleSheet()
B = ParagraphStyle('B', parent=ss['Normal'], fontSize=9.3, leading=12.6, spaceAfter=4)
H1 = ParagraphStyle('H1', parent=ss['Heading1'], fontSize=12.5, spaceBefore=7, spaceAfter=3)
C = ParagraphStyle('C', parent=B, fontSize=7.8, leading=9.5, textColor=colors.HexColor('#444444'))
P = lambda t, s=B: Paragraph(t, s)
def fig(p, w):
    iw, ih = PI.open(p).size; return Image(p, width=w * inch, height=w * inch * ih / iw)
def tbl(d, w, fs=8):
    t = Table(d, colWidths=w, repeatRows=1)
    t.setStyle(TableStyle([('FONTSIZE', (0,0), (-1,-1), fs), ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#dddddd')), ('GRID', (0,0), (-1,-1), 0.4, colors.grey), ('ALIGN', (1,0), (-1,-1), 'CENTER')])); return t
R = 'results/'; s = []
s += [P('Tree-Type Network with Augmentation: Overfitting, Ensembles, Bias-Variance and Loss Landscape', ParagraphStyle('T', parent=ss['Title'], fontSize=15, leading=18)),
      P('Extension of the earlier tree-of-neural-network-nodes assignment. Dataset: Cats vs Dogs, 32x32 (1,902 train / 476 test). Code: https://github.com/Parul-gbu/tree-type-nn (folder augmentation_extension/)', ParagraphStyle('S', parent=B, alignment=1))]
s += [P('1. Setup and innovations', H1),
 P('<b>Model.</b> As before: a small CNN trunk (16/32/32 channels, 6,240 parameters) and a tree whose nodes are tiny MLP heads that correct the parent\'s two error types. '
   '<b>Augmentation</b> = random horizontal flip, random translation of up to 4 pixels (reflect-pad and crop) and brightness/contrast jitter of +/-20%. No rotation or scale. It is used (i) on the fly while training the trunk, and (ii) to build K augmented views of each training image, '
   'on which the tree is trained (all views of an image share its label).'),
 P('<b>What is new here.</b> (a) <i>View-level augmentation inside the tree</i>: the tree is fitted to K views per image, and bootstrap resampling for ensembles is done by image, keeping all views. '
   '(b) <i>Bias/variance without leakage</i>: for every replicate the trunk and tree are retrained from scratch on a different random subset, so trunk variance is included. '
   '(c) <i>Two ensemble types</i>: shared trunk with bootstrapped trees, and independent trunks; majority vote of hard labels over odd numbers of trees. '
   '(d) <i>Landscape probes</i> on node heads: top Hessian eigenvalue, filter-normalised 2D slices, interpolation between independent solutions. '
   '(e) A test of the hypothesis that augmentation shifts the double-descent peak to the right in proportion to the number of views.')]
s += [P('2. Q1: augmentation, overfitting and double descent', H1),
 tbl([['Configuration (3 seeds)', 'Train acc (orig. images)', 'Test acc', 'Nodes'],
      ['A. plain trunk, tree on clean features', '96.0%', '75.8 +/- 0.5', '21'], ['B. plain trunk, tree on 4 aug views', '92.4%', '75.1 +/- 0.9', '43'],
      ['C. aug trunk, tree on clean features', '93.0%', '74.6 +/- 0.7', '27'], ['D. aug trunk, tree on 4 aug views', '84.1%', '77.4 +/- 0.6', '41']], [2.9*inch, 1.5*inch, 1.0*inch, 0.6*inch]),
 Spacer(1, 3),
 P('<b>Result.</b> Only the full combination (D) improves on the baseline: +1.6 points on average, and positive in all three paired seeds (+2.5, +1.3, +1.1). Augmenting only the trunk (C) or only the tree (B) did not help (-1.2, -0.7), so the gain is not additive and, with 3 seeds and a 476-image test set, is weak evidence. '
   'D also does <b>not</b> overfit: it fits only about 84% of its augmented training set at this tree size. To test genuine interpolation we grew bigger trees (width 128, depth 12; 2 seeds). The tree on clean features interpolates (100% train) but drops to 70.6% test, '
   'about 3.7 points below the smaller tree C on the same trunks (74.3% on those seeds). The tree on 4 augmented views reaches 93-95% train and 72.6% test. So forcing interpolation costs accuracy, and augmentation limits how far the tree can memorise.'),
 fig(R + 'dd_aug.png', 6.0),
 P('Figure 1. Head-width sweep with 20% label noise on frozen features (K = views per image). Points up to width 1024 use 2 seeds; wider points use 1 seed.', C),
 P('<b>Double descent.</b> Without augmentation, test error rises from 24.9% (width 4) to a peak of 31.5% (width 128), then falls to 28.6-30.7% at widths 1024-4096: a mild second descent of 1-3 points, partly single-seed. '
   'With 6 views, training error never gets close to zero (18% at width 1024, vs 2.7% without views) and the peak moves from width 128 to about 1024 (31.1%), then 30.0% at 2048. The shift is roughly 8x, in the same range as the 6x data multiplier, as expected if the peak tracks the number of effective samples. '
   'Evidence is suggestive, not conclusive: coarse width grid, one or two seeds. Epoch-wise (Figure 4, right) we saw <b>no</b> second descent in 4000 epochs: test error rises from about 23% to 30% (classical overfitting; early stopping is best), and augmentation lowers the late-epoch level only slightly (about 29% vs 30%).')]
s += [P('3. Q2: ensembles with majority voting', H1), fig(R + 'ensemble.png', 3.4),
 P('Figure 2. Test accuracy vs number of trees (n = 1600). Independent-trunk curves show mean and spread over all subsets of the 5 trained members.', C),
 tbl([['n', 'plain: 1 / 3 / 5 trees', 'aug: 1 / 3 / 5 trees'], ['200', '62.6 / 65.4 / 67.4', '63.0 / 65.3 / 66.6'], ['800', '70.6 / 73.6 / 74.6', '69.8 / 73.0 / 74.8'], ['1600', '73.8 / 75.9 / 75.8', '74.2 / 76.1 / 76.3'], ['1600 shared trunk', '71.6 / 73.4 / 73.1', '71.9 / 73.8 / 75.2']], [1.5*inch, 1.8*inch, 1.8*inch]),
 Spacer(1, 3),
 P('Ensembling gives about +2 points at n=1600 and up to +4.8 at n=200, saturating by 3 to 5 trees. Independent trunks beat a shared trunk, because the shared trunk\'s variance is not averaged away (trunk-plus-head predictions alone have variance 0.25-0.29). '
   'Augmentation adds little inside the ensemble: differences are within about +/-1 point, except the shared-trunk M=5 case (+2.1, one run).')]
s += [P('4. Q3: bias and variance', H1), fig(R + 'bv_vs_n.png', 6.2),
 P('Figure 3. Domingos 0-1 decomposition of a single tree vs training-set size (5 replicates per point, trunk retrained each time).', C),
 P('Error is dominated by bias (0.33 to 0.24 as n goes 200 to 1600) with variance falling 0.21 to 0.13. Because error = bias + (variance on unbiased points) - (variance on biased points), the net variance term (0.047, 0.043, 0.039, 0.021 for n = 200 to 1600, plain trees) is what ensembling can remove. '
   'The 5-tree gains in Section 3 equal these numbers almost exactly (+4.8, +4.3, +4.0, +2.0), but note this is partly by construction: with the same 5 replicates, the 5-tree vote <i>is</i> the "main prediction" of the decomposition, so its error equals the bias. '
   'The independent check is that 3-tree subsets already recover most of the gain. Augmentation did <b>not</b> reduce tree variance consistently (lower at n=1600: 0.118 vs 0.131; higher at n=400 and 800). '
   'For the end-to-end trunk-plus-head classifier, augmentation lowered test error at all four sizes (by 4.3, 3.9, 1.5, 2.7 points for n = 1600, 400, 800, 200), but the tree on top removes this difference. '
   'Across trunk epochs, bias falls (about 0.47 to 0.23-0.27) and variance rises from near zero to 0.25-0.30, the usual early-to-late pattern, but with 5 replicates the curves are too noisy to resolve finer structure.')]
s += [P('5. Q4: loss landscape of node heads', H1), fig(R + 'land_slices.png', 6.3),
 P('Figure 4 (left part). Filter-normalised 2D slices (log10 loss) around trained width-64 heads. Columns: K=1 clean labels, K=1 noisy labels, K=6 clean, K=6 noisy. Top: train loss, bottom: test loss.', C),
 fig(R + 'land_epochwise.png', 4.8), P('Figure 4 (right part). Test and train error vs epoch for width-256 heads.', C),
 tbl([['Setting (width 64)', 'Top Hessian eig. (2 seeds)', 'Test loss', 'Test err (1 seed)', 'Interp. barrier (train loss)'],
      ['K=1, clean labels', '41.7 / 41.5', '1.32', '29.6%', '13.2'], ['K=6, clean labels', '31.6 / 33.4', '0.66', '24.8%', '2.0'],
      ['K=1, 20% noise', '48.6 / 53.9', '0.81', '34.5%', '3.4'], ['K=6, 20% noise', '42.5 / 39.6', '0.58', '29.0%', '2.5']], [1.5*inch, 1.5*inch, 0.8*inch, 1.1*inch, 1.4*inch]),
 Spacer(1, 3),
 P('<b>Observations.</b> (1) Minima are narrow and strongly anisotropic: the slices show a thin valley along one diagonal and steep walls in the orthogonal directions, and the train and test slices have the same shape and aligned minima. '
   '(2) Augmentation flattens the landscape: top eigenvalue is about 22% lower with clean labels and about 20% lower with noise, edge-of-slice loss is lower (22.8 vs 32.2 clean), and test loss halves with clean labels. '
   '(3) The top eigenvalue grows with training (about 3.6 at epoch 40 to 16-47 by epochs 1000-4000 at width 256: progressive sharpening); when the clean data become separable the loss, and its curvature, shrink again (3.4 at epoch 4000, K=1 clean). Power iteration used 25 steps, so values are estimates. '
   '(4) Linear paths between two independently trained solutions have a barrier in every setting, as expected for MLPs, and the barrier is much larger for the plain clean-label solutions (13.2 vs 2.0 train loss); this is one pair of solutions and the barrier is not scale-free, so we treat it as an observation rather than evidence of different basins. '
   '(5) Flatter minima with lower test loss under augmentation is consistent with the flat-minima account of generalisation, but this is correlational, with a single trunk.')]
s += [P('6. Conclusions', H1),
 P('(i) Augmentation gives at best a small test gain for the tree (+1.6 points, 3 seeds; the ensemble and bias-variance runs show differences within about +/-1 point), but a clear gain for the trunk-plus-head classifier and for single heads on frozen features (about 5 points, one seed). '
   '(ii) Forcing a tree to interpolate hurts; augmentation prevents full interpolation and acts as a regulariser. '
   '(iii) We saw a mild, noisy model-wise second descent without augmentation and a right-shifted peak with augmentation; we saw no epoch-wise double descent. '
   '(iv) Majority-vote ensembles of independently trained trees recover most of the net-variance term (about +2 points at n=1600), with 3 to 5 trees enough. '
   '(v) Augmented heads sit in flatter, lower-test-loss minima.')]
s += [P('7. Limitations and future directions', H1),
 P('Limitations: one dataset (Cats vs Dogs; CIFAR-10 not rerun here), 32x32 images, 476 test images (about +/-2 points standard error per accuracy), 3 seeds for Q1 and 5 replicates for bias-variance, single trunk for the sweeps and landscape, simple augmentations. '
   'Future work: more seeds and a finer width grid around the predicted peaks; sweep trunk width and augmentation strength jointly; stronger augmentation (mixup/cutout) to see whether trees can still interpolate; per-branch augmentation and trunks; CIFAR-10 with the same protocol; '
   'sharpness-aware training of node heads; measure the Hessian and barrier for entire trees rather than a single head.')]
SimpleDocTemplate('docs/report_augmentation.pdf', pagesize=letter, leftMargin=0.75*inch, rightMargin=0.75*inch, topMargin=0.6*inch, bottomMargin=0.6*inch, title='Tree-type network with augmentation').build(s)
print('ok')
