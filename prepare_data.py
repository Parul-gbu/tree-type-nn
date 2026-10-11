"""Load CatsVsDogs + CIFAR-10 subsets, resize to 32x32, cache as .npy arrays."""
import numpy as np
from PIL import Image
import glob, os, time

OUT = '/home/claude/project/data_cache'
os.makedirs(OUT, exist_ok=True)

def load_folder(pattern, label, size=(32, 32), limit=None):
    files = sorted(glob.glob(pattern))
    if limit:
        files = files[:limit]
    imgs = []
    for f in files:
        try:
            im = Image.open(f).convert('RGB').resize(size)
            imgs.append(np.array(im, dtype=np.uint8))
        except Exception as e:
            print('skip', f, e)
    return np.stack(imgs), np.full(len(imgs), label, dtype=np.int64)

t0 = time.time()
Xc, yc = load_folder('/home/claude/catsdogs/CatsandDogs/Cats/*.jpg', 0)
Xd, yd = load_folder('/home/claude/catsdogs/CatsandDogs/Dogs/*.jpg', 1)
print('cats', Xc.shape, 'dogs', Xd.shape, f'{time.time()-t0:.1f}s')

X_cd = np.concatenate([Xc, Xd], axis=0)
y_cd = np.concatenate([yc, yd], axis=0)
np.save(f'{OUT}/catsdogs_X.npy', X_cd)
np.save(f'{OUT}/catsdogs_y.npy', y_cd)
print('saved catsdogs:', X_cd.shape, y_cd.shape)

# CIFAR-10: already 32x32 PNGs, extracted earlier at /home/claude/cifar_full/train
import pandas as pd
labels_df = pd.read_csv('/mnt/user-data/uploads/trainLabels.csv')
classes = sorted(labels_df['label'].unique())
cls_to_idx = {c: i for i, c in enumerate(classes)}
print('CIFAR classes:', cls_to_idx)

PER_CLASS = 400  # keep total dataset small enough for CPU-only compute
rng = np.random.default_rng(0)
rows = []
for c in classes:
    ids = labels_df[labels_df['label'] == c]['id'].values
    chosen = rng.choice(ids, min(PER_CLASS, len(ids)), replace=False)
    rows.extend([(i, cls_to_idx[c]) for i in chosen])

t0 = time.time()
imgs, labels = [], []
for i, lab in rows:
    im = Image.open(f'/home/claude/cifar_full/train/{i}.png').convert('RGB')
    imgs.append(np.array(im, dtype=np.uint8))
    labels.append(lab)
X_cifar = np.stack(imgs)
y_cifar = np.array(labels, dtype=np.int64)
print('cifar subset', X_cifar.shape, f'{time.time()-t0:.1f}s')

np.save(f'{OUT}/cifar_X.npy', X_cifar)
np.save(f'{OUT}/cifar_y.npy', y_cifar)
with open(f'{OUT}/cifar_classes.txt', 'w') as f:
    f.write('\n'.join(classes))
print('saved cifar:', X_cifar.shape, y_cifar.shape)
