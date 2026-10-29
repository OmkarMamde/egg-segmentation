import os, glob, random, cv2, numpy as np, pandas as pd
from sklearn.cluster import AgglomerativeClustering

IMG_DIR, MASK_DIR = "data", "outputs/sam_masks"
THRESH, SEED, EVAL_FRAC = 0.35, 42, 0.2

# index every image under data/ by its stem (without train_/valid_/test_ prefixes)
import re
def norm(s):
    return re.sub(r"^(train|valid|test)_", "", s)

index = {}
for p in glob.glob(f"{IMG_DIR}/**/*.*", recursive=True):
    if p.lower().endswith((".jpg", ".jpeg", ".png")):
        index.setdefault(norm(os.path.splitext(os.path.basename(p))[0]), p)

stems, paths, missing = [], [], []
for f in sorted(os.listdir(MASK_DIR)):
    s = os.path.splitext(f)[0]
    p = index.get(norm(s))
    if p is None:
        missing.append(s)
    else:
        stems.append(s); paths.append(p)

print(f"matched {len(paths)} images, missing {len(missing)}")
print(f"matched {len(paths)} images, skipped {len(missing)} masks with no image")
if missing:
    print("skipped:", missing)

def feat(p):
    img = cv2.resize(cv2.imread(p), (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(h, h, norm_type=cv2.NORM_L1).flatten().astype(np.float32)

F = [feat(p) for p in paths]
n = len(F)
D = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        D[i, j] = cv2.compareHist(F[i], F[j], cv2.HISTCMP_BHATTACHARYYA)

labels = AgglomerativeClustering(n_clusters=None, metric="precomputed",
                                 linkage="average", distance_threshold=THRESH).fit_predict(D)
labels[labels == 36] = 7   # manual merge: same herb scene was split into two clusters
labels[labels == 23] = 69   # foil wrapper scene
labels[labels == 68] = 1    # white bag scene
labels[labels == 38] = 32   # pink tub scene
labels[(labels == 14) | (labels == 29)] = 17   # steel pot scene, split into 3 clusters

# assign whole clusters to eval until ~20% of images
rng = random.Random(SEED)
clusters = {}
for i, c in enumerate(labels):
    clusters.setdefault(c, []).append(i)
order = list(clusters); rng.shuffle(order)
target = round(EVAL_FRAC * n)
eval_idx = []
for c in order:
    if len(eval_idx) + len(clusters[c]) <= target:
        eval_idx += clusters[c]
split = ["eval" if i in set(eval_idx) else "dev" for i in range(n)]

df = pd.DataFrame({"stem": stems, "path": paths, "cluster": labels, "split": split})
df.to_csv("outputs/split.csv", index=False)
print("clusters:", len(clusters), "| largest:", sorted(map(len, clusters.values()))[-5:])
print(df.split.value_counts())

# contact sheet: one row per multi-image cluster, to verify visually
multi = [c for c in clusters if len(clusters[c]) > 1]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = max(len(multi), 1)
fig, ax = plt.subplots(rows, 6, figsize=(12, 2 * rows), squeeze=False)
for r, c in enumerate(multi):
    for k in range(6):
        ax[r, k].axis("off")
        if k < len(clusters[c]):
            i = clusters[c][k]
            ax[r, k].imshow(cv2.cvtColor(cv2.resize(cv2.imread(paths[i]), (128, 128)), cv2.COLOR_BGR2RGB))
            ax[r, k].set_title(f"c{c}:{split[i]}", fontsize=7)
plt.tight_layout(); plt.savefig("outputs/clusters.png", dpi=110)

# leakage audit: each eval image vs its closest dev image
dev_i = [i for i in range(n) if split[i] == "dev"]
ev_i = [i for i in range(n) if split[i] == "eval"]
pairs = sorted((min((D[i, k], k) for k in dev_i) + (i,)) for i in ev_i)
for r, (d, k, i) in enumerate(pairs[:10]):
    print(f"#{r} eval c{labels[i]} <-> dev c{labels[k]}  d={float(d):.2f}")
fig, ax = plt.subplots(10, 2, figsize=(4, 20))
for r, (d, k, i) in enumerate(pairs[:10]):
    for c, idx in enumerate((i, k)):
        ax[r, c].imshow(cv2.cvtColor(cv2.resize(cv2.imread(paths[idx]), (128, 128)), cv2.COLOR_BGR2RGB))
        ax[r, c].axis("off")
    ax[r, 0].set_title(f"#{r} eval c{labels[i]}", fontsize=7)
    ax[r, 1].set_title(f"dev c{labels[k]}", fontsize=7)
plt.tight_layout(); plt.savefig("outputs/leak_audit.png", dpi=90)