import os, cv2, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from classical import t10_pool

df = pd.read_csv("outputs/split.csv")
EV = df[df.split == "eval"]
os.makedirs("outputs/eval", exist_ok=True)

rows, vis = [], []
for _, r in EV.iterrows():
    img = cv2.imread(r.path)
    lab = cv2.imread(f"outputs/sam_masks/{r.stem}.png", 0)
    gt = lab > 0
    pr = t10_pool(img) > 0
    tp = (pr & gt).sum(); fp = (pr & ~gt).sum(); fn = (~pr & gt).sum()
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    n_gt = int(lab.max())
    n_pr = cv2.connectedComponents(pr.astype(np.uint8))[0] - 1
    rows.append(dict(stem=r.stem, iou=iou, dice=dice, precision=prec, recall=rec,
                     gt_area_frac=gt.mean(), pred_area_frac=pr.mean(),
                     area_ratio=pr.sum() / max(gt.sum(), 1), n_gt=n_gt, n_pred=n_pr))
    vis.append((r.stem, img, gt, pr, iou))

res = pd.DataFrame(rows)
res.to_csv("outputs/eval/eval_per_image.csv", index=False)

# dev vs eval comparison (dev numbers come from the frozen T10 run)
dev = pd.read_csv("outputs/per_image_T10.csv")
summary = pd.DataFrame({
    "metric": ["mIoU", "median IoU", "mDice", "mean precision", "mean recall",
               "images IoU<0.5", "count match (n_pred == n_gt)"],
    "eval": [res.iou.mean(), res.iou.median(), res.dice.mean(), res.precision.mean(),
             res.recall.mean(), int((res.iou < 0.5).sum()), int((res.n_gt == res.n_pred).sum())],
    "dev (reference)": [dev.iou.mean(), dev.iou.median(), np.nan, np.nan, np.nan,
                        int((dev.iou < 0.5).sum()), np.nan]})
summary.to_csv("outputs/eval/eval_summary.csv", index=False)
print(summary.round(3).to_string(index=False))
print("\nover-segmented (area_ratio>1.5):", int((res.area_ratio > 1.5).sum()),
      "| under-segmented (area_ratio<0.67):", int((res.area_ratio < 0.67).sum()))

# plots
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
ax[0].hist(res.iou, bins=10, range=(0, 1)); ax[0].set_title("IoU distribution (eval)")
ax[0].set_xlabel("IoU"); ax[0].set_ylabel("images")
ax[1].scatter(res.recall, res.precision); ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
ax[1].set_title("Precision vs recall\n(low precision = leakage, low recall = missed egg)")
ax[1].set_xlim(0, 1.02); ax[1].set_ylim(0, 1.02)
ax[2].scatter(res.gt_area_frac, res.iou); ax[2].set_xlabel("egg area / image area")
ax[2].set_ylabel("IoU"); ax[2].set_title("IoU vs egg size")
res.boxplot(column="iou", by="n_gt", ax=ax[3]); ax[3].set_title("IoU by number of eggs")
ax[3].set_xlabel("eggs in image"); plt.suptitle("")
plt.tight_layout(); plt.savefig("outputs/eval/eval_plots.png", dpi=110)

# worst 6: image | GT | prediction | error map (green=TP, red=FP, blue=FN)
worst = sorted(vis, key=lambda v: v[4])[:6]
fig, ax = plt.subplots(len(worst), 4, figsize=(9, 2.3 * len(worst)))
for k, (s, img, gt, pr, iou) in enumerate(worst):
    err = np.zeros((*gt.shape, 3), np.uint8)
    err[gt & pr] = (0, 200, 0); err[~gt & pr] = (220, 0, 0); err[gt & ~pr] = (0, 80, 255)
    for c, (im, t) in enumerate([(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), s[:20]),
                                 (gt, "SAM GT"), (pr, f"pred IoU={iou:.2f}"), (err, "TP / FP / FN")]):
        ax[k, c].imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax[k, c].axis("off"); ax[k, c].set_title(t, fontsize=7)
plt.tight_layout(); plt.savefig("outputs/eval/eval_failures.png", dpi=100)