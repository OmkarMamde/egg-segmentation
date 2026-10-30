import sys, cv2, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import classical
from classical import load_gt, DEV

FN = {"T2": classical.t2_hsv, "T3": classical.t3_hsv_shape, "T4": classical.t4_bgdist,
      "T5": classical.t5_grabcut, "T6": classical.t6_grabcut_tight, "T7": classical.t7_multi, "T8": classical.t8_noskin, "T9": classical.t9_ellscore, "T10": classical.t10_pool}

trial = sys.argv[1] if len(sys.argv) > 1 else "T2"
per = pd.read_csv(f"outputs/per_image_{trial}.csv").sort_values("iou")
# 4 worst + 4 from the middle, to see both kinds of failure
pick = list(per.head(4).stem) + list(per.iloc[40:44].stem)
paths = dict(zip(DEV.stem, DEV.path))
iou_of = per.set_index("stem").iou

fig, ax = plt.subplots(len(pick), 3, figsize=(7, 2.3 * len(pick)))
for r, s in enumerate(pick):
    img = cv2.imread(paths[s]); gt = load_gt(s); pr = FN[trial](img)
    ax[r, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[r, 1].imshow(gt, cmap="gray")
    ax[r, 2].imshow(pr, cmap="gray")
    for c in range(3): ax[r, c].axis("off")
    ax[r, 0].set_title(s[:22], fontsize=6)
    ax[r, 1].set_title("SAM (GT)", fontsize=6)
    ax[r, 2].set_title(f"pred IoU={iou_of[s]:.2f}", fontsize=6)
plt.tight_layout(); plt.savefig(f"outputs/failures_{trial}.png", dpi=100)