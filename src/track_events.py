import cv2, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/tracks.csv")
cap = cv2.VideoCapture("outputs/tracked.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

events = []
for i, g in df.groupby("egg_id"):
    g = g.sort_values("frame").reset_index(drop=True)
    for k in np.where(np.diff(g.frame.values) > 12)[0]:
        a, b = g.iloc[k], g.iloc[k + 1]
        events.append(dict(egg=int(i), lost=int(a.frame), back=int(b.frame),
                           from_xy=(int(a.cx), int(a.cy)), to_xy=(int(b.cx), int(b.cy)),
                           jump=round(float(np.hypot(a.cx - b.cx, a.cy - b.cy)))))
events.sort(key=lambda e: e["lost"])
for e in events:
    print(f"Egg {e['egg']}: lost {e['lost']/fps:.1f}s at {e['from_xy']}, "
          f"back {e['back']/fps:.1f}s at {e['to_xy']}, moved {e['jump']} px")

ev = events[:8]
fig, ax = plt.subplots(len(ev), 2, figsize=(10, 3 * len(ev)), squeeze=False)
for r, e in enumerate(ev):
    for c, key in enumerate(("lost", "back")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e[key]); ok, im = cap.read()
        if ok:
            ax[r, c].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax[r, c].set_title(f"Egg {e['egg']} {key} (frame {e[key]}, {e[key]/fps:.1f}s)", fontsize=8)
        ax[r, c].axis("off")
plt.tight_layout(); plt.savefig("outputs/track_events.png", dpi=80)