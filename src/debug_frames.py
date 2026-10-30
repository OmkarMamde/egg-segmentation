import cv2, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FR = [300, 400, 718, 860]          # 300/400/718 = missed, 860 = detected
cap = cv2.VideoCapture("data/video.mp4")
fig, ax = plt.subplots(len(FR), 4, figsize=(14, 3.2 * len(FR)))
for r, f in enumerate(FR):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = cap.read()
    H, W = fr.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    roi = g[int(0.42 * H):]
    otsu, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"frame {f}: otsu={otsu:.0f} p50={np.percentile(roi, 50):.0f} "
          f"p95={np.percentile(roi, 95):.0f} p99={np.percentile(roi, 99):.0f}")
    ax[r, 0].imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); ax[r, 0].set_title(f"frame {f}")
    for c, t in enumerate([120, 170, 200], 1):
        m = (g >= t).astype(np.uint8); m[:int(0.42 * H)] = 0
        ax[r, c].imshow(m, cmap="gray"); ax[r, c].set_title(f"threshold {t}")
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cn in cnts:
            a = cv2.contourArea(cn)
            if a < 900 or len(cn) < 5:
                continue
            sol = a / cv2.contourArea(cv2.convexHull(cn))
            (_, _), (ma, MA), _ = cv2.fitEllipse(cn)
            x, y, w, h = cv2.boundingRect(cn)
            print(f"   t={t}: area={a:.0f} solidity={sol:.2f} ellipse_fit={a / (np.pi * ma * MA / 4):.2f} at x={x},y={y},{w}x{h}")
    for a in ax[r]:
        a.axis("off")
plt.tight_layout(); plt.savefig("outputs/debug_frames.png", dpi=80)