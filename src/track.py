import collections, cv2, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

VIDEO, OUT = "data/video.mp4", "outputs/tracked.mp4"
ROI_TOP = 0.42
MIN_FRAC, MAX_FRAC = 0.008, 0.08
SEED_REL, GROW_REL = 0.92, 0.70      # fractions of the frame's 99th-percentile brightness

def is_table_scene(fr):
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    return g[g.shape[0] // 2:].mean() < 90

def detect(fr):
    H, W = fr.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    g[:int(ROI_TOP * H)] = 0
    g[int(0.85 * H):, int(0.80 * W):] = 0                     # mocomi logo
    p99 = np.percentile(g[int(ROI_TOP * H):], 99)
    if p99 < 150:                                              # no bright egg in view
        return []
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    seed = cv2.morphologyEx((g >= SEED_REL * p99).astype(np.uint8), cv2.MORPH_OPEN, k5)
    grow = (g >= GROW_REL * p99).astype(np.uint8)
    m = grow & cv2.dilate(seed, k13)                           # grow only near the seeds
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k5)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for c in cnts:
        a = cv2.contourArea(c)
        if len(c) < 5 or not (MIN_FRAC * H * W <= a <= MAX_FRAC * H * W):
            continue
        sol = a / cv2.contourArea(cv2.convexHull(c))
        (_, _), (ma, MA), _ = cv2.fitEllipse(c)
        fit = a / (np.pi * ma * MA / 4)
        if sol < 0.85 or not 0.75 < fit < 1.25:
            continue
        M = cv2.moments(c)
        dets.append(dict(c=c, cx=M["m10"] / M["m00"], cy=M["m01"] / M["m00"]))
    return dets

class Tracker:
    """Centroid tracker: constant-velocity prediction + Hungarian matching + distance gate.
    Re-identification: once the egg count is known (mode of detections over the first frames),
    no new IDs are created; a returning egg takes the ID of the nearest lost track."""
    def __init__(self, base_gate=80, min_hits=3, n_est_frames=120):
        self.tracks, self.next_id = [], 1
        self.base_gate, self.min_hits, self.n_est_frames = base_gate, min_hits, n_est_frames
        self.counts, self.n_eggs, self.reattached = [], None, 0

    def update(self, dets):
        if self.n_eggs is None and dets:
            self.counts.append(len(dets))
            if len(self.counts) >= self.n_est_frames:
                self.n_eggs = collections.Counter(self.counts).most_common(1)[0][0]
        for t in self.tracks:
            t["px"], t["py"] = t["cx"] + t["vx"], t["cy"] + t["vy"]
        r, c, cost = [], [], None
        if self.tracks and dets:
            cost = np.array([[np.hypot(t["px"] - d["cx"], t["py"] - d["cy"]) for d in dets]
                             for t in self.tracks])
            r, c = linear_sum_assignment(cost)
        mt, md = set(), set()
        for i, j in zip(r, c):
            t, d = self.tracks[i], dets[j]
            if cost[i, j] <= self.base_gate + 4 * t["missed"]:
                t["vx"] = 0.5 * t["vx"] + 0.5 * (d["cx"] - t["cx"])
                t["vy"] = 0.5 * t["vy"] + 0.5 * (d["cy"] - t["cy"])
                t.update(cx=d["cx"], cy=d["cy"], c=d["c"], missed=0, hits=t["hits"] + 1)
                t["trail"].append((int(d["cx"]), int(d["cy"])))
                mt.add(i); md.add(j)
        for i, t in enumerate(self.tracks):
            if i not in mt:
                t["missed"] += 1; t["c"] = None; t["vx"] *= 0.8; t["vy"] *= 0.8
        for j, d in enumerate(dets):
            if j not in md:
                self.tracks.append(dict(id=None, cx=d["cx"], cy=d["cy"], vx=0.0, vy=0.0, c=d["c"],
                                        missed=0, hits=1, trail=[(int(d["cx"]), int(d["cy"]))]))
        for t in list(self.tracks):                 # confirm tentative tracks
            if t["id"] is not None or t["hits"] < self.min_hits:
                continue
            confirmed = [u for u in self.tracks if u["id"] is not None]
            if self.n_eggs is not None and len(confirmed) >= self.n_eggs:
                lost = [u for u in confirmed if u["missed"] > 0]
                if lost:                            # returning egg: reuse nearest lost identity
                    u = min(lost, key=lambda u: np.hypot(u["cx"] - t["cx"], u["cy"] - t["cy"]))
                    u.update(cx=t["cx"], cy=t["cy"], c=t["c"], missed=0, vx=0.0, vy=0.0,
                             trail=[(int(t["cx"]), int(t["cy"]))])
                    self.reattached += 1
                self.tracks.remove(t)               # else: all eggs visible, so extra blob = false positive
            else:
                t["id"] = self.next_id; self.next_id += 1
        self.tracks = [t for t in self.tracks if t["id"] is not None or t["missed"] == 0]
        return [t for t in self.tracks if t["id"] is not None and t["c"] is not None]

COL = [(0, 0, 255), (0, 200, 0), (255, 120, 0), (0, 200, 255), (200, 0, 200)]
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
tr, rows, scene, ndet, f = Tracker(), [], [], {}, 0

while True:
    ok, fr = cap.read()
    if not ok:
        break
    out = fr.copy()
    if is_table_scene(fr):
        scene.append(f)
        dets = detect(fr)
        ndet[f] = len(dets)
        for t in tr.update(dets):
            col = COL[(t["id"] - 1) % len(COL)]
            ov = out.copy(); cv2.drawContours(ov, [t["c"]], -1, col, -1)
            out = cv2.addWeighted(ov, 0.45, out, 0.55, 0)
            cv2.drawContours(out, [t["c"]], -1, col, 2)
            cv2.polylines(out, [np.array(t["trail"][-40:], np.int32)], False, col, 2)
            cv2.putText(out, f"Egg {t['id']}", (int(t["cx"]) - 25, int(t["cy"]) - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            rows.append((f, t["id"], round(t["cx"], 1), round(t["cy"], 1), int(cv2.contourArea(t["c"]))))
    vw.write(out); f += 1
vw.release()

if not scene:
    raise SystemExit("No table scene found: adjust the threshold in is_table_scene()")
df = pd.DataFrame(rows, columns=["frame", "egg_id", "cx", "cy", "area"])
df.to_csv("outputs/tracks.csv", index=False)
print(f"table scene: frames {scene[0]}-{scene[-1]} ({scene[0]/fps:.1f}s-{scene[-1]/fps:.1f}s), {len(scene)} frames")
print("estimated egg count:", tr.n_eggs, "| identities re-attached after occlusion:", tr.reattached)
print("unique egg IDs:", [int(i) for i in sorted(df.egg_id.unique())])
print("frames per ID:", {int(k): int(v) for k, v in df.groupby("egg_id").size().items()})
cnt = df.groupby("frame").size().reindex(scene, fill_value=0)
print("tracked eggs per frame, counts of 0/1/2/3+:", [int((cnt == k).sum()) for k in (0, 1, 2)], int((cnt >= 3).sum()))

cap2 = cv2.VideoCapture(OUT)
fig, ax = plt.subplots(3, 4, figsize=(16, 9))
for a, p in zip(ax.ravel(), np.linspace(scene[0], scene[-1], 12).astype(int)):
    cap2.set(cv2.CAP_PROP_POS_FRAMES, int(p)); ok, im = cap2.read()
    if ok:
        a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    a.set_title(f"frame {p} ({p/fps:.1f}s)", fontsize=8); a.axis("off")
plt.tight_layout(); plt.savefig("outputs/track_contact_sheet.png", dpi=90)