import cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("data/video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"{w}x{h}, {fps:.1f} fps, {n} frames, {n / fps:.1f} s")

picks = [int(i * n / 12) for i in range(12)]
fig, ax = plt.subplots(3, 4, figsize=(16, 9))
for a, f in zip(ax.ravel(), picks):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, fr = cap.read()
    if ok:
        a.imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    a.set_title(f"frame {f} ({f / fps:.1f}s)", fontsize=8); a.axis("off")
plt.tight_layout(); plt.savefig("outputs/video_contact_sheet.png", dpi=90)