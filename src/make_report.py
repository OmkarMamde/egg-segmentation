import os
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# ---------- EDIT THESE ----------
NAME = "Omkar Mamde"
REPO = "https://github.com/OmkarMamde/egg-segmentation"
VIDEO = "https://drive.google.com/file/d/1uN3XbthPm1NlQ5ngaxFQXNhVNpkKYYWK/view?usp=sharing"
# --------------------------------

ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13)
S = ParagraphStyle("S", parent=B, fontSize=7.5, leading=9.5)
story = []

def H(t, lvl=1): story.append(Paragraph(t, ss["Heading%d" % lvl]))
def P(t, st=B): story.append(Paragraph(t, st)); story.append(Spacer(1, 5))
def bullets(items):
    for i in items: P("&bull; " + i)

def img(path, caption, w=17 * cm, hmax=21 * cm):
    if not os.path.exists(path):
        P("[missing figure: %s]" % path); return
    iw, ih = ImageReader(path).getSize()
    h = w * ih / iw
    if h > hmax: w, h = hmax * iw / ih, hmax
    story.append(Image(path, width=w, height=h))
    P("<i>%s</i>" % caption, S)

def table(rows, widths):
    data = [[Paragraph(str(c), S) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(t); story.append(Spacer(1, 8))

# ---------- data pulled from your outputs ----------
sp = pd.read_csv("outputs/split.csv")
n_dev, n_ev, n_cl = int((sp["split"] == "dev").sum()), int((sp["split"] == "eval").sum()), sp.cluster.nunique()
log = pd.read_csv("outputs/experiment_log.csv").drop_duplicates("trial", keep="last")
log["n"] = log.trial.str[1:].astype(int); log = log.sort_values("n")
ev = pd.read_csv("outputs/eval/eval_summary.csv")
dev_final = log[log.trial == "T10"].iloc[0]
ev_miou = float(ev[ev.metric == "mIoU"]["eval"].iloc[0])
def fmt(v): return "-" if pd.isna(v) else (str(int(v)) if float(v).is_integer() else "%.3f" % v)

DEC = {
 "T1": "Discarded. Baseline: a global threshold cannot separate a white egg from bright surfaces.",
 "T2": "Kept as the colour-only reference. Fails on white-on-white and backlit images.",
 "T3": "Discarded. Median IoU 0: the shape filter deletes everything when the egg is merged with the background.",
 "T4": "Discarded. The border-is-background assumption breaks when the egg or a hand touches the border.",
 "T5": "Kept. First big gain (GrabCut). Fails when the initial rectangle is dominated by background.",
 "T6": "Discarded as final method. Median improved but mean did not: no single margin fits every image.",
 "T7": "Kept. Per-image margin choice, but the shape score also accepted near-rectangular masks.",
 "T8": "Kept, marginal gain. Skin removal helps on hands but cuts chunks from a bluish egg.",
 "T9": "Kept. Ellipse-IoU shape score. Re-run after seeding the RNG.",
 "T10": "FINAL. Adds ellipse-initialised GrabCut candidates. Best dev mIoU.",
}

# ---------- document ----------
H("CV Intern Task: Egg Segmentation and Tracking")
P("Author: %s<br/>Code: <link href='%s' color='blue'>%s</link><br/>Tracked video: <link href='%s' color='blue'>%s</link>" % (NAME, REPO, REPO, VIDEO, VIDEO))

H("1. Summary", 2)
bullets([
 "Task 1: SAM (ViT-H) masks for 107 of 108 images; 1 image excluded because SAM failed on it.",
 "Task 2: after 10 logged trials the final classical method (T10) reaches dev mIoU %.3f against the SAM masks." % dev_final.dev_mIoU,
 "Task 3: on the held-out eval split (%d images) mIoU is %.3f. The main error is leakage into the background (precision below recall)." % (n_ev, ev_miou),
 "Task 4: both eggs are tracked through the table scene of the video, with one identity swap around 31 s when both eggs were hidden by the hand.",
])

H("2. Data and Task 1: segmentation with a pretrained model", 2)
P("The dataset has 108 images (train 75, valid 22, test 11). I merged the three Roboflow folders and created my own split (section 3). "
  "Ground truth comes from SAM ViT-H automatic mask generation (points_per_side 32, pred_iou_thresh 0.86, stability_score_thresh 0.9). "
  "Because SAM returns many masks per image, I filter them with egg priors: area between 1.5 percent and 40 percent of the image, solidity at least 0.95, ellipse fit between 0.88 and 1.12, no contact with the image border, "
  "overlapping duplicates removed, masks smaller than 25 percent of the largest kept mask dropped, at most 5 eggs per image.")
P("<b>What did not work first.</b> My first filter was looser (minimum area 0.3 percent, solidity 0.93) and produced up to 35 masks per image; only 18 images had exactly one mask although most images contain one egg. "
  "Tightening the filter fixed this but left 4 images with no egg. For three of them a relaxed filter (solidity 0.88, border contact allowed) worked. For one (image 12) the relaxed filter selected the herb background, so I used a SAM box prompt. "
  "Image 42 (a white egg in a white tray) could not be segmented by either method, and I excluded it. This leaves 107 images. The exclusion is listed in <font face='Courier'>outputs/excluded_images.txt</font>.")
P("<b>Limitations of this ground truth.</b> The shape filter biases the masks toward convex, elliptical regions, which can favour my own shape-based algorithm in Task 3. "
  "Some masks are questionable (for example eggs in the white tray, image 117, and a fabric patch counted as an egg in image 52). I did not edit any mask after seeing evaluation results.")

H("3. Dev/eval split (4:1) with reduced leakage", 2)
P("Several photos show the same scene (same bag, tray, pot, keyboard or herbs), so a random split would place near-duplicates on both sides and inflate eval scores. "
  "I described each image by an HSV colour histogram (8x8x8 bins), computed Bhattacharyya distances, clustered with average-linkage agglomerative clustering (distance threshold 0.35), "
  "and assigned whole clusters to dev or eval until eval held about 20 percent of the images (seed 42). Result: %d clusters, %d dev images and %d eval images." % (n_cl, n_dev, n_ev))
P("<b>Audit and manual merges.</b> For every eval image I looked at the closest dev image. The audit found clusters that were clearly one scene but had been separated by the histogram (steel pot, silver foil wrapper, white bag, pink tub, herbs), "
  "and I merged them manually (exact merge lines are in <font face='Courier'>src/split.py</font>). After the merges the closest eval-to-dev distances were 0.39 or larger, compared with 0.31 to 0.38 for the leaking pairs. "
  "Some same-session similarity (same kitchen, same table) cannot be removed with 107 images, so eval scores are probably still slightly optimistic. The split was frozen before any algorithm tuning, and all tuning used dev only.")
img("outputs/clusters.png", "Multi-image clusters used for the split (dev or eval label above each image).", hmax=12 * cm)
img("outputs/leak_audit.png", "Leakage audit: closest eval-to-dev pairs after the merges.", hmax=14 * cm)

story.append(PageBreak())
H("4. Task 2: segmentation with classical image processing", 2)
P("<b>Final algorithm (T10).</b> For each image: (1) run GrabCut at 4 rectangle margins (0.05, 0.15, 0.25, 0.33) and at 3 ellipse-initialised settings (scales 0.45, 0.65, 0.85; sure-foreground core, border strip as sure background) on a 320x320 downscale; "
  "(2) remove skin-coloured pixels (YCrCb Cr 140 to 173, Cb 77 to 127), apply a 7x7 opening, drop components smaller than 25 percent of the largest, fill holes; "
  "(3) pick the candidate with the best ellipse-IoU score (IoU between each component and its own fitted ellipse, area-weighted). The selection uses shape only, never the ground truth. The GrabCut random generator is seeded (cv2.setRNGSeed(0)).")
P("<b>Experiment log</b> (all scores on the %d dev images; IoU and Dice computed against SAM masks). Parameters are as logged by the code." % n_dev)
rows = [["Trial", "What changed / why", "Parameters", "dev mIoU", "dev median", "IoU&lt;0.5", "Decision"]]
for _, r in log.iterrows():
    rows.append([r.trial, "<b>%s</b><br/>Why: %s" % (r.what_changed, r.why), str(r.params).replace("np.float64", ""),
                 "%.3f" % r.dev_mIoU, "%.3f" % r.dev_median_IoU, int(r.n_fail_iou_below_0_5), DEC.get(r.trial, "")])
table(rows, [1.1 * cm, 5.6 * cm, 4.0 * cm, 1.3 * cm, 1.3 * cm, 1.5 * cm, 3.2 * cm])
P("<b>How the trials led to the algorithm.</b> Colour thresholds (T1, T2) failed on white eggs against bright or white backgrounds and on backlit images. A shape filter on top of them (T3) made things worse because the egg was usually merged into a larger blob, so no egg-shaped component existed. "
  "Background modelling from the border (T4) failed when the egg or a hand touched the border. GrabCut (T5) gave the first big gain. A tighter rectangle (T6) fixed typical images but not all, which motivated choosing the margin per image by shape (T7). "
  "The failure grids showed near-rectangular masks being selected, which led to the ellipse-IoU score (T9) and to ellipse-initialised candidates (T10).")
P("<b>Reproducibility note.</b> GrabCut turned out to be non-deterministic between runs. I seeded the generator and re-ran T9 and T10. T5 to T8 were run before seeding, so their numbers carry small run-to-run noise (about 0.01 to 0.02 mIoU), which does not change the ranking of the trials.")
img("outputs/failures_T10.png", "Dev-set failure grid for the final method: image, SAM mask, prediction.", hmax=14 * cm)

story.append(PageBreak())
H("5. Task 3: evaluation and analysis (eval split, %d images)" % n_ev, 2)
P("Metrics are pixel-wise on the union of all eggs in an image: IoU, Dice, precision and recall, plus agreement between predicted and ground-truth egg counts. The algorithm was run on the eval split once, after it was frozen.")
rows = [["Metric", "Eval (%d images)" % n_ev, "Dev (%d images, reference)" % n_dev]]
for _, r in ev.iterrows(): rows.append([r.metric.replace("<", "&lt;"), fmt(r["eval"]), fmt(r["dev (reference)"])])
table(rows, [7 * cm, 5 * cm, 6 * cm])
P("<b>Quantitative findings.</b> Eval mIoU is close to dev mIoU, so the method generalises. The failure rate (IoU below 0.5) is slightly higher on eval (9 of 21, 43 percent) than on dev (29 of 86, 34 percent), but with 21 images one image is worth 5 percentage points. "
  "Precision is lower than recall, so the typical error is leakage into the background, not missed eggs. The IoU histogram is bimodal: the method either works well or fails badly. "
  "Small eggs (below roughly 7 percent of the image area) mostly score below 0.6. IoU also drops with more eggs per image, but the 2-egg and 3-egg groups contain only a few images.")
img("outputs/eval/eval_plots.png", "IoU distribution, precision vs recall, IoU vs egg size, IoU by number of eggs.", hmax=6 * cm)
P("<b>Qualitative findings.</b> The error map shows true positives in green, false positives in red and false negatives in blue.")
img("outputs/eval/eval_failures.png", "Six worst eval images: image, SAM mask, prediction, error map.", hmax=17 * cm)
rows = [["Image", "What went wrong", "Likely cause and possible fix"],
 ["22", "Mask includes the white label next to the egg.", "Egg and label are both smooth and near-white. Needs edge or learned cues."],
 ["108", "Egg found, plus extra blobs from the lid.", "Pale lid patches pass the shape rules. Keep only the single most egg-like component."],
 ["13", "Mask spreads into herbs and shadow.", "Egg is blown out and has no clear boundary. Needs better local models or a two-pass crop."],
 ["152", "Large quadrilateral covering hand and wall.", "Skin filter misses the pale palm. Stronger skin model (HSV and YCrCb)."],
 ["52", "Ring of rug and earphone wire around the egg.", "Shadow and white wire blend with the shell. Crop-and-refine pass."],
 ["117", "Egg barely found.", "White egg in a white tray, and the SAM mask here is itself questionable."]]
table(rows, [1.6 * cm, 6.4 * cm, 10 * cm])
P("<b>Caveats.</b> Two of the six worst cases (117 and 52) involve questionable SAM masks, so part of the measured gap is ground-truth error. "
  "The filtered SAM masks and my shape-based selection share a convexity bias, which probably flatters the scores a little.")
P("<b>Improvements.</b> (1) White-on-white scenes: edge or gradient cues, or a learned model. (2) Small eggs: a coarse pass to find the egg, then GrabCut on a crop. "
  "(3) Leakage: keep only the single most egg-like component and use a stronger skin model. (4) Several touching eggs: distance-transform watershed to split them.")

story.append(PageBreak())
H("6. Task 4: video tracking", 2)
P("The video is 640x360, 24 fps, 60 s. It contains a green intro card (0 to about 8 s), a table scene with two eggs and a hand (about 8.6 s to 42.1 s) and a text card. "
  "yt-dlp could not download it (YouTube extraction error), so I obtained the file from a third-party downloader. Tracking runs only on frames classified as table scene (mean brightness of the lower half of the frame is low).")
P("<b>Segmentation change.</b> Here the eggs are the brightest objects on a dark, uniform table and are off-centre, so I replaced GrabCut by hysteresis thresholding: seeds are pixels above 92 percent of the frame's 99th brightness percentile, "
  "the mask grows to 70 percent of that value but only within a 13x13 dilation of the seeds, then a shape filter (solidity and ellipse fit) and an area filter are applied. The video overlay labels and the logo are masked out.")
P("<b>Tracker.</b> Constant-velocity prediction, Hungarian assignment between predicted positions and detections, a distance gate (80 px plus 4 px per missed frame), and confirmation after 3 hits. "
  "The egg count is estimated as the most common detection count over the first 120 frames with detections (2). After that no new identities are created; a returning detection takes over the nearest lost identity, and extra blobs are rejected as false positives.")
rows = [["Run", "Change", "Unique IDs (ideal 2)", "Frames with no tracked egg (of 804)"],
 ["1", "Otsu threshold (clipped at 120) and centroid tracker, memory 1.5 s", "18", "328"],
 ["2", "Hysteresis threshold (overlay labels were merging with the eggs), memory 5 s", "4", "40"],
 ["3", "Egg count estimated, no new IDs, re-attach to nearest lost track", "2", "40"]]
table(rows, [1.5 * cm, 9.5 * cm, 3.5 * cm, 3.5 * cm])
P("<b>Result and limitations.</b> In the final run, 534 of 804 table-scene frames show both eggs and 230 show one egg. "
  "Detection fails while the hand covers or touches an egg, because fingers merge with the egg and the blob fails the shape test. There were 4 occlusion gaps. In my visual check an identity swap occurred around 31 s: both eggs were hidden at once, "
  "they look identical (areas 4,701 and 4,790 px in one frame), and position alone cannot decide which identity returns. Possible improvements are tracking the hand and attaching a hidden egg to it, or an appearance model.")
img("outputs/track_contact_sheet.png", "Tracked frames (colour and label = identity).", hmax=10 * cm)
img("outputs/track_events.png", "Occlusion events: frame when an egg is lost and when it returns.", hmax=14 * cm)

H("7. AI tools disclosure", 2)
P("I used Claude (Anthropic) as an AI assistant for planning the pipeline, drafting and debugging code (SAM mask filtering, split clustering, trial functions, evaluation and tracking scripts) and for drafting this report. "
  "I ran every experiment myself, inspected the outputs, and made the decisions on exclusions, cluster merges and which trials to keep or discard. I can explain each method and modify it.")

os.makedirs("report", exist_ok=True)
SimpleDocTemplate("report/report.pdf", pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                  topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="CV Intern Task Report", author=NAME).build(story)
print("wrote report/report.pdf")