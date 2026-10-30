import os, cv2, numpy as np, pandas as pd

df = pd.read_csv("outputs/split.csv")
DEV = df[df.split == "dev"]          # eval is never loaded here
LOG = "outputs/experiment_log.csv"

def load_gt(stem):
    return cv2.imread(f"outputs/sam_masks/{stem}.png", 0) > 0

def iou_dice(pred, gt):
    inter = (pred & gt).sum(); union = (pred | gt).sum()
    s = pred.sum() + gt.sum()
    return (inter / union if union else 1.0), (2 * inter / s if s else 1.0)

def run_trial(trial_id, fn, params, what, why):
    ious, dices, per = [], [], []
    for _, r in DEV.iterrows():
        img = cv2.imread(r.path)
        pred = fn(img, **params) > 0
        i, d = iou_dice(pred, load_gt(r.stem))
        ious.append(i); dices.append(d); per.append((r.stem, i))
    row = dict(trial=trial_id, what_changed=what, why=why, params=str(params),
               dev_mIoU=round(np.mean(ious), 4), dev_median_IoU=round(np.median(ious), 4),
               dev_mDice=round(np.mean(dices), 4), n_fail_iou_below_0_5=int(sum(i < 0.5 for i in ious)))
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)
    pd.DataFrame(per, columns=["stem", "iou"]).to_csv(f"outputs/per_image_{trial_id}.csv", index=False)
    print(row)

def keep_big(mask, min_frac=0.01):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    out = np.zeros_like(mask, np.uint8)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= min_frac * mask.size:
            out[lab == k] = 1
    return out

# Trial 1: global Otsu threshold on grayscale (eggs are bright)
def t1_otsu(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m > 0

# Trial 2: low saturation + high value in HSV, then clean-up
def t2_hsv(img, s_max=60, v_min=140, k=7):
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] <= s_max) & (hsv[..., 2] >= v_min)).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    return keep_big(m) > 0

def pick_eggs(m, min_frac=0.01, max_frac=0.5, sol=0.9, ell_lo=0.8, ell_hi=1.2, rel=0.25, max_eggs=3):
    """Keep only egg-shaped components (convex + elliptical), with holes filled."""
    m = m.astype(np.uint8)
    H, W = m.shape
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in cnts:
        a = cv2.contourArea(c)
        if len(c) < 5 or not (min_frac * H * W <= a <= max_frac * H * W):
            continue
        s = a / cv2.contourArea(cv2.convexHull(c))
        (_, _), (ma, MA), _ = cv2.fitEllipse(c)
        e = a / (np.pi * ma * MA / 4)
        if s >= sol and ell_lo < e < ell_hi:
            cands.append((a, c))
    cands.sort(key=lambda x: -x[0])
    out = np.zeros_like(m)
    if cands:
        big = cands[0][0]
        for a, c in cands[:max_eggs]:
            if a >= rel * big:
                cv2.drawContours(out, [c], -1, 1, -1)
    return out

# Trial 3: T2 colour rule + shape filter (isolates the value of a shape prior)
def t3_hsv_shape(img, s_max=60, v_min=140, k=7, sol=0.9):
    return pick_eggs(t2_hsv(img, s_max, v_min, k), sol=sol) > 0

# Trial 4: colour distance from the border background in Lab + shape filter
def t4_bgdist(img, border=0.08, k=7, sol=0.9):
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    H, W = lab.shape[:2]
    b = max(int(border * min(H, W)), 2)
    strip = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                            lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(strip, axis=0)
    d = np.linalg.norm(lab - bg, axis=2)
    d8 = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx((m > 0).astype(np.uint8), cv2.MORPH_OPEN, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    return pick_eggs(m, sol=sol) > 0

# Trial 5: GrabCut initialised with a central rectangle
def t5_grabcut(img, margin=0.08, iters=5, work=320, rel=0.25):
    H, W = img.shape[:2]
    s = cv2.resize(img, (work, work))
    mask = np.zeros((work, work), np.uint8)
    m0 = int(margin * work)
    rect = (m0, m0, work - 2 * m0, work - 2 * m0)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.setRNGSeed(0)
        cv2.grabCut(s, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((H, W), bool)
    fg = ((mask == 1) | (mask == 3)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    out = np.zeros_like(fg)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] >= rel * areas.max():
                out[lab == k] = 1
    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(out)
    cv2.drawContours(filled, cnts, -1, 1, -1)          # fill holes
    return cv2.resize(filled, (W, H), interpolation=cv2.INTER_NEAREST) > 0
def t6_grabcut_tight(img):
    return t5_grabcut(img, margin=0.2)

def shape_score(m):
    """Area-weighted convexity x ellipse-fit. Returns 0 for empty or implausible masks."""
    m8 = m.astype(np.uint8)
    frac = m8.mean()
    if frac < 0.01 or frac > 0.5:
        return 0.0
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num = den = 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if len(c) < 5 or a < 50:
            continue
        sol = a / cv2.contourArea(cv2.convexHull(c))
        (_, _), (ma, MA), _ = cv2.fitEllipse(c)
        e = a / (np.pi * ma * MA / 4)
        fit = max(0.0, 1 - 3 * abs(1 - e))
        num += a * sol * fit
        den += a
    return num / den if den else 0.0

def t7_multi(img, margins=(0.05, 0.15, 0.25, 0.33)):
    best, best_s = None, -1
    for mg in margins:
        m = t5_grabcut(img, margin=mg)
        s = shape_score(m)
        if s > best_s:
            best, best_s = m, s
    return best

def skin_mask(img):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[..., 1], ycc[..., 2]
    return (cr >= 140) & (cr <= 173) & (cb >= 77) & (cb <= 127)

def clean(m, k=7, rel=0.25):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, ker)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m)
    if n > 1:
        big = stats[1:, cv2.CC_STAT_AREA].max()
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= rel * big:
                out[lab == i] = 1
    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(out)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    return filled > 0

def t8_noskin(img, margins=(0.05, 0.15, 0.25, 0.33)):
    sk = skin_mask(img)
    best, best_s = None, -1
    for mg in margins:
        m = clean(t5_grabcut(img, margin=mg) & ~sk)
        s = shape_score(m)
        if s > best_s:
            best, best_s = m, s
    return best

def shape_score2(m):
    """Area-weighted IoU between each component and its own fitted ellipse."""
    m8 = m.astype(np.uint8)
    frac = m8.mean()
    if frac < 0.01 or frac > 0.5:
        return 0.0
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num = den = 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if len(c) < 5 or a < 50:
            continue
        comp = np.zeros_like(m8); cv2.drawContours(comp, [c], -1, 1, -1)
        ell = np.zeros_like(m8); cv2.ellipse(ell, cv2.fitEllipse(c), 1, -1)
        inter = (comp & ell).sum(); union = (comp | ell).sum()
        num += a * (inter / union if union else 0.0)
        den += a
    return num / den if den else 0.0

def grabcut_ellipse(img, scale=0.6, iters=5, work=320):
    H, W = img.shape[:2]
    s = cv2.resize(img, (work, work))
    mask = np.full((work, work), cv2.GC_PR_BGD, np.uint8)
    c, r = (work // 2, work // 2), int(scale * work / 2)
    cv2.ellipse(mask, c, (r, r), 0, 0, 360, cv2.GC_PR_FGD, -1)
    cv2.ellipse(mask, c, (int(0.4 * r), int(0.4 * r)), 0, 0, 360, cv2.GC_FGD, -1)
    b = int(0.03 * work)
    mask[:b] = mask[-b:] = cv2.GC_BGD
    mask[:, :b] = mask[:, -b:] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.setRNGSeed(0)
        cv2.grabCut(s, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.zeros((H, W), bool)
    fg = ((mask == 1) | (mask == 3)).astype(np.uint8)
    return cv2.resize(fg, (W, H), interpolation=cv2.INTER_NEAREST) > 0

def _best(img, cands):
    sk = skin_mask(img)
    best, bs = None, -1
    for c in cands:
        m = clean(c & ~sk)
        s = shape_score2(m)
        if s > bs:
            best, bs = m, s
    return best

def t9_ellscore(img, margins=(0.05, 0.15, 0.25, 0.33)):
    return _best(img, [t5_grabcut(img, margin=mg) for mg in margins])

def t10_pool(img, margins=(0.05, 0.15, 0.25, 0.33), scales=(0.45, 0.65, 0.85)):
    cands = [t5_grabcut(img, margin=mg) for mg in margins]
    cands += [grabcut_ellipse(img, scale=s) for s in scales]
    return _best(img, cands)

if __name__ == "__main__":
    run_trial("T9", t9_ellscore, dict(margins=(0.05, 0.15, 0.25, 0.33)),
              "T8 with shape score = IoU between mask and its fitted ellipse",
              "old score accepted near-rectangular masks (visible in T7/T8 failures)")
    run_trial("T10", t10_pool, dict(margins=(0.05, 0.15, 0.25, 0.33), scales=(0.45, 0.65, 0.85)),
              "T9 + GrabCut candidates initialised from an ellipse mask (sure core, probable ring)",
              "rectangle init marks the whole box probable-foreground and leaks into white cloth/bag")