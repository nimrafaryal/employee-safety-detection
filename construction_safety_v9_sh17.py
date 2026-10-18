"""
Construction-site PPE monitor (v9, for SH17 models) - "showcase" edition

New in v9 - TWO-PASS processing for a stable result:
- Pass 1 analyses the WHOLE video (tracking + PPE evidence for every worker).
- Each worker then gets ONE final verdict per item from all of its evidence, so the
  label does not flip around and is settled long before the clip ends.
- Pass 2 renders: each worker gets a short "ANALYZING" scan (the H/V/G/S chips are
  revealed one by one), then the final SAFE / VIOLATION verdict stays locked.
- Items that are never visible (e.g. feet out of frame) no longer block SAFE: they show
  "?" (not visible). Workers with no verifiable item at all show "NOT CLEAR".
- Worker boxes are gap-filled and smoothed over time (no jitter, no blinking).
- PPE boxes shown on screen always agree with the verdict.

Same detection logic as v7 (multi-scale PPE detection, body-part violations, skin check,
scene-cut reset, smoothing). New in v8 is the LOOK:

- A large dashboard panel BESIDE the video (it never covers workers):
    live compliance gauge, worker counts, SAFE / VIOLATION / ANALYZING tiles,
    per-item PPE bars (wearing vs missing vs not visible), a live worker list,
    a timeline chart, and a legend that explains every colour and box style.
- On the video:
    glowing corner brackets per worker (green = safe, red = violation, amber = checking),
    a scanning line while a worker is still being checked, a pulsing red glow on violations,
    a "NO GLOVES" style callout under the worker,
    the actual PPE detections (solid boxes) and bare body parts (dashed red boxes),
    alert pop-ups when a new violation is confirmed, an intro title, a progress bar
    with violation markers, and a soft vignette.
"""
from ultralytics import YOLO
from pathlib import Path
from collections import deque
import csv
import math
import subprocess
import urllib.request
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ============================ CONFIG ============================
VIDEO_PATH  = "/content/test_video.mp4"
MODEL_PATH  = "/content/sh17_yolo8m.pt"
OUTPUT_PATH = "/content/safety_v9.mp4"
REPORT_PATH = "/content/worker_report.csv"
FONT_DIR    = "/content/fonts"

IMGSZ       = 960         # size for worker tracking; 960/1280 = better for small/far workers
PPE_SIZES   = (960, 320)  # PPE is detected at BOTH sizes: 960 for far workers, 320 for close-ups
SKIN_CHECK  = True        # double-check "bare hand" boxes by skin colour (stops white gloves -> NO GLOVES)
SKIN_T      = 0.55        # a "hand" box needs at least this share of skin-coloured pixels to count as bare
PERSON_CONF = 0.40        # a track is only started/kept on detections at least this confident
PPE_CONF    = 0.25        # PPE detection confidence (low: missing a helmet is worse than noise)
PART_CONF   = 0.50        # a BARE head/hand/foot must be at least this confident to count
MIN_HAND_PX = 40          # hands smaller than this (px) are too small to tell glove vs bare
MIN_VIOL_S  = 0.5         # a violation must last this long (s) to be logged / alerted
MIN_COUNT_S = 0.5         # a worker must be tracked this long (s) to be in the total count
PPE_EVERY   = 1           # run the PPE model every N frames
MIN_HITS    = 4           # frames a track must exist before it is drawn
EDGE        = 4           # px: a box touching the frame edge is "cut off"
COVER_T     = 0.20        # overlap (vs the smaller box) needed to call a body part "covered"
PART_GROW   = 0.35        # enlarge body-part boxes by this much before the coverage test
SCENE_CUT_T = 30          # mean pixel change that counts as a scene cut (0 = never reset)

ALPHA_SEEN   = 0.30   # weight of "PPE seen on this worker"
ALPHA_BARE   = 0.20   # weight of "bare body part seen"
ALPHA_MISS   = 0.04   # weight of "should be visible but nothing detected" (vest only)
OK_T, BAD_T  = 0.65, 0.30   # hysteresis thresholds
MIN_EVIDENCE = 5      # updates needed before any verdict for an item

# ---- look & feel ----
TITLE        = "SITE SAFETY"
SUBTITLE     = "AI PPE COMPLIANCE MONITOR"
PANEL_RATIO  = 0.50       # dashboard width as a fraction of video height (0.50 -> 540 px at 1080p)
SHOW_PPE     = True       # draw the PPE / bare-body-part boxes found on each worker
INTRO_S      = 3.0        # seconds the intro title stays on screen (0 = off)
VIGNETTE     = 0.30       # darken the video edges (0 = off)
ALERT_S      = 3.0        # seconds an alert pop-up stays on screen
HISTORY_S    = 12         # seconds shown in the timeline chart
ANALYZE_S    = 0.6        # length of the "ANALYZING" scan before a worker's verdict is shown
GAP_FILL_S   = 0.5        # fill tracking gaps up to this long (s) so boxes don't blink
SMOOTH_K     = 2          # box smoothing: average over +-K frames (0 = off)

ITEMS = {
    "helmet": {"ppe": {"helmet"}, "part": {"head"}, "y": (-0.20, 0.30), "pad_x": 0.10,
               "min_h": 50, "absent": None, "tag": "H", "name": "HELMET",
               "missing": "NO HELMET", "bare": "BARE HEAD"},
    "vest":   {"ppe": {"safety-vest", "vest", "safety-suit"}, "part": set(), "y": (0.10, 0.80),
               "pad_x": 0.05, "min_h": 60, "absent": 0.0, "tag": "V", "name": "VEST",
               "missing": "NO VEST", "bare": ""},
    "gloves": {"ppe": {"gloves", "glove"}, "part": {"hands", "hand"}, "y": (0.10, 0.95),
               "pad_x": 0.35, "min_h": 80, "absent": None, "tag": "G", "name": "GLOVES",
               "missing": "NO GLOVES", "bare": "BARE HAND"},
    "shoes":  {"ppe": {"shoes", "shoe", "boots"}, "part": {"foot", "feet"}, "y": (0.70, 1.10),
               "pad_x": 0.15, "min_h": 80, "absent": None, "tag": "S", "name": "SHOES",
               "missing": "NO SHOES", "bare": "BARE FOOT"},
}
REQUIRED = ["helmet", "vest", "gloves", "shoes"]   # all must be OK for SAFE

# Palette (RGB)
C_BG      = (8, 12, 24)
C_CARD    = (17, 25, 44)
C_LINE    = (34, 47, 74)
C_TEXT    = (234, 240, 250)
C_MUTED   = (128, 142, 170)
C_ACCENT  = (34, 211, 238)    # cyan
C_SAFE    = (34, 229, 139)    # neon green
C_DANGER  = (255, 61, 90)     # hot red
C_WARN    = (255, 176, 32)    # amber
C_UNKNOWN = (91, 107, 134)    # slate
STATUS_COL = {"SAFE": C_SAFE, "VIOLATION": C_DANGER, "ANALYZING": C_WARN, "NOT CLEAR": C_UNKNOWN}
# ================================================================


def norm(name):
    return name.lower().replace("_", "-").replace(" ", "-")


device = 0 if torch.cuda.is_available() else "cpu"
use_half = device == 0
print("Device:", "GPU" if use_half else "CPU (enable GPU: Runtime > Change runtime type)")

ppe_model = YOLO(MODEL_PATH)
names = {i: norm(n) for i, n in ppe_model.names.items()}
print("Model classes:", names)

LOOKUP = {}   # class name -> (item, "ppe" | "part")
for item, cfg in ITEMS.items():
    for n in cfg["ppe"]:
        LOOKUP[n] = (item, "ppe")
    for n in cfg["part"]:
        LOOKUP[n] = (item, "part")
PPE_CLASSES = [i for i, n in names.items() if n in LOOKUP]
for item, cfg in ITEMS.items():
    if not (cfg["ppe"] & set(names.values())):
        print(f"WARNING: no PPE class for '{item}' found in this model")
    if cfg["part"] and not (cfg["part"] & set(names.values())):
        print(f"WARNING: no body-part class for '{item}' -> its violations can't be detected")

person_ids = [i for i, n in names.items() if n == "person"]
if person_ids:
    track_model, PERSON_CLASSES = YOLO(MODEL_PATH), person_ids
    print("Tracking workers with the model's 'Person' class")
else:
    track_model, PERSON_CLASSES = YOLO("yolo11s.pt"), [0]
    print("No 'Person' class -> tracking workers with pretrained yolo11s")

TRACKER_CFG = str(Path(OUTPUT_PATH).with_name("bytetrack_site.yaml"))
Path(TRACKER_CFG).write_text(
    "tracker_type: bytetrack\n"
    f"track_high_thresh: {PERSON_CONF}\n"
    "track_low_thresh: 0.1\n"
    f"new_track_thresh: {PERSON_CONF + 0.05:.2f}\n"
    "track_buffer: 90\n"
    "match_thresh: 0.8\n"
    "fuse_score: True\n"
)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

S = H / 1080.0                      # UI scale: everything is designed for 1080p
PW = int(H * PANEL_RATIO)           # dashboard width
CW = PW + W                         # output canvas width
CW += CW % 2                        # even width for H.264

writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (CW, H))
if not writer.isOpened():
    raise RuntimeError("Could not create output video.")

state = {}          # wid -> item -> {"score", "n", "status"}
hits = {}
first_seen, last_seen = {}, {}
ever_violated = {}
viol_frames = {}    # wid -> item -> consecutive frames in "bad"
counted = set()     # workers tracked for at least MIN_COUNT_S
id_offset = 0       # added to tracker IDs so IDs stay unique after a scene-cut reset
vis_by_frame = {}   # frame -> [(wid, box, item, "ppe"|"bare")]
track_boxes = {}    # wid -> {frame: box}
ev_count = {}       # (wid, item) -> [frames with PPE seen, frames with bare part seen]
alerts = []         # [{"text", "sub", "frame"}]
alert_marks = []    # frames where a new violation was confirmed (timeline markers)
history = deque(maxlen=int(HISTORY_S * fps))   # (n_safe, n_viol, n_check) per frame


# ================================================================ detection logic
def inter(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def area(a):
    return max(1.0, (a[2] - a[0]) * (a[3] - a[1]))


def iou(a, b):
    i = inter(a, b)
    return i / (area(a) + area(b) - i)


def occluded_ids(workers):
    ids, out = list(workers), set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if iou(workers[ids[i]], workers[ids[j]]) > 0.30:
                out.update((ids[i], ids[j]))
    return out


def visible(box, item):
    x1, y1, x2, y2 = box
    if y2 - y1 < ITEMS[item]["min_h"]:
        return False
    if item in ("helmet", "vest") and y1 <= EDGE:
        return False
    if item == "vest" and (y2 - y1) < 1.2 * (x2 - x1):
        return False
    if item == "shoes" and y2 >= H - EDGE:
        return False
    if item == "gloves" and (x1 <= EDGE or x2 >= W - EDGE):
        return False
    return True


def match_worker(pbox, item, workers):
    px1, py1, px2, py2 = pbox
    cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
    top, bottom = ITEMS[item]["y"]
    pad = ITEMS[item]["pad_x"]
    best, best_score = None, -1e9
    for wid, (x1, y1, x2, y2) in workers.items():
        w, h = x2 - x1, y2 - y1
        reg = (x1 - pad * w, y1 + top * h, x2 + pad * w, y1 + bottom * h)
        if not (reg[0] <= cx <= reg[2] and reg[1] <= cy <= reg[3]):
            continue
        if (px2 - px1) > 1.5 * w:
            continue
        score = inter(pbox, reg) / area(pbox) - 0.3 * abs(cx - (x1 + x2) / 2) / max(w, 1)
        if score > best_score:
            best, best_score = wid, score
    return best


def grow(b, f, up_extra=0.0):
    w, h = b[2] - b[0], b[3] - b[1]
    return (b[0] - f * w, b[1] - (f + up_extra) * h, b[2] + f * w, b[3] + f * h)


def covered(part, ppe_boxes, item):
    g = grow(part, PART_GROW, up_extra=0.5 if item == "helmet" else 0.0)
    return any(inter(g, p) / min(area(g), area(p)) >= COVER_T for p in ppe_boxes)


def nms(dets, thr=0.5):
    """Remove duplicate boxes of the same class (the two PPE sizes often find the same object)."""
    keep = []
    for d in sorted(dets, key=lambda t: -t[2]):
        if all(k[1] != d[1] or iou(k[0], d[0]) < thr for k in keep):
            keep.append(d)
    return keep


def detect_ppe(frame):
    dets = []
    for sz in PPE_SIZES:
        r = ppe_model.predict(frame, conf=PPE_CONF, classes=PPE_CLASSES, imgsz=sz,
                              device=device, half=use_half, verbose=False)[0]
        if r.boxes is not None:
            dets += list(zip(r.boxes.xyxy.cpu().tolist(), r.boxes.cls.int().cpu().tolist(),
                             r.boxes.conf.cpu().tolist()))
    return nms(dets)


def skin_fraction(frame, b):
    x1, y1, x2, y2 = (max(0, int(v)) for v in b)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 1.0
    ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    sat = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 1]
    skin = (cv2.inRange(ycc, (0, 135, 85), (255, 180, 135)) > 0) & (sat >= 40)
    return float(skin.mean())


def update_evidence(workers, dets, frame, frame_idx):
    boxes, vis = {}, []
    for xyxy, c, cf in dets:
        item, kind = LOOKUP[names[c]]
        if kind == "part" and cf < PART_CONF:
            continue
        if item == "gloves" and kind == "part":
            if max(xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]) < MIN_HAND_PX:
                continue
            if SKIN_CHECK and skin_fraction(frame, xyxy) < SKIN_T:
                continue
        wid = match_worker(xyxy, item, workers)
        if wid is not None:
            boxes.setdefault((wid, item, kind), []).append(xyxy)

    hidden = occluded_ids(workers)
    for wid, box in workers.items():
        for item, cfg in ITEMS.items():
            rec = state.setdefault(wid, {}).setdefault(item, {"score": 0.5, "n": 0, "status": None})
            ppe = boxes.get((wid, item, "ppe"), [])
            parts = boxes.get((wid, item, "part"), [])
            bare = [p for p in parts if not covered(p, ppe, item)]
            vis += [(wid, p, item, "ppe") for p in ppe] + [(wid, p, item, "bare") for p in bare]
            ec = ev_count.setdefault((wid, item), [0, 0])
            ec[0] += bool(ppe)
            ec[1] += bool(bare)
            if bare:
                target, a = 0.0, ALPHA_BARE
            elif ppe:
                target, a = 1.0, ALPHA_SEEN
            elif cfg["absent"] is not None and wid not in hidden and visible(box, item):
                target, a = cfg["absent"], ALPHA_MISS
            else:
                continue
            rec["score"] += a * (target - rec["score"])
            rec["n"] += 1
            if rec["n"] >= MIN_EVIDENCE:
                if rec["score"] >= OK_T:
                    rec["status"] = "ok"
                elif rec["score"] <= BAD_T:
                    rec["status"] = "bad"
    vis_by_frame[frame_idx] = vis


def final_item_status(wid, item):
    """One final answer per item, from ALL evidence of this worker's track."""
    st = state.get(wid, {}).get(item, {}).get("status")
    if st:
        return st
    ppe_n, bare_n = ev_count.get((wid, item), (0, 0))
    if bare_n >= 2 and bare_n > ppe_n:
        return "bad"
    if ppe_n >= 2 and ppe_n >= bare_n:
        return "ok"
    return None                                   # never clearly visible


def verdict_of(st):
    if any(st[i] == "bad" for i in REQUIRED):
        return "VIOLATION"
    if any(st[i] == "ok" for i in REQUIRED):
        return "SAFE"
    return "NOT CLEAR"


def smooth_track(fb):
    """Fill short gaps by interpolation, then average boxes over +-SMOOTH_K frames."""
    frs = sorted(fb)
    full = {}
    gap = max(1, int(GAP_FILL_S * fps))
    for f0, f1 in zip(frs, frs[1:]):
        full[f0] = fb[f0]
        if 1 < f1 - f0 <= gap:
            a, b = np.array(fb[f0]), np.array(fb[f1])
            for f in range(f0 + 1, f1):
                full[f] = (a + (b - a) * (f - f0) / (f1 - f0)).tolist()
    full[frs[-1]] = fb[frs[-1]]
    keys = sorted(full)
    arr = np.array([full[k] for k in keys], dtype=np.float32)
    if SMOOTH_K > 0 and len(arr) > 2:
        sm = np.empty_like(arr)
        for i in range(len(arr)):
            sm[i] = arr[max(0, i - SMOOTH_K):i + SMOOTH_K + 1].mean(0)
        arr = sm
    return {k: arr[i].tolist() for i, k in enumerate(keys)}


def is_scene_cut(frame, prev_small):
    small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype("float32")
    cut = prev_small is not None and SCENE_CUT_T > 0 and float(abs(small - prev_small).mean()) > SCENE_CUT_T
    return cut, small


def reset_tracker():
    global track_model, id_offset
    id_offset = max([id_offset] + list(hits))
    try:
        for t in track_model.predictor.trackers:
            t.reset()
    except Exception:
        track_model = YOLO(track_model.ckpt_path or MODEL_PATH)


# ================================================================ visuals
def get_fonts():
    """Rajdhani (techy HUD font) from Google Fonts; falls back to system fonts."""
    Path(FONT_DIR).mkdir(parents=True, exist_ok=True)
    found = {}
    for style in ("Bold", "SemiBold", "Medium"):
        p = Path(FONT_DIR) / f"Rajdhani-{style}.ttf"
        if not p.exists():
            try:
                urllib.request.urlretrieve(
                    f"https://raw.githubusercontent.com/google/fonts/main/ofl/rajdhani/Rajdhani-{style}.ttf", p)
            except Exception:
                pass
        try:
            ImageFont.truetype(str(p), 10)
            found[style] = str(p)
        except Exception:
            pass
    fallback = None
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if Path(p).exists():
            fallback = p
            break
    bold = found.get("Bold") or fallback
    semi = found.get("SemiBold") or bold
    med = found.get("Medium") or semi
    print("Fonts:", "Rajdhani" if "Bold" in found else (fallback or "PIL default"))
    return bold, semi, med


FONT_FILES = dict(zip(("bold", "semi", "med"), get_fonts()))
_font_cache = {}


def F(size, w="bold"):
    key = (int(size * S), w)
    if key not in _font_cache:
        path = FONT_FILES[w]
        _font_cache[key] = ImageFont.truetype(path, max(8, key[0])) if path else ImageFont.load_default()
    return _font_cache[key]


def sc(v):
    return int(round(v * S))


def rgba(c, a):
    return (c[0], c[1], c[2], int(max(0, min(1, a)) * 255))


def text_w(txt, font):
    return font.getlength(txt)


def draw_text(d, xy, txt, font, fill, anchor="la"):
    d.text(xy, txt, font=font, fill=fill, anchor=anchor)


def tick(d, cx, cy, r, col, width):
    d.line([(cx - r, cy), (cx - r * 0.3, cy + r * 0.7), (cx + r, cy - r * 0.8)], fill=col, width=width, joint="curve")


def cross(d, cx, cy, r, col, width):
    d.line([(cx - r, cy - r), (cx + r, cy + r)], fill=col, width=width)
    d.line([(cx - r, cy + r), (cx + r, cy - r)], fill=col, width=width)


def qmark(d, cx, cy, r, col):
    draw_text(d, (cx, cy), "?", F(r * 2.4 / S, "bold"), col, "mm")


def status_icon(d, cx, cy, r, status, col):
    if status == "ok":
        tick(d, cx, cy, r, col, max(2, sc(3)))
    elif status == "bad":
        cross(d, cx, cy, r * 0.8, col, max(2, sc(3)))
    else:
        qmark(d, cx, cy, r, col)


def item_col(status):
    return C_SAFE if status == "ok" else C_DANGER if status == "bad" else C_UNKNOWN


def dashed_rect(d, b, col, width, dash):
    x1, y1, x2, y2 = b
    for (ax, ay, bx, by) in [(x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)]:
        length = math.hypot(bx - ax, by - ay)
        n = max(1, int(length // (dash * 2)))
        for k in range(n + 1):
            t0, t1 = (2 * k * dash) / length, min(1, (2 * k + 1) * dash / length)
            if t0 >= 1:
                break
            d.line([(ax + (bx - ax) * t0, ay + (by - ay) * t0), (ax + (bx - ax) * t1, ay + (by - ay) * t1)],
                   fill=col, width=width)


def brackets(d, b, col, alpha, width, L):
    x1, y1, x2, y2 = b
    for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        d.line([(px, py + dy * L), (px, py), (px + dx * L, py)], fill=rgba(col, alpha), width=width, joint="curve")


def make_panel_bg():
    """Static dashboard background: dark vertical gradient + faint grid + cyan edge."""
    g = np.linspace(0, 1, H)[:, None]
    top, bot = np.array(C_BG, float), np.array((14, 20, 38), float)
    bg = (top * (1 - g) + bot * g)[:, None, :].repeat(PW, axis=1)
    step = sc(36)
    bg[::step, :, :] += 6
    bg[:, ::step, :] += 6
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    bg[:, PW - sc(3):PW] = C_ACCENT
    return bg                                    # RGB


def make_vignette():
    if VIGNETTE <= 0:
        return None
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / math.sqrt(2)
    return (1 - VIGNETTE * np.clip(r, 0, 1) ** 2.2)[..., None].astype(np.float32)


PANEL_BG = make_panel_bg()
VIG = make_vignette()


def card(d, b, title=None, accent=C_ACCENT):
    d.rounded_rectangle(b, radius=sc(14), fill=rgba(C_CARD, 0.92), outline=rgba(C_LINE, 1), width=max(1, sc(2)))
    if title:
        x1, y1 = b[0] + sc(18), b[1] + sc(12)
        d.rounded_rectangle((x1, y1 + sc(4), x1 + sc(5), y1 + sc(22)), radius=sc(2), fill=accent)
        draw_text(d, (x1 + sc(14), y1), title, F(22, "bold"), C_MUTED)


def draw_dashboard(d, frame_idx, rows, n_total, counts, item_counts):
    pad = sc(22)
    x0, x1 = pad, PW - pad - sc(3)
    t = frame_idx / fps
    pulse = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 1.2)

    # ---- header
    draw_text(d, (x0, sc(18)), TITLE, F(56, "bold"), C_TEXT)
    draw_text(d, (x0 + sc(2), sc(80)), SUBTITLE, F(21, "semi"), C_ACCENT)
    lx = x1 - sc(92)
    d.rounded_rectangle((lx, sc(30), x1, sc(64)), radius=sc(17), fill=rgba(C_DANGER, 0.18),
                        outline=rgba(C_DANGER, 0.9), width=max(1, sc(2)))
    d.ellipse((lx + sc(12), sc(40), lx + sc(26), sc(54)), fill=rgba(C_DANGER, 0.35 + 0.65 * pulse))
    draw_text(d, (lx + sc(34), sc(47)), "LIVE", F(22, "bold"), C_TEXT, "lm")
    mm, ss = divmod(int(t), 60)
    draw_text(d, (x1, sc(84)), f"{mm:02d}:{ss:02d}.{int((t % 1) * 10)}", F(20, "med"), C_MUTED, "ra")

    # ---- compliance gauge + worker counts
    y = sc(112)
    card(d, (x0, y, x1, y + sc(180)), "COMPLIANCE")
    n_now = len(rows)
    cx, cy, r = x0 + sc(100), y + sc(106), sc(60)
    d.arc((cx - r, cy - r, cx + r, cy + r), 0, 360, fill=rgba(C_LINE, 1), width=sc(16))
    decided = counts["SAFE"] + counts["VIOLATION"]
    if decided:
        pct = counts["SAFE"] / decided
        col = C_SAFE if pct >= 0.8 else C_WARN if pct >= 0.5 else C_DANGER
        if pct > 0:
            d.arc((cx - r, cy - r, cx + r, cy + r), -90, -90 + 360 * pct, fill=col, width=sc(16))
        draw_text(d, (cx, cy - sc(6)), f"{int(round(pct * 100))}%", F(40, "bold"), col, "mm")
        draw_text(d, (cx, cy + sc(24)), "SAFE", F(17, "semi"), C_MUTED, "mm")
    else:
        draw_text(d, (cx, cy), "--", F(44, "bold"), C_MUTED, "mm")
    rx = x0 + sc(200)
    draw_text(d, (rx, y + sc(44)), "WORKERS ON SCREEN", F(19, "semi"), C_MUTED)
    draw_text(d, (rx, y + sc(56)), str(n_now), F(68, "bold"), C_ACCENT)
    draw_text(d, (rx, y + sc(140)), "TOTAL SEEN", F(19, "semi"), C_MUTED, "lm")
    draw_text(d, (rx + sc(104), y + sc(140)), str(n_total), F(36, "bold"), C_TEXT, "lm")

    # ---- status tiles
    y += sc(194)
    tw = (x1 - x0 - 2 * sc(12)) / 3
    for k, (label, key) in enumerate((("SAFE", "SAFE"), ("VIOLATION", "VIOLATION"), ("ANALYZING", "ANALYZING"))):
        bx = x0 + k * (tw + sc(12))
        col = STATUS_COL[key]
        hot = key == "VIOLATION" and counts[key] > 0
        fill = rgba(col, 0.16 + 0.22 * pulse) if hot else rgba(C_CARD, 0.92)
        d.rounded_rectangle((bx, y, bx + tw, y + sc(90)), radius=sc(14), fill=fill,
                            outline=rgba(col, 0.9 if counts[key] else 0.35), width=max(1, sc(2)))
        d.rounded_rectangle((bx + sc(14), y, bx + tw - sc(14), y + sc(5)), radius=sc(2), fill=col)
        draw_text(d, (bx + tw / 2, y + sc(40)), str(counts[key]), F(50, "bold"),
                  col if counts[key] else C_MUTED, "mm")
        draw_text(d, (bx + tw / 2, y + sc(74)), label, F(18, "bold"), C_TEXT, "mm")

    # ---- PPE breakdown
    y += sc(104)
    card(d, (x0, y, x1, y + sc(200)), "PPE CHECK  (workers on screen)")
    by = y + sc(48)
    for item in REQUIRED:
        ok_n, bad_n, unk_n = item_counts[item]
        ix = x0 + sc(20)
        d.rounded_rectangle((ix, by, ix + sc(34), by + sc(34)), radius=sc(8), fill=rgba(C_ACCENT, 0.15),
                            outline=rgba(C_ACCENT, 0.8), width=max(1, sc(2)))
        draw_text(d, (ix + sc(17), by + sc(17)), ITEMS[item]["tag"], F(22, "bold"), C_ACCENT, "mm")
        draw_text(d, (ix + sc(46), by + sc(17)), ITEMS[item]["name"], F(22, "bold"), C_TEXT, "lm")
        bx1, bx2 = x0 + sc(180), x1 - sc(96)
        d.rounded_rectangle((bx1, by + sc(9), bx2, by + sc(25)), radius=sc(8), fill=rgba(C_LINE, 1))
        tot = ok_n + bad_n + unk_n
        if tot:
            cur = bx1
            for n, col in ((ok_n, C_SAFE), (bad_n, C_DANGER), (unk_n, C_UNKNOWN)):
                if n:
                    w = (bx2 - bx1) * n / tot
                    d.rounded_rectangle((cur, by + sc(9), cur + w, by + sc(25)), radius=sc(8), fill=col)
                    cur += w
        tx = x1 - sc(86)
        tick(d, tx + sc(8), by + sc(17), sc(7), C_SAFE, max(2, sc(3)))
        draw_text(d, (tx + sc(20), by + sc(17)), str(ok_n), F(22, "bold"), C_TEXT, "lm")
        cross(d, tx + sc(48), by + sc(17), sc(6), C_DANGER, max(2, sc(3)))
        draw_text(d, (tx + sc(60), by + sc(17)), str(bad_n), F(22, "bold"),
                  C_DANGER if bad_n else C_TEXT, "lm")
        by += sc(37)

    # ---- worker list
    y += sc(214)
    list_h = sc(200)
    card(d, (x0, y, x1, y + list_h), "WORKERS")
    ry = y + sc(46)
    max_rows = 4
    shown = sorted(rows, key=lambda r: ({"VIOLATION": 0, "ANALYZING": 1, "SAFE": 2, "NOT CLEAR": 3}[r[1]], r[0]))
    if not shown:
        draw_text(d, ((x0 + x1) / 2, y + list_h / 2 + sc(14)), "no workers in view", F(22, "med"), C_MUTED, "mm")
    for wid, verdict, st, missing in shown[:max_rows]:
        col = STATUS_COL[verdict]
        d.rounded_rectangle((x0 + sc(14), ry, x1 - sc(14), ry + sc(34)), radius=sc(10),
                            fill=rgba(col, 0.10), outline=rgba(col, 0.55), width=max(1, sc(1)))
        d.rounded_rectangle((x0 + sc(14), ry, x0 + sc(20), ry + sc(34)), radius=sc(3), fill=col)
        draw_text(d, (x0 + sc(32), ry + sc(17)), f"#{wid}", F(24, "bold"), C_TEXT, "lm")
        cx_ = x0 + sc(100)
        for item in REQUIRED:
            ic = item_col(st[item])
            d.rounded_rectangle((cx_, ry + sc(4), cx_ + sc(50), ry + sc(30)), radius=sc(7),
                                fill=rgba(ic, 0.22), outline=rgba(ic, 0.9), width=max(1, sc(1)))
            draw_text(d, (cx_ + sc(14), ry + sc(17)), ITEMS[item]["tag"], F(20, "bold"), C_TEXT, "mm")
            status_icon(d, cx_ + sc(36), ry + sc(17), sc(7), st[item], ic)
            cx_ += sc(56)
        draw_text(d, (x1 - sc(26), ry + sc(17)), verdict, F(20, "bold"), col, "rm")
        ry += sc(37)
    if len(shown) > max_rows:
        draw_text(d, (x1 - sc(26), y + sc(16)), f"+{len(shown) - max_rows} more", F(18, "semi"), C_MUTED, "ra")

    # ---- timeline chart
    y += list_h + sc(12)
    ch = sc(96)
    card(d, (x0, y, x1, y + ch), f"LAST {HISTORY_S}s")
    lg = x1 - sc(20)
    for lab, col in (("violation", C_DANGER), ("safe", C_SAFE)):
        w = text_w(lab, F(17, "semi"))
        draw_text(d, (lg, y + sc(16)), lab, F(17, "semi"), C_MUTED, "ra")
        d.ellipse((lg - w - sc(16), y + sc(20), lg - w - sc(6), y + sc(30)), fill=col)
        lg -= w + sc(30)
    gx1, gx2, gy1, gy2 = x0 + sc(20), x1 - sc(20), y + sc(44), y + ch - sc(12)
    if len(history) > 1:
        peak = max(2, max(max(h[0], h[1]) for h in history))
        n = history.maxlen
        for idx, col in ((0, C_SAFE), (1, C_DANGER)):
            pts = [(gx1 + (gx2 - gx1) * (n - len(history) + k) / (n - 1),
                    gy2 - (gy2 - gy1) * h[idx] / peak) for k, h in enumerate(history)]
            d.polygon(pts + [(pts[-1][0], gy2), (pts[0][0], gy2)], fill=rgba(col, 0.16))
            d.line(pts, fill=col, width=max(2, sc(3)), joint="curve")
    d.line([(gx1, gy2), (gx2, gy2)], fill=rgba(C_LINE, 1), width=1)

    # ---- legend (how to read the video)
    y += ch + sc(14)
    lx = x0 + sc(6)
    for status, lab in (("ok", "wearing"), ("bad", "missing"), (None, "not visible yet")):
        ic = item_col(status)
        d.rounded_rectangle((lx, y, lx + sc(26), y + sc(26)), radius=sc(6), fill=rgba(ic, 0.22),
                            outline=ic, width=max(1, sc(1)))
        status_icon(d, lx + sc(13), y + sc(13), sc(7), status, ic)
        draw_text(d, (lx + sc(34), y + sc(13)), lab, F(19, "semi"), C_TEXT, "lm")
        lx += sc(48) + text_w(lab, F(19, "semi"))
    y += sc(36)
    lx = x0 + sc(6)
    d.rectangle((lx, y + sc(3), lx + sc(26), y + sc(23)), outline=C_SAFE, width=max(1, sc(2)))
    draw_text(d, (lx + sc(34), y + sc(13)), "PPE found", F(19, "semi"), C_TEXT, "lm")
    lx += sc(150)
    dashed_rect(d, (lx, y + sc(3), lx + sc(26), y + sc(23)), C_DANGER, max(1, sc(2)), sc(4))
    draw_text(d, (lx + sc(34), y + sc(13)), "bare body part", F(19, "semi"), C_TEXT, "lm")
    draw_text(d, (x0 + sc(6), H - sc(14)), "H helmet  V vest  G gloves  S shoes",
              F(18, "med"), C_MUTED, "ld")


def draw_worker(d, b, wid, verdict, st, missing, t, frame_idx):
    col = STATUS_COL[verdict]
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    pulse = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 1.5 + wid)
    L = max(sc(16), int(min(w, h) * 0.2))

    # body tint + thin full outline
    if verdict == "VIOLATION":
        d.rectangle(b, fill=rgba(col, 0.05 + 0.10 * pulse))
    d.rectangle(b, outline=rgba(col, 0.35), width=max(1, sc(1)))
    # glowing corner brackets (wide faint + narrow bright)
    glow_a = 0.25 + (0.25 * pulse if verdict == "VIOLATION" else 0)
    brackets(d, b, col, glow_a, sc(12), L)
    brackets(d, b, col, 1.0, max(2, sc(4)), L)

    # scanning line while still checking
    if verdict == "ANALYZING":
        sy = y1 + (frame_idx * max(2, h / (fps * 1.2))) % max(1, h)
        for k in range(sc(14)):
            d.line([(x1, sy - k), (x2, sy - k)], fill=rgba(col, 0.45 * (1 - k / sc(14))), width=1)

    # tag: [#id STATUS] + item chips
    f_id, f_st = F(24, "bold"), F(20, "bold")
    th = sc(36)
    tag_w = sc(16) + text_w(f"#{wid}", f_id) + sc(10) + text_w(verdict, f_st) + sc(14)
    chips_w = len(REQUIRED) * sc(44) + sc(6)
    tx = max(x1, PW + sc(4))
    tx = min(tx, CW - (tag_w + chips_w) - sc(4))
    ty = y1 - th - sc(8) if y1 - th - sc(8) > 0 else y2 + sc(8)
    d.rounded_rectangle((tx, ty, tx + tag_w + chips_w, ty + th), radius=sc(10), fill=rgba(C_BG, 0.82),
                        outline=rgba(col, 0.95), width=max(1, sc(2)))
    d.rounded_rectangle((tx, ty, tx + tag_w, ty + th), radius=sc(10), fill=rgba(col, 0.95))
    draw_text(d, (tx + sc(12), ty + th / 2), f"#{wid}", f_id, C_BG, "lm")
    draw_text(d, (tx + sc(12) + text_w(f"#{wid}", f_id) + sc(10), ty + th / 2), verdict, f_st, C_BG, "lm")
    cx = tx + tag_w + sc(6)
    for item in REQUIRED:
        ic = item_col(st[item])
        d.rounded_rectangle((cx, ty + sc(5), cx + sc(38), ty + th - sc(5)), radius=sc(6),
                            fill=rgba(ic, 0.30), outline=ic, width=max(1, sc(1)))
        draw_text(d, (cx + sc(11), ty + th / 2), ITEMS[item]["tag"], F(18, "bold"), C_TEXT, "mm")
        status_icon(d, cx + sc(27), ty + th / 2, sc(6), st[item], ic)
        cx += sc(44)

    # violation callout under the worker
    if missing:
        txt = "  +  ".join(ITEMS[i]["missing"] for i in missing)
        f = F(24, "bold")
        cw = text_w(txt, f) + sc(56)
        cx1 = min(max((x1 + x2) / 2 - cw / 2, PW + sc(4)), CW - cw - sc(4))
        cy1 = min(y2 + sc(10), H - sc(48))
        if ty > y2:                                     # tag already sits below the box
            cy1 = max(y1 - sc(50), 0) if y1 > sc(60) else min(ty + th + sc(8), H - sc(48))
        d.rounded_rectangle((cx1, cy1, cx1 + cw, cy1 + sc(38)), radius=sc(19),
                            fill=rgba(C_DANGER, 0.75 + 0.25 * pulse))
        warn_icon(d, cx1 + sc(22), cy1 + sc(19), sc(11), C_TEXT)
        draw_text(d, (cx1 + sc(40), cy1 + sc(19)), txt, f, C_TEXT, "lm")


def warn_icon(d, cx, cy, r, col):
    d.polygon([(cx, cy - r), (cx + r, cy + r * 0.85), (cx - r, cy + r * 0.85)], outline=col, width=max(2, sc(2)))
    d.line([(cx, cy - r * 0.35), (cx, cy + r * 0.3)], fill=col, width=max(2, sc(2)))
    d.ellipse((cx - sc(1.5), cy + r * 0.5, cx + sc(1.5), cy + r * 0.5 + sc(3)), fill=col)


def draw_ppe_boxes(d, boxes):
    f = F(16, "bold")
    for b, item, kind in boxes:
        bb = (b[0] + PW, b[1], b[2] + PW, b[3])
        if kind == "ppe":
            col, lab = C_SAFE, ITEMS[item]["name"]
            d.rectangle(bb, outline=rgba(col, 0.9), width=max(1, sc(2)))
        else:
            col, lab = C_DANGER, ITEMS[item]["bare"]
            dashed_rect(d, bb, rgba(col, 1), max(2, sc(3)), sc(6))
        lw = text_w(lab, f) + sc(10)
        ly = bb[1] - sc(20) if bb[1] > sc(22) else bb[3] + sc(2)
        d.rectangle((bb[0], ly, bb[0] + lw, ly + sc(18)), fill=rgba(col, 0.85))
        draw_text(d, (bb[0] + sc(5), ly + sc(9)), lab, f, C_BG, "lm")


def draw_alerts(d, frame_idx):
    live = [a for a in alerts if frame_idx - a["frame"] < ALERT_S * fps][-3:]
    y = sc(24)
    for a in reversed(live):
        age = (frame_idx - a["frame"]) / fps
        slide = min(1.0, age / 0.25)                       # slide in
        fade = 1.0 if age < ALERT_S - 0.5 else max(0.0, (ALERT_S - age) / 0.5)
        w, h = sc(360), sc(78)
        x = CW - sc(24) - w + (1 - slide) * (w + sc(24))
        d.rounded_rectangle((x, y, x + w, y + h), radius=sc(14), fill=rgba(C_BG, 0.88 * fade),
                            outline=rgba(C_DANGER, fade), width=max(2, sc(3)))
        d.rounded_rectangle((x, y, x + sc(10), y + h), radius=sc(4), fill=rgba(C_DANGER, fade))
        warn_icon(d, x + sc(42), y + h / 2, sc(16), rgba(C_DANGER, fade))
        draw_text(d, (x + sc(72), y + sc(12)), "SAFETY ALERT  -  " + a["sub"], F(18, "semi"),
                  rgba(C_MUTED, fade))
        draw_text(d, (x + sc(72), y + sc(34)), a["text"], F(32, "bold"), rgba(C_TEXT, fade))
        y += h + sc(12)


def draw_intro(d, frame_idx):
    t = frame_idx / fps
    if INTRO_S <= 0 or t > INTRO_S:
        return
    a = 1.0 if t < INTRO_S - 0.8 else max(0.0, (INTRO_S - t) / 0.8)
    cx = PW + W / 2
    w, h = sc(860), sc(140)
    y = H * 0.12
    d.rounded_rectangle((cx - w / 2, y, cx + w / 2, y + h), radius=sc(20), fill=rgba(C_BG, 0.78 * a),
                        outline=rgba(C_ACCENT, a), width=max(2, sc(3)))
    draw_text(d, (cx, y + sc(48)), "AI CONSTRUCTION SAFETY MONITOR", F(50, "bold"), rgba(C_TEXT, a), "mm")
    draw_text(d, (cx, y + sc(102)), "HELMET  -  VEST  -  GLOVES  -  SHOES  -  WORKER COUNT",
              F(24, "semi"), rgba(C_ACCENT, a), "mm")


def draw_progress(d, frame_idx):
    y = H - sc(8)
    d.rectangle((PW, y, CW, H), fill=rgba(C_BG, 0.7))
    if total > 0:
        d.rectangle((PW, y, PW + W * frame_idx / total, H), fill=rgba(C_ACCENT, 0.9))
        for m in alert_marks:
            mx = PW + W * m / total
            d.rectangle((mx - sc(2), y - sc(8), mx + sc(2), H), fill=C_DANGER)


def render(frame_bgr, frame_idx, workers, rows, n_total, counts, item_counts, ppe_boxes):
    t = frame_idx / fps
    video = frame_bgr
    if VIG is not None:
        video = (frame_bgr.astype(np.float32) * VIG).astype(np.uint8)
    canvas = np.empty((H, CW, 3), np.uint8)
    canvas[:, :PW] = PANEL_BG
    canvas[:, PW:PW + W] = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
    if CW > PW + W:
        canvas[:, PW + W:] = C_BG
    base = Image.fromarray(canvas).convert("RGBA")
    over = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)

    if SHOW_PPE:
        draw_ppe_boxes(d, ppe_boxes)
    for wid, verdict, st, missing in rows:
        b = workers[wid]
        draw_worker(d, (b[0] + PW, b[1], b[2] + PW, b[3]), wid, verdict, st, missing, t, frame_idx)
    draw_progress(d, frame_idx)
    draw_alerts(d, frame_idx)
    draw_intro(d, frame_idx)
    draw_dashboard(d, frame_idx, rows, n_total, counts, item_counts)

    out = Image.alpha_composite(base, over).convert("RGB")
    return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)


# ================================================================ PASS 1: analyse
print("PASS 1/2: analysing the video ...")
frame_idx, prev_small, n_cuts = 0, None, 0
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        cut, prev_small = is_scene_cut(frame, prev_small)
        if cut:
            reset_tracker()
            n_cuts += 1

        tr = track_model.track(frame, persist=True, tracker=TRACKER_CFG, classes=PERSON_CLASSES,
                               conf=0.1, imgsz=IMGSZ, device=device, half=use_half, verbose=False)[0]
        workers = {}
        if tr.boxes is not None and tr.boxes.id is not None:
            for box, tid in zip(tr.boxes.xyxy.cpu().tolist(), tr.boxes.id.int().cpu().tolist()):
                wid = tid + id_offset
                hits[wid] = hits.get(wid, 0) + 1
                first_seen.setdefault(wid, frame_idx)
                last_seen[wid] = frame_idx
                track_boxes.setdefault(wid, {})[frame_idx] = box
                if hits[wid] >= MIN_HITS:
                    workers[wid] = box

        if workers and frame_idx % PPE_EVERY == 0:
            update_evidence(workers, detect_ppe(frame), frame, frame_idx)

        if frame_idx % 50 == 0:
            print(f"  analysed {frame_idx}/{total} frames")
finally:
    cap.release()
total = frame_idx or total

# ================================================================ final verdicts
min_len = max(MIN_HITS, MIN_COUNT_S * fps)
tracks = {wid: smooth_track(fb) for wid, fb in track_boxes.items() if len(fb) >= min_len}
counted = set(tracks)
final = {}
reveal = {}            # wid -> (scan start frame, verdict frame)
for wid, fb in tracks.items():
    st = {i: final_item_status(wid, i) for i in REQUIRED}
    final[wid] = (st, verdict_of(st))
    f0, f1 = min(fb), max(fb)
    scan = min(int(ANALYZE_S * fps), max(1, int((f1 - f0) * 0.4)))
    reveal[wid] = (f0, f0 + scan)
    missing = [i for i in REQUIRED if st[i] == "bad"]
    if missing:
        ever_violated[wid] = {ITEMS[i]["missing"] for i in missing}
        for i in missing:
            alerts.append({"text": ITEMS[i]["missing"], "sub": f"WORKER #{wid}", "frame": f0 + scan})
        alert_marks.append(f0 + scan)
alerts.sort(key=lambda a: a["frame"])
print(f"Workers: {len(tracks)} | verdicts:",
      {w: final[w][1] for w in sorted(final)})


def ppe_boxes_for(frame_idx, shown):
    """PPE / bare boxes to draw: latest within 3 frames, and only if they agree with the verdict."""
    out, seen = [], set()
    for f in range(frame_idx, frame_idx - 3, -1):
        for wid, b, item, kind in vis_by_frame.get(f, []):
            if wid not in shown or (wid, item, kind) in seen:
                continue
            st = shown[wid].get(item)
            if (kind == "ppe" and st == "ok") or (kind == "bare" and st == "bad"):
                out.append((b, item, kind))
        seen |= {(w, i, k) for w, _, i, k in vis_by_frame.get(f, [])}
    return out


# ================================================================ PASS 2: render
print("PASS 2/2: rendering ...")
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        workers, rows, shown = {}, [], {}
        counts = {"SAFE": 0, "VIOLATION": 0, "ANALYZING": 0, "NOT CLEAR": 0}
        item_counts = {i: [0, 0, 0] for i in REQUIRED}      # wearing, missing, not visible
        for wid, fb in tracks.items():
            if frame_idx not in fb:
                continue
            workers[wid] = fb[frame_idx]
            st_final, verdict_final = final[wid]
            s0, s1 = reveal[wid]
            if frame_idx < s1:                                 # scanning: reveal chips one by one
                step = (s1 - s0) / len(REQUIRED)
                st = {i: (st_final[i] if frame_idx >= s0 + (k + 1) * step else None)
                      for k, i in enumerate(REQUIRED)}
                verdict = "ANALYZING"
            else:
                st, verdict = st_final, verdict_final
            missing = [i for i in REQUIRED if st[i] == "bad"] if verdict == "VIOLATION" else []
            rows.append((wid, verdict, st, missing))
            shown[wid] = st
            counts[verdict] += 1
            if verdict != "ANALYZING":
                for i in REQUIRED:
                    item_counts[i][0 if st[i] == "ok" else 1 if st[i] == "bad" else 2] += 1
        history.append((counts["SAFE"], counts["VIOLATION"], counts["ANALYZING"]))
        n_total = sum(1 for w in tracks if reveal[w][0] <= frame_idx)

        writer.write(render(frame, frame_idx, workers, rows, n_total, counts, item_counts,
                            ppe_boxes_for(frame_idx, shown)))
        if frame_idx % 50 == 0:
            print(f"  rendered {frame_idx}/{total} frames")
finally:
    cap.release()
    writer.release()

# ================================================================ report
print("Scene cuts detected:", n_cuts)
with open(REPORT_PATH, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["worker_id", "first_frame", "last_frame", "seconds_on_screen", "verdict"]
                + REQUIRED + ["violations"])
    for wid in sorted(counted):
        st, verdict = final[wid]
        wr.writerow([wid, first_seen[wid], last_seen[wid], round(hits[wid] / fps, 1), verdict]
                    + [{"ok": "wearing", "bad": "MISSING"}.get(st[i], "not visible") for i in REQUIRED]
                    + ["; ".join(sorted(ever_violated.get(wid, []))) or "-"])

print("\nSaved:", OUTPUT_PATH, "| exists:", Path(OUTPUT_PATH).exists())
print("Total workers seen:", len(counted))
print("Workers ever flagged:", {k: sorted(v) for k, v in ever_violated.items()} or "none")
print("Report:", REPORT_PATH)

try:
    web = OUTPUT_PATH.replace(".mp4", "_web.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", OUTPUT_PATH,
                    "-vcodec", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", web], check=True)
    print("Browser-friendly copy:", web)
except Exception as e:
    print("ffmpeg conversion skipped:", e)
