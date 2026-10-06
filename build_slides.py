"""
Research slide deck — Optical Tactile Sensing Research
Plain academic style: black / white / grey only.
Widescreen 16:9 (13.33 × 7.5 inches)

Run:  python3 build_slides.py
Out:  TactileSensing_Research.pptx
"""

import os
from PIL import Image as PILImage
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "TactileSensing_Research.pptx")

# ─── Slide geometry ────────────────────────────────────────────────────────────
SW, SH  = Inches(13.33), Inches(7.5)
LM      = Inches(0.52)        # left / right margin
CW      = SW - 2 * LM        # usable width  ≈ 12.29"
BTOP    = Inches(1.24)        # body top  (below header rule)
FBOT    = Inches(7.15)        # body bottom (above footer bar)
BH      = FBOT - BTOP         # usable body height  ≈ 5.91"

# Two-column grid
COL_W   = Inches(5.88)        # width of each column
COL_GAP = Inches(0.41)        # gap between columns
RC      = LM + COL_W + COL_GAP   # right-column left edge  ≈ 6.81"

# ─── Colour palette ────────────────────────────────────────────────────────────
INK  = RGBColor(0x0D, 0x0D, 0x0D)   # near-black — body text
SUB  = RGBColor(0x55, 0x55, 0x55)   # mid grey   — secondary text / captions
LITE = RGBColor(0xC0, 0xC0, 0xC0)   # light grey — rules / table borders
BG   = RGBColor(0xF4, 0xF4, 0xF4)   # near-white — alt table rows / panels
HDR  = RGBColor(0x18, 0x18, 0x18)   # near-black — table header cells
WHT  = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Calibri"

prs = Presentation()
prs.slide_width  = SW
prs.slide_height = SH
BLANK = prs.slide_layouts[6]   # completely blank


# ══════════════════════════════════════════════════════════════════════════════
#   PRIMITIVE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def S():
    """New blank white slide."""
    sl = prs.slides.add_slide(BLANK)
    bg = sl.background.fill
    bg.solid(); bg.fore_color.rgb = WHT
    return sl

def BOX(s, l, t, w, h, fill=WHT, border=None, border_w=Pt(0.5)):
    """Solid rectangle, no default border."""
    sh = s.shapes.add_shape(1, l, t, w, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if border:
        sh.line.color.rgb = border
        sh.line.width = border_w
    else:
        sh.line.fill.background()
    return sh

def RULE(s, y, color=LITE, l=None, w=None):
    """Thin horizontal rule (implemented as a 1-pt-tall rectangle)."""
    l = l if l is not None else LM
    w = w if w is not None else CW
    r = s.shapes.add_shape(1, l, y, w, Inches(0.006))
    r.fill.solid(); r.fill.fore_color.rgb = color
    r.line.fill.background()

def TXT(s, text, l, t, w, h,
        sz=13, bold=False, italic=False,
        clr=INK, align=PP_ALIGN.LEFT):
    """Simple single-paragraph text box."""
    bx = s.shapes.add_textbox(l, t, w, h)
    bx.word_wrap = True
    tf = bx.text_frame; tf.word_wrap = True
    p  = tf.paragraphs[0]; p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = FONT; run.font.size = Pt(sz)
    run.font.bold = bold; run.font.italic = italic
    run.font.color.rgb = clr
    return bx

def IMG(s, rel_path, l, t, w, h=None):
    """
    Insert image. h=None → compute from aspect ratio.
    rel_path is relative to HERE.
    """
    fp = os.path.join(HERE, rel_path) if not os.path.isabs(rel_path) else rel_path
    if not os.path.exists(fp):
        BOX(s, l, t, w, h or Inches(2), fill=BG, border=LITE)
        TXT(s, f"[missing: {os.path.basename(fp)}]",
            l + Inches(0.1), t + Inches(0.1),
            w - Inches(0.2), Inches(0.3), sz=8, italic=True, clr=SUB)
        return 0
    im   = PILImage.open(fp)
    ar   = im.size[0] / im.size[1]
    if h is None:
        h = w / ar
    s.shapes.add_picture(fp, l, t, w, h)
    return h   # return actual height placed

def CAP(s, text, l, t, w):
    """Image caption — small italic grey, centred under the image."""
    TXT(s, text, l, t, w, Inches(0.30), sz=8, italic=True,
        clr=SUB, align=PP_ALIGN.CENTER)

def HEADER(s, title, sub=None):
    """
    Standard slide header:
      • 0.07" black accent bar at top
      • Title text
      • Optional italic subtitle
      • 1-pt black rule at y=1.10"
    """
    BOX(s, 0, 0, SW, Inches(0.07), fill=INK)
    TXT(s, title, LM, Inches(0.12), CW, Inches(0.60),
        sz=26, bold=True, clr=INK)
    if sub:
        TXT(s, sub, LM, Inches(0.72), CW, Inches(0.28),
            sz=9, italic=True, clr=SUB)
    RULE(s, Inches(1.10), color=INK)

def FOOTER(s, note="Optical Tactile Sensing Research"):
    """Pale grey footer bar at bottom of every content slide."""
    BOX(s, 0, FBOT, SW, SH - FBOT, fill=BG)
    RULE(s, FBOT, color=LITE)
    TXT(s, note, LM, FBOT + Inches(0.07), CW, Inches(0.25),
        sz=8, clr=SUB)

def SECHEAD(s, text, l, t, width=None):
    """
    Section sub-heading with a left accent bar.
    Returns the bottom y of this element.
    """
    width = width or COL_W
    h     = Inches(0.28)
    BOX(s, l, t, Inches(0.04), h, fill=INK)
    TXT(s, text, l + Inches(0.12), t, width - Inches(0.12), h,
        sz=13, bold=True, clr=INK)
    return t + h + Inches(0.06)   # return next y


def BULLETS(s, items, l, t, w, h, sz=12):
    """
    Bullet list. Items starting with "  " (two spaces) are sub-bullets.
    Returns the textbox.
    """
    bx = s.shapes.add_textbox(l, t, w, h)
    bx.word_wrap = True
    tf = bx.text_frame; tf.word_wrap = True
    first = True
    for item in items:
        sub   = item.startswith("  ")
        clean = item.strip()
        if first: p = tf.paragraphs[0]; first = False
        else:      p = tf.add_paragraph()
        p.alignment    = PP_ALIGN.LEFT
        p.space_before = Pt(2 if sub else 5)
        run = p.add_run(); run.font.name = FONT
        if sub:
            run.text           = "    ◦  " + clean
            run.font.size      = Pt(sz - 1)
            run.font.color.rgb = SUB
        else:
            run.text           = "•  " + clean
            run.font.size      = Pt(sz)
            run.font.color.rgb = INK
    return bx


def TABLE(s, hdrs, rows, l, t, w,
          hdr_sz=10, row_sz=10, row_h=Inches(0.37)):
    """
    Draw a plain table with black header row and alternating body rows.
    Returns the bottom y of the table.
    """
    nc   = len(hdrs)
    cw   = w // nc
    hh   = Inches(0.40)
    # Header
    for ci, h in enumerate(hdrs):
        BOX(s, l + ci * cw, t, cw, hh, fill=HDR)
        TXT(s, h,
            l + ci * cw + Inches(0.08), t + Inches(0.08),
            cw - Inches(0.16), hh - Inches(0.10),
            sz=hdr_sz, bold=True, clr=WHT)
    # Body rows
    for ri, row in enumerate(rows):
        rt = t + hh + ri * row_h
        bg = WHT if ri % 2 == 0 else BG
        for ci, cell in enumerate(row):
            BOX(s, l + ci * cw, rt, cw, row_h, fill=bg, border=LITE)
            TXT(s, str(cell),
                l + ci * cw + Inches(0.08), rt + Inches(0.06),
                cw - Inches(0.16), row_h - Inches(0.08),
                sz=row_sz, clr=INK)
    return t + hh + len(rows) * row_h


def MONO_BLOCK(s, lines, l, t, w):
    """Render an architecture text block in a monospaced font."""
    bx = s.shapes.add_textbox(l, t, w, Inches(len(lines) * 0.29 + 0.15))
    bx.word_wrap = False
    tf = bx.text_frame; tf.word_wrap = False
    first = True
    for line in lines:
        if first: p = tf.paragraphs[0]; first = False
        else:      p = tf.add_paragraph()
        p.alignment    = PP_ALIGN.LEFT
        p.space_before = Pt(1)
        run = p.add_run()
        run.text           = line
        run.font.name      = "Courier New"
        run.font.size      = Pt(10)
        run.font.color.rgb = SUB if line.strip() in ("↓", "↓↓") else INK
    return bx


def STAT_BOX(s, value, label, l, t, w=Inches(2.5), h=Inches(1.0)):
    """A highlighted stat: large number + small label."""
    BOX(s, l, t, w, h, fill=BG, border=LITE)
    TXT(s, value, l, t + Inches(0.12), w, Inches(0.48),
        sz=26, bold=True, clr=INK, align=PP_ALIGN.CENTER)
    TXT(s, label, l, t + Inches(0.58), w, Inches(0.38),
        sz=9, clr=SUB, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
#   SLIDE 1 — Title
# ══════════════════════════════════════════════════════════════════════════════
s = S()
BOX(s, 0, 0, SW, Inches(0.10), fill=INK)
BOX(s, 0, SH - Inches(0.10), SW, Inches(0.10), fill=INK)

TXT(s, "Optical Tactile Sensing Research",
    LM, Inches(1.60), CW, Inches(1.30),
    sz=40, bold=True, clr=INK, align=PP_ALIGN.CENTER)

RULE(s, Inches(3.10), color=LITE, l=Inches(2.5), w=Inches(8.33))

TXT(s,
    "Contact estimation  ·  dense spatial mapping  ·  3D skin geometry reconstruction\n"
    "from visual deformation of a patterned elastomer skin",
    LM, Inches(3.25), CW, Inches(0.85),
    sz=16, clr=SUB, align=PP_ALIGN.CENTER)

TXT(s, "Ishan Ahluwalia   ·   Sung Robotics Group",
    LM, Inches(4.80), CW, Inches(0.38),
    sz=12, clr=SUB, align=PP_ALIGN.CENTER)
TXT(s, "2026",
    LM, Inches(5.22), CW, Inches(0.32),
    sz=11, clr=LITE, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
#   SLIDE 2 — Project Overview
# ══════════════════════════════════════════════════════════════════════════════
s = S(); HEADER(s, "Project Overview"); FOOTER(s)

# Left column: description + pipeline stages
y = BTOP
TXT(s,
    "A low-cost, camera-based tactile sensor that uses the visual deformation "
    "of a patterned elastomer skin to infer contact properties. "
    "Three progressively richer models are built on the same hardware.",
    LM, y, COL_W, Inches(0.82), sz=12, clr=INK)

y += Inches(0.92)
y = SECHEAD(s, "Research Stages", LM, y)

stages = [
    ("Stage 1 — ResNet18 Regressor",
     "Predict 4 scalars per frame: contact location (x, y), indentation depth, and force."),
    ("Stage 2 — DenseContactNet",
     "Predict full spatial maps (contact probability, depth profile, pressure map) "
     "over a 33×37 sensor grid, plus the same 4 scalars."),
    ("Stage 3 — PointCloudNet",
     "Predict the complete 3D deformed skin geometry as a 2048-point structured cloud."),
]
for title, desc in stages:
    BOX(s, LM, y, Inches(0.05), Inches(0.92), fill=INK)
    TXT(s, title, LM + Inches(0.14), y, COL_W - Inches(0.14), Inches(0.32),
        sz=12, bold=True, clr=INK)
    TXT(s, desc, LM + Inches(0.14), y + Inches(0.30), COL_W - Inches(0.14), Inches(0.60),
        sz=11, clr=SUB)
    y += Inches(1.05)

y += Inches(0.10)
BULLETS(s, [
    "Single USB camera + load cell — sensor cost < $50",
    "All models run live at 30 fps on Apple Silicon (MPS)",
    "Ground truth: actuator displacement + load cell force per frame",
], LM, y, COL_W, Inches(0.90), sz=11)

# Right column: project pipeline image (2750×2582, ratio ≈ 1.065)
img_w = COL_W
img_h = img_w / (2750/2582)   # ≈ 5.52"
img_t = BTOP + (BH - img_h) / 2   # vertically centred
IMG(s, "assets/figures/pipeline.png", RC, img_t, img_w, img_h)
CAP(s, "End-to-end pipeline: raw frame → extraction → contact estimation → dense maps → 3D geometry",
    RC, img_t + img_h + Inches(0.05), COL_W)


# ══════════════════════════════════════════════════════════════════════════════
#   SLIDE 3 — Hardware Setup
# ══════════════════════════════════════════════════════════════════════════════
s = S(); HEADER(s, "Hardware Setup"); FOOTER(s)

# Left: component table + operating principle
TABLE(s,
      ["Component", "Specification"],
      [
          ["Actuator",      "MakerBot 3D printer — repurposed as precision linear stage"],
          ["Camera",        "USB 640×480 @ 30 fps — looks up at skin underside"],
          ["Load cell",     "SparkFun NAU7802 Qwiic Scale @ 40 SPS"],
          ["MCU",           "Arduino RedBoard Qwiic — 115 200 baud serial"],
          ["Skin",          "Transparent elastomer with printed dot array"],
          ["Press speed",   "10 mm/min — 0 → 10 mm in 60 s per session"],
      ],
      LM, BTOP, COL_W, hdr_sz=11, row_sz=11, row_h=Inches(0.38))

y = BTOP + Inches(0.40) + 6 * Inches(0.38) + Inches(0.18)
y = SECHEAD(s, "Operating Principle", LM, y)
BULLETS(s, [
    "Actuator indents skin at a known (x, y) position at 10 mm/min",
    "Camera captures dot-field deformation at 30 fps",
    "Arduino streams force readings over serial, synchronised to frames",
    "Ground truth per frame: actuator position (mm) + load cell force (N)",
], LM, y, COL_W, Inches(1.10), sz=11)

# Right: two hardware photos side-by-side
ph_w = (COL_W - Inches(0.20)) / 2   # ≈ 2.84" each
ph_h = ph_w / (900/675)             # ≈ 2.13"
ph_t = BTOP
IMG(s, "assets/hardware/hardware_front.jpg", RC, ph_t, ph_w, ph_h)
IMG(s, "assets/hardware/hardware_top.jpg",   RC + ph_w + Inches(0.20), ph_t, ph_w, ph_h)
CAP(s, "Front: skin mounted in actuator frame",
    RC, ph_t + ph_h + Inches(0.04), ph_w)
CAP(s, "Top: Arduino + NAU7802 load cell",
    RC + ph_w + Inches(0.20), ph_t + ph_h + Inches(0.04), ph_w)

# Description text below photos
y2 = ph_t + ph_h + Inches(0.35)
y2 = SECHEAD(s, "Sensor Details", RC, y2, COL_W)
BULLETS(s, [
    "Camera looks up through the transparent skin body",
    "Dot array on skin surface deforms under contact",
    "Load cell measures total contact force to ±5 mN",
    "Serial timestamp aligns force samples to camera frames",
    "Tare via 't' command at any time during recording",
], RC, y2, COL_W, Inches(1.75), sz=11)


# ══════════════════════════════════════════════════════════════════════════════
#   SLIDE 4 — Data Collection Protocol
# ══════════════════════════════════════════════════════════════════════════════
s = S(); HEADER(s, "Data Collection Protocol", "dataset/record.py"); FOOTER(s)

# Stat boxes across top
stats = [
    ("9",         "contact positions"),
    ("~453",      "frames / session"),
    ("4 074",     "total labeled frames"),
    ("0 – 10 mm", "displacement range"),
    ("10 mm/min", "press speed"),
]
bw = CW / len(stats)
for i, (val, lbl) in enumerate(stats):
    STAT_BOX(s, val, lbl, LM + i * bw, BTOP, w=bw - Inches(0.08), h=Inches(0.88))

y_after_stats = BTOP + Inches(0.88) + Inches(0.18)

# Left column: recording protocol
y = y_after_stats
y = SECHEAD(s, "Recording Protocol", LM, y)
BULLETS(s, [
    "Actuator presses to 9 y-positions: 0, −2, −4, … −16 mm  (all at x = −140 mm)",
    "Each session: 0 → 10 mm at 10 mm/min, then retract",
    "Camera records at 30 fps throughout the press",
    "Arduino logs force + timestamp at 40 SPS over USB serial",
    "record.py synchronises streams and saves per-frame CSV + images",
], LM, y, COL_W, Inches(1.55), sz=11)

y += Inches(1.65)
y = SECHEAD(s, "CSV Row Format (per frame)", LM, y)
MONO_BLOCK(s, [
    "time_s,  displacement_mm,  frame,  force_n,  image_path,  extracted_path",
    "0.000,   0.0000,           0,      0.000,    frames/frame_00000.jpg, ...",
    "0.128,   0.0213,           1,      0.001,    frames/frame_00001.jpg, ...",
    " ...",
    "60.060,  10.010,           452,    1.847,    frames/frame_00452.jpg, ...",
], LM, y, COL_W)

# Right column: location map + ROI preview
y_r = y_after_stats
y_r = SECHEAD(s, "Sensor Grid & ROI", RC, y_r)
loc_w = COL_W
loc_h = IMG(s, "dense_contact/assets/location_map.png", RC, y_r, loc_w)
y_r  += loc_h + Inches(0.05)
CAP(s, "Sensor contact grid — all 9 recorded y-positions shown on the skin surface",
    RC, y_r, loc_w)
y_r += Inches(0.32)

roi_h = IMG(s, "dense_contact/assets/roi_preview.png", RC, y_r, loc_w)
y_r  += roi_h + Inches(0.05)
CAP(s, "Camera ROI (region of interest) — the cropped input window fed to every model",
    RC, y_r, loc_w)


# ══════════════════════════════════════════════════════════════════════════════
#   SLIDE 5 — Tactile Skin & Pattern Extraction
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HEADER(s, "Tactile Skin & Pattern Extraction", "preprocessing/live_extraction.py")
FOOTER(s)

# Left: sensor description
y = BTOP
y = SECHEAD(s, "How the Sensor Works", LM, y)
BULLETS(s, [
    "Transparent elastomer skin with a printed dot array on its inner surface",
    "Camera mounted beneath the skin looks up through the transparent body",
    "Contact deforms the dots — shifts and compressions encode geometry and force",
    "No strain gauges or pressure arrays needed — dots are the only signal",
], LM, y, COL_W, Inches(1.30), sz=12)

y += Inches(1.40)
y = SECHEAD(s, "Extraction Pipeline  (4 stages)", LM, y)

steps = [
    ("CLAHE",
     "Contrast-limited adaptive histogram equalisation normalises local contrast, "
     "compensating for uneven illumination across the skin."),
    ("Adaptive threshold",
     "Pixels brighter than their local neighbourhood are marked as dot centres — "
     "works regardless of global brightness."),
    ("Brightness floor",
     "Hard minimum intensity of 130 rejects dim pixels that pass the adaptive "
     "threshold but belong to background noise."),
    ("Morphological opening",
     "Erosion then dilation removes single-pixel speckle while preserving the "
     "full dot shape and size."),
]
for i, (title, desc) in enumerate(steps):
    ty = y + i * Inches(0.98)
    BOX(s, LM, ty + Inches(0.05), Inches(0.04), Inches(0.26), fill=INK)
    TXT(s, f"{i+1}.  {title}",
        LM + Inches(0.13), ty, COL_W - Inches(0.13), Inches(0.30),
        sz=12, bold=True, clr=INK)
    TXT(s, desc,
        LM + Inches(0.13), ty + Inches(0.30), COL_W - Inches(0.13), Inches(0.60),
        sz=11, clr=SUB)

# Right: wide before/after image (719×200, ratio 3.595)
pe_w = COL_W
pe_h = pe_w / (719/200)   # ≈ 1.64"
pe_t = BTOP + Inches(0.05)
IMG(s, "assets/figures/pattern_extraction.jpg", RC, pe_t, pe_w, pe_h)
CAP(s, "Left: raw camera frame   ·   Right: extracted binary dot pattern (network input)",
    RC, pe_t + pe_h + Inches(0.05), pe_w)

# Sample raw + extracted stacked below
y_r = pe_t + pe_h + Inches(0.38)
y_r = SECHEAD(s, "Single-Frame Examples", RC, y_r)

sr_w = (COL_W - Inches(0.20)) / 2
sr_h = sr_w / (978/767)        # sample_raw ratio ≈ 1.275 → h ≈ 2.21"
se_h = sr_w / (1806/926)       # sample_extracted ratio ≈ 1.950 → h ≈ 1.45"

# Keep the taller one, truncate if needed
max_img_h = FBOT - Inches(0.45) - y_r
actual_h  = min(sr_h, max_img_h)
se_actual = sr_w / (1806/926)
se_actual = min(se_actual, max_img_h)

IMG(s, "dense_contact/assets/sample_raw.png",       RC, y_r, sr_w, actual_h)
IMG(s, "dense_contact/assets/sample_extracted.png", RC + sr_w + Inches(0.20), y_r, sr_w, se_actual)
CAP(s, "Raw",       RC, y_r + actual_h + Inches(0.03), sr_w)
CAP(s, "Extracted", RC + sr_w + Inches(0.20), y_r + se_actual + Inches(0.03), sr_w)


# ══════════════════════════════════════════════════════════════════════════════
#   SLIDE 6 — Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════
s = S(); HEADER(s, "Dataset Overview"); FOOTER(s)

# Left: session table
y = BTOP
y = SECHEAD(s, "Recording Sessions", LM, y, COL_W)
TABLE(s,
      ["Session", "Y-pos", "Force range", "Frames"],
      [
          ["(-140, 0)",   "0 mm",    "0 – 2.38 N", "453"],
          ["(-140, −2)",  "−2 mm",   "0 – 2.13 N", "453"],
          ["(-140, −4)",  "−4 mm",   "0 – 1.52 N", "453"],
          ["(-140, −6)",  "−6 mm*",  "0 – 0.62 N", "453"],
          ["(-140, −8)",  "−8 mm",   "0 – 0.50 N", "453"],
          ["(-140,−10)", "−10 mm",  "0 – 0.52 N", "453"],
          ["(-140,−12)", "−12 mm",  "0 – 0.56 N", "453"],
          ["(-140,−14)", "−14 mm*", "0 – 1.27 N", "453"],
          ["(-140,−16)", "−16 mm",  "0 – 2.02 N", "453"],
      ],
      LM, y, COL_W, hdr_sz=10, row_sz=10, row_h=Inches(0.36))

tbl_bottom = y + Inches(0.40) + 9 * Inches(0.36)
TXT(s, "* marked sessions withheld from training — used as unseen validation only",
    LM, tbl_bottom + Inches(0.08), COL_W, Inches(0.30), sz=9, italic=True, clr=SUB)

# Right: dataset samples image + description
y_r = BTOP
y_r = SECHEAD(s, "Sample Grid", RC, y_r)
ds_w = COL_W
ds_h = ds_w / (1214/374)   # ratio 3.246 → h ≈ 1.81"
IMG(s, "assets/figures/dataset_samples.jpg", RC, y_r, ds_w, ds_h)
y_r += ds_h + Inches(0.05)
CAP(s,
    "Grid: rows = contact location, cols = depth (1 / 5 / 9 mm). "
    "Left half = raw frame, right half = extracted pattern.",
    RC, y_r, ds_w)

y_r += Inches(0.34)
y_r = SECHEAD(s, "Key Statistics", RC, y_r)
BULLETS(s, [
    "9 sessions × 453 frames = 4,074 labeled frames total",
    "Displacement 0 – 10 mm  |  Force 0 – 2.38 N",
    "Frame rate: 30 fps  |  Force sampling: 40 SPS",
    "Validation holdout: y = −6 mm and y = −14 mm (full sessions)",
    "Data volume: ~28 GB raw recordings, ~15 GB extracted images",
], RC, y_r, COL_W, Inches(1.60), sz=11)


# ══════════════════════════════════════════════════════════════════════════════
#   SLIDE 7 — Force–Displacement Characteristics
# ══════════════════════════════════════════════════════════════════════════════
s = S(); HEADER(s, "Force–Displacement Characteristics"); FOOTER(s)

TXT(s,
    "Each session records a full 0→10 mm press at 10 mm/min. "
    "Force response varies strongly with contact y-position: "
    "central contact (y=0) sees the stiffest skin; off-centre contacts engage "
    "less material, producing lower peak forces. "
    "The non-linearity reflects Hertz contact mechanics on the curved elastomer skin.",
    LM, BTOP, CW, Inches(0.72), sz=12, clr=INK)

plot_files = [
    "assets/plots/force_vs_displacement_x138.png",
    "assets/plots/force_vs_displacement_x140.png",
    "assets/plots/force_vs_displacement_x142.png",
    "assets/plots/force_vs_displacement_x144.png",
    "assets/plots/force_vs_displacement_x146.png",
    "assets/plots/force_vs_displacement_x202.png",
    "assets/plots/force_vs_displacement_x204.png",
    "assets/plots/force_vs_displacement_x206.png",
    "assets/plots/force_vs_displacement_x208.png",
    "assets/plots/force_vs_displacement_x210.png",
]
labels = [
    "y = 0 mm",  "y = −2 mm", "y = −4 mm", "y = −6 mm", "y = −8 mm",
    "y = −8 mm (x202)", "y = −10 mm", "y = −12 mm", "y = −14 mm", "y = −16 mm",
]
# Two rows of 5
n_cols = 5
p_w    = (CW - (n_cols - 1) * Inches(0.12)) / n_cols   # ≈ 2.35"
p_ar   = 1484 / 882    # ≈ 1.682
p_h    = p_w / p_ar    # ≈ 1.40"
row_y  = [BTOP + Inches(0.82), BTOP + Inches(0.82) + p_h + Inches(0.36)]

for idx, (pf, lbl) in enumerate(zip(plot_files, labels)):
    row = idx // n_cols
    col = idx %  n_cols
    px  = LM + col * (p_w + Inches(0.12))
    py  = row_y[row]
    # clip to footer
    if py + p_h + Inches(0.30) <= FBOT:
        IMG(s, pf, px, py, p_w, p_h)
        CAP(s, lbl, px, py + p_h + Inches(0.04), p_w)


# ══════════════════════════════════════════════════════════════════════════════
#   SLIDE 8 — Contact Estimation: ResNet18 Architecture
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HEADER(s, "Stage 1 — Contact Estimation: ResNet18",
       "contact_estimation/train_model.py")
FOOTER(s)

# Left: architecture + training config
y = BTOP
y = SECHEAD(s, "Architecture", LM, y)
MONO_BLOCK(s, [
    "Input  : 224×224 extracted pattern  (RGB)",
    "↓",
    "ResNet18  backbone — pretrained ImageNet, fine-tuned",
    "↓",
    "GlobalAveragePool  →  Dropout(0.4)",
    "↓",
    "Linear(512→128)  →  ReLU  →  Linear(128→4)",
    "↓",
    "Outputs:  [ loc_x (mm),  loc_y (mm),  disp (mm),  force (N) ]",
], LM, y, COL_W)
y += Inches(2.75)

y = SECHEAD(s, "Design Rationale", LM, y)
BULLETS(s, [
    "ResNet18 chosen for strong feature extraction on texture/dot patterns",
    "Pre-trained backbone reduces data requirement for fine-tuning",
    "Four outputs share the same backbone — joint training improves all tasks",
    "Dropout before the head prevents co-adaptation on the small dataset",
    "Differential LR: slow backbone updates preserve low-level features",
], LM, y, COL_W, Inches(1.65), sz=11)

# Right: training configuration table
y_r = BTOP
y_r = SECHEAD(s, "Training Configuration", RC, y_r)
TABLE(s,
      ["Parameter", "Value"],
      [
          ["Loss",               "L1 (MAE) on all 4 normalised outputs"],
          ["Backbone LR",        "1 × 10⁻⁵  (fine-tune slowly)"],
          ["Head LR",            "1 × 10⁻⁴"],
          ["Scheduler",          "Cosine annealing over 150 epochs"],
          ["Batch size",         "32"],
          ["Augmentation",       "H/V flip, ±8° rotation"],
          ["Val holdout",        "y = −6 mm and y = −14 mm (full sessions)"],
          ["Early stopping",     "Patience = 25 epochs"],
          ["Best epoch",         "131 / 150"],
          ["Framework",          "PyTorch — MPS (Apple Silicon)"],
      ],
      RC, y_r, COL_W, hdr_sz=10, row_sz=10, row_h=Inches(0.36))


# ══════════════════════════════════════════════════════════════════════════════
#   SLIDE 9 — ResNet18 Results
# ══════════════════════════════════════════════════════════════════════════════
s = S(); HEADER(s, "Stage 1 — Results: Unseen Contact Locations"); FOOTER(s)

TXT(s,
    "Validation uses y = −6 mm and y = −14 mm — two complete press sessions that "
    "the model never saw during training. This evaluates generalisation to unseen "
    "contact positions, not just unseen frames from known positions.",
    LM, BTOP, CW, Inches(0.68), sz=12, clr=INK)

y = BTOP + Inches(0.78)

# Left: MAE table + takeaways
y = SECHEAD(s, "Validation MAE (unseen locations)", LM, y)
TABLE(s,
      ["Output", "Train MAE", "Val MAE", "Val / Train ratio"],
      [
          ["Location X",    "—",        "—",        "not reported"],
          ["Location Y",    "0.34 mm",  "0.52 mm",  "1.53×"],
          ["Displacement",  "0.23 mm",  "0.38 mm",  "1.65×"],
          ["Force",         "0.026 N",  "0.055 N",  "2.12×"],
      ],
      LM, y, COL_W, hdr_sz=10, row_sz=11, row_h=Inches(0.37))

tbl_b = y + Inches(0.40) + 4 * Inches(0.37)
y2 = tbl_b + Inches(0.20)
y2 = SECHEAD(s, "Key Findings", LM, y2)
BULLETS(s, [
    "Location Y MAE of 0.52 mm on positions never seen during training",
    "Displacement MAE of 0.38 mm across the full 0–10 mm range",
    "Force MAE of 55 mN — sufficient for contact detection and estimation",
    "Val/train ratio < 2.2× — model generalises, does not simply memorise",
    "Training used 9 sessions; val used 2 fully withheld sessions",
], LM, y2, COL_W, Inches(1.70), sz=11)

# Right: prediction traces image (2100×1200, ratio 1.75)
y_r = BTOP + Inches(0.78)
tr_w = COL_W
tr_h = min(tr_w / (2100/1200), FBOT - Inches(0.38) - y_r)
IMG(s, "contact_estimation/assets/prediction_traces.png", RC, y_r, tr_w, tr_h)
CAP(s,
    "Full-press traces (0→10 mm) for both unseen val sessions. "
    "White = ground truth · colour = prediction · shading = error.",
    RC, y_r + tr_h + Inches(0.05), tr_w)


# ══════════════════════════════════════════════════════════════════════════════
#   SLIDE 10 — ResNet18 Performance Plots
# ══════════════════════════════════════════════════════════════════════════════
s = S(); HEADER(s, "Stage 1 — Performance Visualisation"); FOOTER(s)

TXT(s,
    "Predicted vs. actual across all sessions (train = blue, unseen val = orange). "
    "Bottom panel shows how MAE evolves with indentation depth — "
    "error tends to be higher at shallow depths where deformation signals are weaker.",
    LM, BTOP, CW, Inches(0.62), sz=12, clr=INK)

perf_w = COL_W
perf_h = min(perf_w / (2400/1800), FBOT - Inches(0.38) - BTOP - Inches(0.70))
IMG(s, "contact_estimation/assets/performance.png",
    LM, BTOP + Inches(0.72), perf_w, perf_h)
CAP(s, "Top: predicted vs actual.   Middle: residuals.   Bottom: MAE vs depth.",
    LM, BTOP + Inches(0.72) + perf_h + Inches(0.05), perf_w)

# Right: comparison images if available
y_r = BTOP + Inches(0.72)
y_r = SECHEAD(s, "Deglare Comparison", RC, y_r)

cmp_files = [
    ("assets/figures/comparison.png",           "No deglare"),
    ("assets/figures/comparison_deglare.png",   "With deglare"),
    ("assets/figures/comparison_deglare_model.png", "Model output"),
]
# Stack vertically — each image is wide (approx 3:1)
for path, lbl in cmp_files:
    if os.path.exists(os.path.join(HERE, path)):
        im = PILImage.open(os.path.join(HERE, path))
        ar = im.size[0] / im.size[1]
        cw2 = COL_W
        ch  = min(cw2 / ar, Inches(1.45))
        if y_r + ch + Inches(0.28) > FBOT:
            break
        IMG(s, path, RC, y_r, cw2, ch)
        y_r += ch + Inches(0.05)
        CAP(s, lbl, RC, y_r, cw2)
        y_r += Inches(0.28)


# ══════════════════════════════════════════════════════════════════════════════
#   SLIDE 11 — Grad-CAM
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HEADER(s, "Stage 1 — Grad-CAM: What the Model Attends To",
       "contact_estimation/explain.py")
FOOTER(s)

TXT(s,
    "Grad-CAM visualises the gradient of each output with respect to the last "
    "convolutional feature map, revealing which regions of the dot pattern drive "
    "each prediction. Rows = indentation depth (1.5 / 5 / 9 mm). "
    "Columns: raw pattern  |  Location Y attention  |  Displacement attention  |  Force attention.",
    LM, BTOP, CW, Inches(0.72), sz=12, clr=INK)

gc_w = CW
gc_h = min(gc_w / (2194/1410), FBOT - Inches(0.40) - BTOP - Inches(0.78))
IMG(s, "contact_estimation/assets/gradcam.png",
    LM, BTOP + Inches(0.80), gc_w, gc_h)
CAP(s,
    "Bright regions = high gradient magnitude. "
    "Force attention concentrates near the contact centre; "
    "location attention extends along the gradient axis.",
    LM, BTOP + Inches(0.80) + gc_h + Inches(0.05), gc_w)


# ══════════════════════════════════════════════════════════════════════════════
#   SLIDE 12 — DenseContactNet: Motivation
# ══════════════════════════════════════════════════════════════════════════════
s = S(); HEADER(s, "Stage 2 — Dense Contact Estimation: Motivation"); FOOTER(s)

# Left
y = BTOP
y = SECHEAD(s, "Limitation of Scalar Outputs", LM, y)
BULLETS(s, [
    "ResNet18 predicts 4 numbers per frame — contact is reduced to a single point",
    "No information about contact area, pressure distribution, or depth profile",
    "Multi-finger or edge contacts cannot be distinguished from point contacts",
    "Scalar outputs discard spatial structure that is directly visible in the dot field",
], LM, y, COL_W, Inches(1.30), sz=12)

y += Inches(1.42)
y = SECHEAD(s, "DenseContactNet Goal", LM, y)
BULLETS(s, [
    "Predict the full spatial distribution of contact across the sensor surface",
    "Output 3 spatial maps simultaneously at 33×37 grid resolution:",
    "  contact_map   — probability of contact at each grid point",
    "  depth_map     — normalised Hertz indentation depth",
    "  pressure_map  — normalised Hertz contact pressure",
    "Retain the 4 scalar outputs for direct comparison with Stage 1",
], LM, y, COL_W, Inches(1.80), sz=12)

y += Inches(1.92)
y = SECHEAD(s, "Sensor Grid", LM, y)
BULLETS(s, [
    "37 columns: X = 138 to 210 mm in 2 mm steps",
    "33 rows: Y = 0 to 16 mm in 0.5 mm steps",
    "Grid cell (i, j) maps to physical location (X[j], Y[i]) on skin",
], LM, y, COL_W, Inches(0.95), sz=11)

# Right: contact grid map (2139×654, ratio 3.27)
y_r = BTOP
y_r = SECHEAD(s, "Predicted Contact Grid Map", RC, y_r)
cg_w = COL_W
cg_h = cg_w / (2139/654)   # ≈ 1.81"
IMG(s, "assets/figures/contact_grid_map.png", RC, y_r, cg_w, cg_h)
y_r += cg_h + Inches(0.05)
CAP(s,
    "Each cell = predicted contact probability at that sensor location. "
    "Heatmap aggregated over the validation set.",
    RC, y_r, cg_w)

y_r += Inches(0.35)
y_r = SECHEAD(s, "Why Spatial Maps Matter", RC, y_r)
BULLETS(s, [
    "Contact area changes with indentation depth (Hertz theory: a ∝ depth^½)",
    "Pressure peaks at contact centre and falls to zero at edge",
    "Spatial maps let downstream tasks (e.g. grasping) use full contact geometry",
    "Soft-argmax on contact_map gives differentiable location estimate",
    "Maps provide interpretable intermediate representation for debugging",
], RC, y_r, COL_W, Inches(1.65), sz=11)


# ══════════════════════════════════════════════════════════════════════════════
#   SLIDE 13 — DenseContactNet: Architecture
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HEADER(s, "Stage 2 — DenseContactNet Architecture", "dense_contact/model.py")
FOOTER(s)

# Left: encoder + decoder description
y = BTOP
y = SECHEAD(s, "Encoder — ResNet18 Backbone", LM, y)
BULLETS(s, [
    "Input: (1, 224, 224) grayscale extracted pattern",
    "Pretrained on ImageNet — skip connections at layer1, 2, 3, 4",
    "Feature maps: 64ch @ 56×56, 128ch @ 28×28, 256ch @ 14×14, 512ch @ 7×7",
    "Fine-tuned at lr = 1×10⁻⁵ (10× slower than decoder)",
], LM, y, COL_W, Inches(1.22), sz=11)

y += Inches(1.32)
y = SECHEAD(s, "Decoder — U-Net with Skip Connections", LM, y)
BULLETS(s, [
    "4 decoder blocks, each: bilinear upsample → concat skip → 2× Conv-BN-ReLU",
    "Block 1: 512+256ch → 256ch  (14×14 → 28×28)",
    "Block 2: 256+128ch → 128ch  (28×28 → 56×56)",
    "Block 3: 128+64ch  → 64ch   (56×56 → 112×112)",
    "Block 4: 64+64ch   → 32ch   (112×112 → 224×224)",
    "Decoder LR = 1×10⁻⁴ — 10× faster than backbone",
], LM, y, COL_W, Inches(1.60), sz=11)

y += Inches(1.72)
y = SECHEAD(s, "Output Heads", LM, y)
BULLETS(s, [
    "3 × 1×1 Conv → Sigmoid → adaptive pool to (33, 37): contact / depth / pressure maps",
    "Soft-argmax on contact_map → loc_x, loc_y (differentiable location)",
    "Global average pool + 2-layer MLP → displacement, force (scalars)",
], LM, y, COL_W, Inches(0.95), sz=11)

# Right: architecture summary table
y_r = BTOP
y_r = SECHEAD(s, "Output Summary", RC, y_r)
TABLE(s,
      ["Output", "Shape", "Range", "Head"],
      [
          ["contact_map",  "(33, 37)", "[0, 1]", "Conv → Sigmoid → pool"],
          ["depth_map",    "(33, 37)", "[0, 1]", "Conv → Sigmoid → pool"],
          ["pressure_map", "(33, 37)", "[0, 1]", "Conv → Sigmoid → pool"],
          ["loc_x",        "scalar",  "mm",     "soft-argmax on contact_map"],
          ["loc_y",        "scalar",  "mm",     "soft-argmax on contact_map"],
          ["displacement", "scalar",  "[0, 1]", "GAP → MLP(512→128→1)"],
          ["force",        "scalar",  "[0, 1]", "GAP → MLP(512→128→1)"],
      ],
      RC, y_r, COL_W, hdr_sz=9, row_sz=9, row_h=Inches(0.36))

tbl_b = y_r + Inches(0.40) + 7 * Inches(0.36)
y_r   = tbl_b + Inches(0.20)
y_r   = SECHEAD(s, "Why U-Net?", RC, y_r)
BULLETS(s, [
    "Skip connections preserve fine spatial detail lost during downsampling",
    "Encoder captures global context; decoder recovers spatial resolution",
    "Same architecture succeeds in semantic segmentation on natural images",
    "Spatial maps require full-resolution output — pooling alone is insufficient",
], RC, y_r, COL_W, Inches(1.30), sz=11)


# ══════════════════════════════════════════════════════════════════════════════
#   SLIDE 14 — DenseContactNet: Ground-Truth Synthesis & Loss
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HEADER(s, "Stage 2 — Ground-Truth Synthesis & Loss Function",
       "dense_contact/dataset.py  ·  dense_contact/train.py")
FOOTER(s)

# Left: how GT maps are synthesized
y = BTOP
y = SECHEAD(s, "Ground-Truth Map Synthesis", LM, y)
TXT(s,
    "No spatial ground truth is available — contact maps are synthesized "
    "analytically from scalar labels (loc_x, loc_y, displacement) using "
    "Hertz contact mechanics:",
    LM, y, COL_W, Inches(0.58), sz=11, clr=SUB)
y += Inches(0.68)

synth_steps = [
    ("Contact map",
     "2D Gaussian centred at (loc_x, loc_y) with σ = indentor radius (10 mm). "
     "Peak = 1.0; decays smoothly to 0 at the contact boundary."),
    ("Depth map",
     "Hertz parabolic profile within the contact patch: "
     "depth(r) = δ₀ × max(0, 1 − r²/a²), where a = contact radius "
     "and δ₀ = peak indentation. Normalised by displacement_max."),
    ("Pressure map",
     "Hertz ellipsoidal distribution: p(r) = p₀ × √max(0, 1 − r²/a²). "
     "Normalised by pressure_max. Integrates to total force over the patch."),
]
for title, desc in synth_steps:
    BOX(s, LM, y + Inches(0.06), Inches(0.04), Inches(0.88), fill=INK)
    TXT(s, title, LM + Inches(0.13), y, COL_W - Inches(0.13), Inches(0.28),
        sz=12, bold=True, clr=INK)
    TXT(s, desc, LM + Inches(0.13), y + Inches(0.28), COL_W - Inches(0.13), Inches(0.65),
        sz=11, clr=SUB)
    y += Inches(1.00)

# Right: loss function table
y_r = BTOP
y_r = SECHEAD(s, "Loss Function", RC, y_r)
TXT(s, "L  =  λ_c·MSE(contact_map) + λ_d·MSE(depth_map) + λ_p·MSE(pressure_map)\n"
       "    + λ_dp·L1(displacement) + λ_f·L1(force) + λ_x·L1(loc_x) + λ_y·L1(loc_y)",
    RC, y_r, COL_W, Inches(0.58), sz=10, clr=INK)
y_r += Inches(0.65)

TABLE(s,
      ["Term", "λ", "Rationale"],
      [
          ["MSE(contact_map)",  "1.0", "Spatial probability — equal to depth/pressure"],
          ["MSE(depth_map)",    "1.0", "All maps on same [0,1] scale"],
          ["MSE(pressure_map)", "1.0", "All maps on same [0,1] scale"],
          ["L1(displacement)",  "0.5", "Half weight vs spatial maps"],
          ["L1(force)",         "0.5", "Half weight vs spatial maps"],
          ["L1(loc_x)",         "0.5", "X range = 72 mm — easier task"],
          ["L1(loc_y)",         "2.0", "Y range = 16 mm — 4× harder; upweighted"],
      ],
      RC, y_r, COL_W, hdr_sz=9, row_sz=9, row_h=Inches(0.36))

tbl_b = y_r + Inches(0.40) + 7 * Inches(0.36)
y_r   = tbl_b + Inches(0.20)
y_r   = SECHEAD(s, "Training Setup", RC, y_r)
BULLETS(s, [
    "26,699 train frames  |  6,598 val frames",
    "Batch size 32, up to 150 epochs, early stopping patience 25",
    "Val holdout: y = 6 mm and y = 14 mm (full sessions withheld)",
    "Same cosine LR schedule and differential LR as Stage 1",
], RC, y_r, COL_W, Inches(1.10), sz=11)


# ══════════════════════════════════════════════════════════════════════════════
#   SLIDE 15 — Physical Skin Model for 3D Reconstruction
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HEADER(s, "Stage 3 — Physical Skin Model for Point Cloud Generation",
       "reconstruction/generate_pointclouds.py  ·  reconstruction/contact_sim.py")
FOOTER(s)

# Left
y = BTOP
y = SECHEAD(s, "Why Synthesize Point Clouds?", LM, y)
BULLETS(s, [
    "No depth sensor in the hardware — 3D ground truth must be computed",
    "Physical deformation model converts scalar labels to 3D geometry",
    "Model is parameterized by known geometry: indentor radius, skin curvature",
    "Allows 2,331 sessions × 453 frames = ~1.06M ground-truth clouds",
], LM, y, COL_W, Inches(1.18), sz=11)

y += Inches(1.28)
y = SECHEAD(s, "Skin Rest Geometry", LM, y)
BULLETS(s, [
    "Modelled as a spherical cap: radius R₀ = 60 mm, half-angle θ_max = 45°",
    "Approximates the real sensor's curved elastomer surface",
    "32 × 64 structured grid of (X, Y, Z) points sampled analytically",
    "rest_positions.npy stores the 2048 × 3 undeformed cloud",
], LM, y, COL_W, Inches(1.18), sz=11)

y += Inches(1.28)
y = SECHEAD(s, "Deformation Model", LM, y)
BULLETS(s, [
    "Indentor always approaches vertically (along skin-centre normal)",
    "Off-centre contacts hit the curved surface at an oblique angle",
    "→ naturally produces asymmetric deformation without extra modelling",
    "Skin points within contact patch displaced vertically by Hertz profile:",
    "  Δz(r) = δ₀ × max(0, 1 − r²/a²)  where r = radial distance from contact",
    "  contact radius a derived from Hertz: a = √(R_ind × δ₀)",
], LM, y, COL_W, Inches(1.80), sz=11)

# Right: two point cloud preview images
y_r = BTOP
y_r = SECHEAD(s, "Point Cloud Previews", RC, y_r)

pc1_w = COL_W
pc1_h = pc1_w / (2963/1527)   # ≈ 3.03"
pc2_h = pc1_w / (3503/1577)   # ≈ 2.65"

remaining = FBOT - Inches(0.38) - y_r
each_h    = min(pc1_h, (remaining - Inches(0.55)) / 2)

IMG(s, "reconstruction/assets/pointcloud_preview.png",  RC, y_r, pc1_w, each_h)
CAP(s, "Deformed skin geometry — contact visible as central depression",
    RC, y_r + each_h + Inches(0.04), pc1_w)
y_r += each_h + Inches(0.32)

IMG(s, "reconstruction/assets/pointcloud_preview2.png", RC, y_r, pc1_w, each_h)
CAP(s, "Side view — curvature of the spherical cap rest geometry clearly visible",
    RC, y_r + each_h + Inches(0.04), pc1_w)


# ══════════════════════════════════════════════════════════════════════════════
#   SLIDE 16 — PointCloudNet Architecture & Training
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HEADER(s, "Stage 3 — PointCloudNet Architecture & Training",
       "dense_contact/model.py  ·  dense_contact/train_pointcloud.py")
FOOTER(s)

y = BTOP
y = SECHEAD(s, "Architecture", LM, y)
MONO_BLOCK(s, [
    "Input  : (1, 224, 224) grayscale extracted pattern",
    "↓",
    "ResNet18  encoder  (pretrained, same as DenseContactNet)",
    "↓",
    "U-Net decoder  (4 blocks with skip connections)",
    "↓",
    "PointCloud head: (B, 32, H, W) → reshape → (B, 2048, 3)   [X,Y,Z in mm]",
    "↓",
    "Auxiliary heads: displacement (scalar)  ·  force (scalar)",
], LM, y, COL_W)
y += Inches(2.75)

y = SECHEAD(s, "Point Cloud Structure", LM, y)
BULLETS(s, [
    "2048 = 32 rows × 64 cols — fixed structured grid per frame",
    "Point k ↔ grid cell (k // 64, k % 64) — same order in every frame",
    "MSE loss works directly on ordered clouds (no nearest-neighbour matching)",
    "Each (X, Y, Z) in mm — physical units, not normalised",
    "Z = 0 at skin centre rest position; negative Z = inward deformation",
], LM, y, COL_W, Inches(1.55), sz=11)

# Right
y_r = BTOP
y_r = SECHEAD(s, "Loss Function", RC, y_r)
TABLE(s,
      ["Term", "λ", "Units"],
      [
          ["MSE(pred_cloud, target_cloud)", "1.0", "mm²"],
          ["L1(displacement)",              "0.5", "normalised [0,1]"],
          ["L1(force)",                     "0.5", "normalised [0,1]"],
      ],
      RC, y_r, COL_W, hdr_sz=10, row_sz=11, row_h=Inches(0.38))

tbl_b = y_r + Inches(0.40) + 3 * Inches(0.38)
y_r   = tbl_b + Inches(0.20)
y_r   = SECHEAD(s, "Training Configuration", RC, y_r)
TABLE(s,
      ["Parameter", "Value"],
      [
          ["Optimiser",      "Adam"],
          ["LR",             "1×10⁻⁴ (uniform)"],
          ["Batch size",     "32"],
          ["Max epochs",     "150 (interrupted at 9)"],
          ["Val holdout",    "y = 6 mm, y = 14 mm"],
          ["Data",           "26,699 train / 6,598 val"],
          ["Device",         "Apple MPS"],
      ],
      RC, y_r, COL_W, hdr_sz=10, row_sz=10, row_h=Inches(0.36))

tbl_b2 = y_r + Inches(0.40) + 7 * Inches(0.36)
# Note: no further elements on right column — table fills the available space


# ══════════════════════════════════════════════════════════════════════════════
#   SLIDE 17 — PointCloudNet Results
# ══════════════════════════════════════════════════════════════════════════════
s = S(); HEADER(s, "Stage 3 — PointCloudNet Results"); FOOTER(s)

TXT(s,
    "Training was interrupted after 9 epochs (early results). "
    "Even at this early stage, the model achieves sub-50 µm point-cloud RMSE "
    "on the validation set — demonstrating that the architecture and loss are well-matched.",
    LM, BTOP, CW, Inches(0.62), sz=12, clr=INK)

y = BTOP + Inches(0.72)

# Left
y = SECHEAD(s, "Validation Metrics (Best Checkpoint — Epoch 8)", LM, y)
TABLE(s,
      ["Metric", "Train", "Val (best)"],
      [
          ["Point cloud RMSE",  "< 0.050 mm", "0.044 mm"],
          ["Displacement MAE",  "—",          "0.275 mm"],
          ["Force MAE",         "—",          "0.005 N"],
          ["Val loss",          "—",          "0.0226"],
      ],
      LM, y, COL_W, hdr_sz=11, row_sz=12, row_h=Inches(0.38))

tbl_b = y + Inches(0.40) + 4 * Inches(0.38)
y2 = tbl_b + Inches(0.20)
y2 = SECHEAD(s, "Interpretation", LM, y2)
BULLETS(s, [
    "RMSE 0.044 mm = 44 µm — sub-pixel accuracy on the physical skin surface",
    "Displacement MAE 0.275 mm — comparable to Stage 1 (0.38 mm) at only 9 epochs",
    "Force MAE 0.005 N — significantly better than Stage 1 (0.055 N)",
    "Training interrupted early; further epochs expected to improve all metrics",
    "Point-cloud outputs can be visualised live with view_pointclouds.py",
], LM, y2, COL_W, Inches(1.65), sz=11)

# Right: both point cloud previews
y_r = BTOP + Inches(0.72)
y_r = SECHEAD(s, "Reconstructed Skin Geometry", RC, y_r)

remaining = FBOT - Inches(0.38) - y_r
each_h2   = min(COL_W / (2963/1527), (remaining - Inches(0.45)) / 2)
IMG(s, "reconstruction/assets/pointcloud_preview.png",  RC, y_r, COL_W, each_h2)
CAP(s, "Top view — contact depression at sensor centre",
    RC, y_r + each_h2 + Inches(0.04), COL_W)
y_r += each_h2 + Inches(0.32)
IMG(s, "reconstruction/assets/pointcloud_preview2.png", RC, y_r, COL_W, each_h2)
CAP(s, "Perspective view — curvature and deformation simultaneously visible",
    RC, y_r + each_h2 + Inches(0.04), COL_W)


# ══════════════════════════════════════════════════════════════════════════════
#   SLIDE 18 — Live Inference
# ══════════════════════════════════════════════════════════════════════════════
s = S()
HEADER(s, "Live Inference",
       "contact_estimation/live_predict.py  ·  dense_contact/infer_live.py")
FOOTER(s)

y = BTOP
TABLE(s,
      ["Script", "Model", "Output @ 30 fps"],
      [
          ["contact_estimation/live_predict.py",
           "ResNet18",
           "loc_x, loc_y, displacement, force overlaid on video feed"],
          ["dense_contact/infer_live.py",
           "DenseContactNet",
           "contact / depth / pressure maps rendered in real time"],
          ["dense_contact/infer_pointcloud.py",
           "PointCloudNet",
           "3D deformed skin geometry from saved frames"],
          ["dense_contact/infer_pointcloud_live.py",
           "PointCloudNet",
           "Animated 3D skin point cloud in real time"],
      ],
      LM, y, CW, hdr_sz=11, row_sz=11, row_h=Inches(0.42))

y += Inches(0.40) + 4 * Inches(0.42) + Inches(0.22)

# CE pipeline image (2685×772, ratio 3.478)
ce_w = CW
ce_h = min(ce_w / (2685/772), FBOT - Inches(0.38) - y - Inches(0.10))
IMG(s, "contact_estimation/assets/pipeline.png", LM, y, ce_w, ce_h)
CAP(s,
    "Live inference pipeline: ROI crop → CLAHE → adaptive threshold → model → overlay.",
    LM, y + ce_h + Inches(0.05), ce_w)


# ══════════════════════════════════════════════════════════════════════════════
#   SLIDE 19 — Summary & Future Work
# ══════════════════════════════════════════════════════════════════════════════
s = S(); HEADER(s, "Summary & Future Work"); FOOTER(s)

# Left
y = BTOP
y = SECHEAD(s, "Contributions", LM, y)
BULLETS(s, [
    "Low-cost vision-based tactile sensor: USB camera + patterned elastomer skin (<$50)",
    "Stage 1 — ResNet18: 4-scalar contact estimation, 0.52 mm loc. accuracy on unseen positions",
    "Stage 2 — DenseContactNet: simultaneous contact, depth, and pressure spatial maps",
    "  U-Net decoder with ResNet18 backbone, Hertzian pseudo-label synthesis",
    "Stage 3 — PointCloudNet: full 3D skin geometry at 0.044 mm RMSE (44 µm)",
    "  Ordered structured point cloud — enables direct MSE training",
    "All three models run live at 30 fps on Apple Silicon (MPS backend)",
], LM, y, COL_W, Inches(2.30), sz=11)

y += Inches(2.42)
y = SECHEAD(s, "Future Work", LM, y)
BULLETS(s, [
    "Collect a 2D grid dataset (vary both x and y contact positions)",
    "Train on real 3D deformation data (depth camera or stereo reconstruction)",
    "Extend to multi-contact and sliding contact scenarios",
    "Deploy on the Sung Robotics Group underwater robotic gripper",
    "Benchmark against commercial tactile sensor arrays (GelSight, DIGIT)",
    "Investigate self-supervised pre-training on unlabelled skin video",
], LM, y, COL_W, Inches(1.80), sz=11)

# Right: at-a-glance stats table + contact grid
y_r = BTOP
y_r = SECHEAD(s, "At a Glance", RC, y_r)
TABLE(s,
      ["Metric", "Value"],
      [
          ["Sensor cost",                "< $50"],
          ["Location MAE (val, Stage 1)","0.52 mm"],
          ["Displacement MAE (val, S1)", "0.38 mm"],
          ["Force MAE (val, Stage 1)",   "0.055 N"],
          ["Point cloud RMSE (Stage 3)", "0.044 mm  (44 µm)"],
          ["Dataset size",               "4,074 frames · 9 sessions"],
          ["Inference speed",            "30 fps — live"],
          ["Training device",            "Apple MPS (M-series Mac)"],
      ],
      RC, y_r, COL_W, hdr_sz=10, row_sz=10, row_h=Inches(0.36))


# ══════════════════════════════════════════════════════════════════════════════
#   SAVE
# ══════════════════════════════════════════════════════════════════════════════
prs.save(OUT)
n = len(prs.slides)
print(f"Saved: {OUT}  ({n} slides)")
