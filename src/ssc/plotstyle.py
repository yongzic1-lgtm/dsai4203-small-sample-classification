"""Shared figure style for every report figure (core report and extensions E1-E4): Times New Roman, print size.

    from ssc import plotstyle as ps
    plt = ps.setup()                                  # Agg backend + rcParams (Times New Roman, 8-9 pt, white)
    fig, ax = plt.subplots(figsize=(ps.FULL_W, 2.3))  # draw at the PRINTED size: 8 pt in the PNG = 8 pt on paper
    ax.plot(x, y, color=ps.color("S003"))             # colour follows the entity in every figure
    ps.clean(ax, grid="y")                            # no top/right spines, hairline grid on one axis
    ps.panel_title(ax, "(a) Something")               # bold "(a)" + 9 pt title, left-aligned
    ps.nines(ax, "x")                                 # AUC axis x -> -log10(1 - x): 0.8, 0.9, 0.99, 0.999, 0.9999
    ps.refline(ax, 0.947, "x")                        # reference value: 0.6 pt dashed secondary ink
    ps.callout(ax, "final model", xy=(x0, y0), xytext=(x1, y1))   # ink text + thin leader line
    ps.save(fig, path)                                # 300 dpi PNG, tight bbox, small margin, white

API (kept small on purpose): setup, clean, panel_title, label, callout, hairline, refline, nines, nines_break,
break_at, clip_auc, colorbar, diverging_cmap, sequential_cmap, color, name, shade, save; constants below.

Rules (dataviz method; light mode / print):
- categorical hues in a fixed order, never cycled; the same entity has the same colour in every figure (ENTITY):
  S003 / transductive GMM blue, S005 / strict QDA orange, S006 / self-training aqua, Bayesian QDA yellow,
  foundation models violet, leaky or label-reading controls red, every other model or reference in grey ink;
  at most 3 hues in one scatter; more only in dot/line plots with direct labels;
- sequential = one hue light -> dark (BLUE_RAMP); diverging = blue <-> red around the neutral grey MID, always
  with a small colour bar (colorbar());
- text is always in ink (INK, INK2, MUTED), never in a series colour;
- thin marks (lines 1.0-1.4 pt), small markers with a 0.5 pt white edge, transparency for dense data;
- solid hairline grid (GRID), axis lines AXIS, no top/right spines, no dual y-axes, no 3-D;
- every reference VALUE (0, 0.5, a threshold, a ceiling) is a 0.6 pt dashed INK2 line (refline), labelled in INK2;
  solid hairlines (hairline) are for structure only (dividers, baselines);
- text >= 7.5 pt (8 pt for row labels and annotations); one name per entity in every figure (LABEL / name()).
FULL_W = 6.27 in is the text width of reports/report.pdf (A4, 2.54 cm margins = 15.92 cm).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------------------- palette
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
CATEGORICAL = (BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED)   # fixed order, never cycled

# ink
INK = "#0b0b0b"        # primary text, key reference lines
INK2 = "#52514e"       # secondary text, leader lines, "other" models / baselines
MUTED = "#898781"      # quiet annotations, tick marks
GRID = "#e1e0d9"       # gridline hairline (solid)
AXIS = "#c3c2b7"       # axis spines, reference hairlines
LIGHT = "#d6d5ce"      # very light fill (background points)
MID = "#f0efec"        # neutral midpoint of diverging scales, light bands
GREY = INK2            # grey ink for baselines and honest controls

# colour follows the entity in every figure (keys are case-insensitive; unknown keys -> GREY)
ENTITY = {
    "s003": BLUE, "gmm": BLUE, "transductive": BLUE,
    "s005": ORANGE, "strict_qda": ORANGE, "qda": ORANGE,
    "s006": AQUA, "selftrain": AQUA, "self-training": AQUA,
    "bqda": YELLOW, "bayes_qda": YELLOW, "bayesian_qda": YELLOW,
    "foundation": VIOLET, "tabpfn": VIOLET, "tabicl": VIOLET,
    "leaky": RED, "label_reading": RED, "control_leaky": RED,
    "s001": GREY, "s002": GREY, "s004": GREY, "other": GREY, "baseline": GREY, "honest": GREY,
    "reference": INK, "bayes": INK, "public": INK,
}

# one name per entity in every figure (core and extensions); keys are case-insensitive
LABEL = {
    "s001": "S001 L1 logistic, all 150", "s002": "S002 QDA, reg 0.1", "s003": "S003 GMM",
    "s004": "S004 E3 top-20 + L2", "s005": "S005 strict QDA", "s006": "S006 self-training",
    "bqda": "Bayesian QDA, label-free block", "bqda_infold": "Bayesian QDA, in-fold block",
    "r210": "Log-loss QDA (r210)", "r155": "GMM, tight prior (r155)", "ext_qda": "Ext.-grid QDA (r930, post hoc)",
    "tabpfn": "TabPFN v2", "tabicl": "TabICL v2",
}

# sequential (one hue, light -> dark) and diverging (blue <-> neutral grey <-> red)
BLUE_RAMP = ("#cde2fb", "#9cc5f4", "#6aa6ea", "#2a78d6", "#1d5fb0", "#154a8c", "#0d366b")
DIVERGING = ("#1d5fb0", "#6aa6ea", "#b9d3f2", MID, "#f2b9b6", "#ea7f7a", "#c2302d")

FULL_W = 6.27          # in: text width of reports/report.pdf
DPI = 300

# nines axis: an AUC x is drawn at -log10(1 - x); exact 1.0 cannot sit on it, so clip_auc caps it at CAP
NINES_TICKS = (0.8, 0.9, 0.99, 0.999, 0.9999)
CAP = 1 - 1e-5         # where an AUC of exactly 1.0 is drawn (labelled "1.0" by nines(..., cap=True))

_FONT_FILES = ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf")
_FONT_DONE = False


def color(key: str) -> str:
    """Colour of an entity (sub id, model family or role); grey ink for anything not in ENTITY."""
    return ENTITY.get(str(key).lower(), GREY)


def name(key: str) -> str:
    """The one display name of an entity (LABEL); the key itself if it has none."""
    return LABEL.get(str(key).lower(), str(key))


def shade(c: str, f: float) -> tuple:
    """Colour `c` mixed with black (f > 0, darker) or with white (f < 0, lighter) by |f| (0..1); RGB tuple."""
    from matplotlib.colors import to_rgb
    a = np.array(to_rgb(c))
    return tuple(a * (1 - f)) if f >= 0 else tuple(a + (1 - a) * (-f))


def _register_fonts() -> None:
    global _FONT_DONE
    if _FONT_DONE:
        return
    from matplotlib import font_manager as fm
    for name in _FONT_FILES:
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            try:
                fm.fontManager.addfont(str(p))
            except Exception:
                pass
    _FONT_DONE = True


def setup():
    """Agg backend + shared rcParams (Times New Roman, also for maths text such as $v_1$); returns
    matplotlib.pyplot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _register_fonts()
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold", "mathtext.fallback": "stix", "axes.unicode_minus": True,
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "legend.title_fontsize": 8,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.size": 0, "ytick.minor.size": 0, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.titlepad": 4, "axes.titlelocation": "left", "axes.titlecolor": INK,
        "axes.labelpad": 2.5,
        "lines.linewidth": 1.2, "lines.markersize": 4, "lines.markeredgewidth": 0.5, "patch.linewidth": 0,
        "legend.frameon": False, "legend.handletextpad": 0.4, "legend.columnspacing": 1.1,
        "legend.borderaxespad": 0.2, "legend.handlelength": 1.3, "legend.borderpad": 0.2,
        "figure.dpi": 100, "savefig.dpi": DPI, "savefig.facecolor": "white", "figure.facecolor": "white",
        "axes.facecolor": "white", "image.interpolation": "nearest",
    })
    return plt


def clean(ax, grid: str = "y", left: bool = True, bottom: bool = True) -> None:
    """No top/right spines; solid hairline grid on `grid` ('x', 'y', 'both' or ''); optional left/bottom spine."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if not left:
        ax.tick_params(axis="y", length=0)
    if not bottom:
        ax.tick_params(axis="x", length=0)
    ax.grid(False)
    if grid in ("x", "both"):
        ax.grid(axis="x", visible=True, color=GRID, lw=0.5, ls="-")
    if grid in ("y", "both"):
        ax.grid(axis="y", visible=True, color=GRID, lw=0.5, ls="-")
    ax.set_axisbelow(True)


_LETTER = re.compile(r"^\s*(\([a-z]\))\s*(.*)$", re.S)


def panel_title(ax, text: str, letter: str | None = None, pad: float = 4, x: float = 0.0, sub: str | None = None):
    """Panel title, 9 pt, left-aligned at axes fraction `x`: a bold "(a)" then the title in regular weight.
    The letter may be given as `letter="a"` or as a leading "(a) " in `text`. `sub` adds an 8 pt INK2 line under
    the title (the title moves up by one line: leave about 0.15 in above the axes)."""
    m = _LETTER.match(text)
    if letter is None and m:
        letter, text = m.group(1), m.group(2)
    elif letter is not None and not letter.startswith("("):
        letter = f"({letter})"
    from matplotlib.transforms import offset_copy
    if sub:
        trs = offset_copy(ax.transAxes, fig=ax.figure, x=0, y=pad, units="points")
        ax.text(x, 1.0, sub, transform=trs, ha="left", va="baseline", fontsize=8, color=INK2, clip_on=False)
        pad = pad + 10.5
    tr = offset_copy(ax.transAxes, fig=ax.figure, x=0, y=pad, units="points")
    if not letter:
        return ax.text(x, 1.0, text, transform=tr, ha="left", va="baseline", fontsize=9, color=INK, clip_on=False)
    t1 = ax.text(x, 1.0, letter, transform=tr, ha="left", va="baseline", fontsize=9, fontweight="bold",
                 color=INK, clip_on=False)
    t2 = ax.annotate(" " + text if text else "", xy=(1.0, 0.0), xycoords=t1, ha="left", va="bottom",
                     fontsize=9, color=INK, annotation_clip=False)
    return t1, t2


def label(ax, x, y, text, color: str = INK2, size: float = 7.8, **kw):
    """Direct label in ink (never in the series colour)."""
    kw.setdefault("va", "center")
    return ax.text(x, y, text, color=color, fontsize=size, **kw)


def callout(ax, text, xy, xytext, textcoords: str = "data", size: float = 7.8, color: str = INK2, **kw):
    """Annotation with a thin straight leader line in secondary ink from the text to the point `xy`."""
    kw.setdefault("ha", "left")
    kw.setdefault("va", "center")
    return ax.annotate(text, xy=xy, xytext=xytext, textcoords=textcoords, fontsize=size, color=color,
                       arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=1.5, shrinkB=2.0), **kw)


def hairline(ax, value: float, axis: str = "x", color: str = AXIS, lw: float = 0.7, **kw):
    """Structural hairline (solid): divider or baseline at x=value (axis="x") or y=value (axis="y"); for reference
    VALUES use refline()."""
    return (ax.axvline if axis == "x" else ax.axhline)(value, color=color, lw=lw, zorder=kw.pop("zorder", 1), **kw)


def refline(ax, value: float, axis: str = "x", color: str = INK2, lw: float = 0.6, **kw):
    """Reference VALUE line (a threshold, 0, 0.5, a ceiling): 0.6 pt dashed secondary ink; label it with label()."""
    kw.setdefault("zorder", 1.5)
    return (ax.axvline if axis == "x" else ax.axhline)(value, color=color, lw=lw, ls=(0, (4.0, 2.5)),
                                                       dash_capstyle="butt", **kw)


# ----------------------------------------------------------------------------------------- nines axis

def nines_fwd(x):
    """AUC -> -log10(1 - AUC). Finite for x = 1 (1 - x floored at 1e-8), but plot data through clip_auc() so
    that exact 1.0 sits at CAP; the floor only lets an axis limit reach a little beyond CAP."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log10(np.clip(1.0 - x, 1e-8, None))


def nines_inv(t):
    return 1.0 - 10.0 ** (-np.asarray(t, dtype=float))


def clip_auc(v, cap: float = CAP):
    """AUC values for a nines axis: exact 1.0 (or anything above `cap`) becomes `cap`; document this in the caption."""
    return np.minimum(np.asarray(v, dtype=float), cap)


def nines(ax, axis: str = "x", lo: float = 0.8, hi: float | None = None, ticks=NINES_TICKS,
          cap: bool = False, cap_label: str = "1.0", minor: bool = True, brk: bool = False) -> None:
    """Put an AUC axis on the nines scale x -> -log10(1 - x) with tick labels 0.8, 0.9, 0.99, ...
    lo/hi set the limits (hi defaults to a little beyond CAP if cap, else the last tick). With cap=True a tick
    at CAP is labelled `cap_label` (the core figures pass "= 1"): plot exact 1.0 values through clip_auc() and say
    so in the caption. brk=True (with cap) also draws an axis-break mark on the spine between the last tick and
    the cap (nines_break, at break_at()), so the cap does not read as the next decade; minor ticks stop below it.
    Minor ticks (2..9 within each decade of 1 - x, e.g. 0.95, 0.995) are drawn as short unlabeled marks."""
    from matplotlib.ticker import FixedLocator, FixedFormatter
    if axis == "x":
        ax.set_xscale("function", functions=(nines_fwd, nines_inv))
        lim, a = ax.set_xlim, ax.xaxis
    else:
        ax.set_yscale("function", functions=(nines_fwd, nines_inv))
        lim, a = ax.set_ylim, ax.yaxis
    hi = hi if hi is not None else (1 - 10 ** (np.log10(1 - CAP) - 0.12) if cap else ticks[-1])
    tk = [t for t in ticks if lo - 1e-12 <= t <= hi + 1e-12]
    labels = [f"{t:g}" for t in tk]
    t_brk = break_at(tk) if (cap and brk) else None
    if cap:
        tk.append(CAP)
        labels.append(cap_label)
    a.set_major_locator(FixedLocator(tk))
    a.set_major_formatter(FixedFormatter(labels))
    if minor:
        top = min(hi, CAP) if t_brk is None else float(nines_inv(t_brk - 0.08))
        mn = []
        for k in range(1, 7):
            for m in range(2, 10):
                v = 1 - m * 10.0 ** (-k)
                if lo < v < top and not any(abs(v - t) < 1e-12 for t in tk):
                    mn.append(v)
        a.set_minor_locator(FixedLocator(mn))
        ax.tick_params(axis=axis, which="minor", length=1.5, width=0.4, color=AXIS)
    lim(lo, hi)
    if t_brk is not None:
        nines_break(ax, axis, t_brk)


def break_at(ticks=NINES_TICKS) -> float:
    """Where nines(..., brk=True) puts the break, in -log10(1 - AUC) units: 55% of the way from the last tick
    below CAP to CAP (4.55 for the default ticks, i.e. AUC 0.999972). Values plotted between that and CAP would
    sit beyond the break: the calling script should check that there are none (other than exact 1.0)."""
    t_last = max(float(nines_fwd(t)) for t in ticks if t < CAP)
    return t_last + 0.55 * (float(nines_fwd(CAP)) - t_last)


def nines_break(ax, axis: str = "y", t: float | None = None, size: float = 5.0) -> None:
    """Axis-break mark (white gap + two short slanted strokes in INK2) on the left spine (axis='y') or the bottom
    spine (axis='x') at -log10(1 - AUC) = t; drawn in points, so it looks the same at every axis size."""
    from matplotlib.path import Path as MPath
    from matplotlib.transforms import blended_transform_factory as btf
    v = float(nines_inv(break_at() if t is None else t))
    if axis == "y":
        tr, xy, gap = btf(ax.transAxes, ax.transData), ([0.0], [v]), "|"
        verts = [(-1, -1.0), (1, -0.2), (-1, 0.2), (1, 1.0)]
    else:
        tr, xy, gap = btf(ax.transData, ax.transAxes), ([v], [0.0]), "_"
        verts = [(-1.0, -1), (-0.2, 1), (0.2, -1), (1.0, 1)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.MOVETO, MPath.LINETO]
    ax.plot(*xy, marker=gap, ms=size * 0.62, mew=2.6, color="white", ls="none", transform=tr, clip_on=False,
            zorder=20)
    ax.plot(*xy, marker=MPath(verts, codes), ms=size, mew=0.7, mec=INK2, mfc="none", ls="none", transform=tr,
            clip_on=False, zorder=21)


# ----------------------------------------------------------------------------------------- colour scales

def diverging_cmap(name: str = "ssc_div"):
    """Blue <-> neutral grey (#f0efec) <-> red; centre it with vmin = -vmax (or 0 / 1 around 0.5)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, DIVERGING, N=256)


def sequential_cmap(name: str = "ssc_seq", white_start: bool = False):
    """One hue light -> dark (BLUE_RAMP); white_start=True prepends white for densities."""
    from matplotlib.colors import LinearSegmentedColormap
    cols = (("#ffffff",) if white_start else ()) + BLUE_RAMP
    return LinearSegmentedColormap.from_list(name, cols, N=256)


def colorbar(ax, mappable, label: str = "", ticks=None, ticklabels=None, loc=(1.02, 0.0, 0.03, 0.6),
             orientation: str = "vertical"):
    """Small colour bar in an inset of `ax` (loc = [x0, y0, w, h] in axes fraction); no outline, ink labels."""
    cax = ax.inset_axes(list(loc))
    cb = ax.figure.colorbar(mappable, cax=cax, orientation=orientation)
    cb.outline.set_visible(False)
    cax.tick_params(length=1.5, width=0.4, color=AXIS, labelsize=7.5, labelcolor=INK2, pad=1.5)
    if ticks is not None:
        cb.set_ticks(list(ticks))
        if ticklabels is not None:
            cb.set_ticklabels(list(ticklabels))
    if label:
        cb.set_label(label, fontsize=8, color=INK2, labelpad=2)
    return cb


def save(fig, path, pad: float = 0.03) -> None:
    """300 dpi PNG, tight bounding box, small margin, white background."""
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=pad, facecolor="white")
