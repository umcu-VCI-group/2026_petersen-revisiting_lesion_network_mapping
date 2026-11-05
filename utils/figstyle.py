"""Figure conventions shared by all figures: rcParams, text homogenization, panel layout,
and cropping of surface views out of the combined slice + surface renders."""
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.text as mtext
import numpy as np
from PIL import Image


def apply_figure_style(sizes=(8, 7, 6)):
    base, secondary, tick = sizes
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": base, "axes.labelsize": base, "axes.titlesize": base,
        "legend.fontsize": secondary, "xtick.labelsize": tick, "ytick.labelsize": tick,
        "axes.linewidth": 0.6, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.titlelocation": "left",
        "lines.linewidth": 1.2, "patch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.weight": "bold", "axes.titleweight": "bold", "axes.labelweight": "bold",
    })


# --- text conventions: bold upright text, bold upright T, "label permutation-based" --------------
_MATH = re.compile(r"\$([^$]+)\$")


def _bold_math(m):
    s = m.group(1)
    if s.strip() == r"\beta":
        return "\u03b2"
    s = s.replace(r"\mathrm{", r"\mathbf{").replace(r"\mathit{", r"\mathbf{")
    return "$" + (s if s.startswith(r"\mathbf{") else r"\mathbf{" + s + "}") + "$"


def homogenize_text(s):
    s = s.replace("T-statistic", "T")
    s = re.sub(r"(?<![Ll]abel )Permutation-based", "Label permutation-based", s)
    s = re.sub(r"(?<![Ll]abel )permutation-based", "label permutation-based", s)
    if s.strip() == "Permutation":
        s = "Label permutation-based"
    if s.strip() == "One-sample":
        s = "One-sample t-test"
    return _MATH.sub(_bold_math, s)


def homogenize_fig(fig):
    """Apply the text conventions to every text element (call right before savefig)."""
    for t in fig.findobj(mtext.Text):
        if t.get_text():
            t.set_text(homogenize_text(t.get_text()))
        t.set_fontweight("bold")
        t.set_fontstyle("normal")
    return fig


# --- panel layout ---------------------------------------------------------------------------------
def spread_labels(vals, min_gap):
    """Label positions near `vals`, pushed apart to at least `min_gap` (order preserved)."""
    order = np.argsort(vals)
    pos = np.array(vals, dtype=float)[order]
    for _ in range(200):
        moved = False
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < min_gap:
                d = (min_gap - (pos[i] - pos[i - 1])) / 2
                pos[i - 1] -= d; pos[i] += d; moved = True
        if not moved:
            break
    out = np.empty_like(pos); out[order] = pos
    return out


def place_row_titles(fig, row, pad=0.012, letter_x=None, letter_size=10, title_size=7.5, title_gap=None):
    """row: list of (ax, letter, title) sharing one baseline above the highest axes of the row."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    y = max(ax.get_position().y1 for ax, _, _ in row) + pad
    for ax, letter, title in row:
        lx = None
        if letter:
            lx = letter_x[letter] if (letter_x and letter in letter_x) else \
                inv.transform(ax.get_tightbbox(r).get_points())[0, 0] - 0.005
            fig.text(lx, y, letter, ha="left", va="baseline", fontsize=letter_size, fontweight="bold")
        if title:
            tx = lx + title_gap if (title_gap is not None and lx is not None) else ax.get_position().x0
            fig.text(tx, y, title, ha="left", va="baseline", fontsize=title_size)
    return y


# --- surface views cropped from the combined slice + surface renders -----------------------------
def _nonwhite(a, thr=245):
    alpha = a[..., 3] if a.shape[-1] == 4 else np.full(a.shape[:2], 255)
    return (a[..., :3].astype(int).min(-1) < thr) & (alpha > 10)


def _runs(mask1d, min_len=20, gap=12):
    idx = np.where(mask1d)[0]
    if len(idx) == 0:
        return []
    out, s, p = [], idx[0], idx[0]
    for v in idx[1:]:
        if v - p > gap:
            if p - s >= min_len:
                out.append((s, p + 1))
            s = v
        p = v
    if p - s >= min_len:
        out.append((s, p + 1))
    return out


def render_info(path):
    """Sidecar of a combined render (brainplot.combined_figure): colour scale `vmax`, image fraction
    `surface_x0` where the surface column starts, and `rows` = domains with a drawn surface strip."""
    return json.loads(Path(f"{path}.json").read_text())


def surface_views(path):
    """{domain: four cortical views (left lat/med, right lat/med)} cropped from a combined render."""
    info = render_info(path)
    a = np.asarray(Image.open(path).convert("RGBA")); nw = _nonwhite(a)
    x0 = int(a.shape[1] * info["surface_x0"]); region = nw[:, x0:]
    rows = []
    for (y0, y1) in _runs(region.sum(1) > 15, min_len=150, gap=25):
        boxes = [(x0 + c0, x0 + c1) for (c0, c1) in _runs(region[y0:y1].sum(0) > 3, min_len=150, gap=20)]
        xa, xb = min(b[0] for b in boxes), max(b[1] for b in boxes); w = (xb - xa) / 4
        views = []
        for q in range(4):
            sub = a[y0:y1, int(xa + q * w):int(xa + (q + 1) * w)]
            ys, xs = np.where(_nonwhite(sub))
            views.append(sub[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
        rows.append(views)
    assert len(rows) == len(info["rows"]), f"{path}: {len(rows)} surface rows found, {len(info['rows'])} expected"
    return dict(zip(info["rows"], rows))


def pack_rows(path, gaps, pad=20):
    """Remove the empty vertical space between content bands of a render (Figure 3a)."""
    g = np.asarray(Image.open(path).convert("RGB")); W = g.shape[1]
    bands = _runs((g.min(-1) < 235).sum(1) > 2, min_len=5, gap=60)
    assert len(bands) == len(gaps) + 1, f"{len(bands)} content bands, {len(gaps) + 1} expected"
    parts = []
    for i, (a, b) in enumerate(bands):
        parts.append(g[a:b])
        if i < len(gaps):
            parts.append(np.full((gaps[i], W, 3), 255, np.uint8))
    white = np.full((pad, W, 3), 255, np.uint8)
    return np.vstack([white] + parts + [white])
