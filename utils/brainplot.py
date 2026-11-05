"""Brain renders: axial slices (nilearn) and cortical surfaces (yabplot, Conte69 midthickness).

All maps show the signed voxelwise T, masked by significance where a p-map is given, on a shared
symmetric colour scale (97.5th percentile of in-brain |T| across domains, or the full range).
Radiological orientation (R on the viewer's left)."""
import json
import os
from textwrap import fill

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import yabplot as yab
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from nilearn import plotting

from config import DOMAINS, F_COPE, F_TSTAT, FIG_DIR, KEYS, THR, has_results, load_map, load_mask, result_path
from figstyle import homogenize_fig

LH, RH = yab.data.get_surface_paths("midthickness", "bmesh")
VIEWS = ["left_lateral", "left_medial", "right_lateral", "right_medial"]
CUTS = [-26, -10, 6, 22]
BRAIN, AFFINE, BG = load_mask()


def proj(family, fname, key, interp="linear"):
    """Volume -> surface; p-maps with interp='nearest' to keep the threshold boundary exact."""
    return yab.project_vol2surf(str(result_path(family, key, fname)), mask_medial_wall=True, interpolation=interp)


def autocrop(img):
    m = img[..., :3].mean(-1) < 250
    if not m.any():
        return img
    ys, xs = np.where(m)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def surf_img(lh, rh, vmax, layout=(1, 4)):
    lm, rm = yab.load_vertexwise_mesh(LH, RH, lh, rh)
    pl = yab.plot_vertexwise(lm, rm, views=VIEWS, layout=layout, cmap="RdBu_r", vminmax=[-vmax, vmax],
                             nan_color=(0.83, 0.83, 0.83), display_type="object", zoom=1.5)
    for nm in list(getattr(pl, "scalar_bars", {}).keys()):
        pl.remove_scalar_bar(nm)
    img = pl.screenshot(transparent_background=False, return_img=True, scale=3)
    pl.close()
    return autocrop(img)


def _vol(src, k, fname=F_TSTAT):
    """src: PALM family name, or {domain: NIfTI path} (e.g. mean aggregation maps)."""
    if isinstance(src, dict):
        return np.where(BRAIN, nib.load(str(src[k])).get_fdata(), 0.0)
    return load_map(src, k, fname, brain=BRAIN)


def tscale(family, keys, fname=F_TSTAT, mode="robust"):
    allv = np.abs(np.concatenate([_vol(family, k, fname)[BRAIN] for k in keys]))
    if isinstance(family, dict):
        allv = allv[allv != 0]
    return float(np.ceil(np.nanmax(allv))) if mode == "full" else float(np.round(np.nanpercentile(allv, 97.5)))


def slices(ax, data, vmax, cuts=CUTS):
    img = nib.Nifti1Image(data.astype(np.float32), AFFINE)
    return plotting.plot_stat_map(img, bg_img=BG, display_mode="z", cut_coords=cuts, axes=ax, threshold=1e-6,
                                  vmax=vmax, symmetric_cbar=True, cmap="RdBu_r", colorbar=False,
                                  annotate=False, draw_cross=False, black_bg=False, radiological=True)


def mark_rl(display):
    ax = list(display.axes.values())[0].ax
    ax.text(0.03, 0.97, "R", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.text(0.97, 0.97, "L", transform=ax.transAxes, ha="right", va="top", fontsize=9)


def ycenter(display, ax):
    if display is None:
        p = ax.get_position(); return (p.y0 + p.y1) / 2
    ps = [c.ax.get_position() for c in display.axes.values()]
    return (min(p.y0 for p in ps) + max(p.y1 for p in ps)) / 2


def colorbar(fig, x0, x1, vmax, label, y=0.035):
    w = 0.26 * (x1 - x0); c = (x0 + x1) / 2
    cax = fig.add_axes([c - w / 2, y, w, 0.011])
    sm = ScalarMappable(cmap="RdBu_r", norm=Normalize(-vmax, vmax)); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both", ticks=[-vmax, 0, vmax])
    cb.set_label(label, fontsize=11); cb.ax.tick_params(labelsize=9)


def save(fig, outname, info=None, surface_ax=None):
    """Save with a tight bounding box. For combined renders, `info` is written to <outname>.json
    together with the image fraction at which the surface column (surface_ax) starts."""
    homogenize_fig(fig)
    if info is not None:
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())                 # inches, as used by savefig
        pad = plt.rcParams["savefig.pad_inches"]
        x0 = surface_ax.get_position().x0 * fig.get_figwidth()
        info["surface_x0"] = (x0 - (bb.x0 - pad)) / (bb.width + 2 * pad)
        (FIG_DIR / f"{outname}.json").write_text(json.dumps(info))
    fig.savefig(FIG_DIR / outname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG_DIR / outname)


def combined_figure(family, pmap, label, outname, vmax="robust", keys=KEYS, cbar_label=None):
    """Axial slices + cortical surface strip per domain (Figure 1 layout). pmap=None: unthresholded;
    family may also be a {domain: NIfTI path} dict (unthresholded)."""
    keys = [k for k in keys if isinstance(family, dict) or has_results(family, k)]
    V = tscale(family, keys, mode=vmax)
    n = len(keys)
    fig = plt.figure(figsize=(16.5, 2.3 * n))
    gs = fig.add_gridspec(n, 3, width_ratios=[0.18, 1.0, 1.7], wspace=0.02, hspace=0.06,
                          left=0.01, right=0.99, top=0.93, bottom=0.085)
    ax_sl = [fig.add_subplot(gs[i, 1]) for i in range(n)]
    ax_sf = [fig.add_subplot(gs[i, 2]) for i in range(n)]
    fig.canvas.draw()
    disp, has_surf = [], []
    for i, k in enumerate(keys):
        T = _vol(family, k)
        data = np.where(load_map(family, k, pmap, brain=BRAIN) >= THR, T, 0.0) if pmap else T
        disp.append(slices(ax_sl[i], data, V) if np.any(data != 0) else None)
        if disp[-1] is None:
            ax_sl[i].axis("off")
        tl, tr = (yab.project_vol2surf(str(family[k]), mask_medial_wall=True, interpolation="linear")
                  if isinstance(family, dict) else proj(family, F_TSTAT, k))
        if pmap:
            pl_, pr = proj(family, pmap, k, "nearest")
            tl, tr = np.where(pl_ >= THR, tl, np.nan), np.where(pr >= THR, tr, np.nan)
        has_surf.append(bool(np.isfinite(np.concatenate([tl, tr])).any()))
        ax_sf[i].axis("off")
        if has_surf[-1]:
            ax_sf[i].imshow(surf_img(tl, tr, V), aspect="equal")
    if disp[0] is not None:
        mark_rl(disp[0])
    drawn = [d for d in disp if d is not None]
    ytop = max(c.ax.get_position().y1 for d in drawn for c in d.axes.values()) + 0.010
    for col, head in ((1, "Axial slices"), (2, "Cortical surface")):
        p = gs[0, col].get_position(fig)
        fig.text((p.x0 + p.x1) / 2, ytop, head, fontsize=14, ha="center", va="bottom")
    p1, p2 = gs[0, 1].get_position(fig), gs[0, 2].get_position(fig)
    for i, k in enumerate(keys):
        yc = ycenter(disp[i], ax_sl[i])
        fig.text(p1.x0 - 0.006, yc, fill(DOMAINS[k], 16), va="center", ha="right", fontsize=12)
        if disp[i] is None and not has_surf[i]:
            fig.text((p1.x0 + p2.x1) / 2, yc, "n.s.", va="center", ha="center", fontsize=12)
        elif disp[i] is None:
            fig.text((p1.x0 + p1.x1) / 2, yc, "n.s.", va="center", ha="center", fontsize=12)
        elif not has_surf[i]:
            fig.text((p2.x0 + p2.x1) / 2, yc, "n.s.", va="center", ha="center", fontsize=12)
    colorbar(fig, p1.x0, p2.x1, V, cbar_label or (rf"$T$ ({label}, $p_{{FDR}}$ < 0.05)" if pmap else r"$T$"))
    # sidecar for Figures 4 and 5: colour scale, surface-column position, domains with a surface strip
    save(fig, outname, info={"vmax": V, "rows": [k for k, s in zip(keys, has_surf) if s]}, surface_ax=ax_sf[0])


def slice_contrast_figure(family, pmaps, labels, outname, keys=KEYS):
    """Axial slices of two significance maps side by side (panel a | panel b)."""
    keys = [k for k in keys if has_results(family, k)]
    V = tscale(family, keys)
    n = len(keys)
    fig = plt.figure(figsize=(13.5, 2.15 * n))
    gs = fig.add_gridspec(n, 4, width_ratios=[0.18, 1, 0.16, 1], wspace=0.02, hspace=0.05,
                          left=0.02, right=0.99, top=0.95, bottom=0.075)
    cols = [[fig.add_subplot(gs[i, c]) for i in range(n)] for c in (1, 3)]
    fig.canvas.draw()
    disp = [[], []]
    for c, pm in enumerate(pmaps):
        for i, k in enumerate(keys):
            T = load_map(family, k, F_TSTAT, brain=BRAIN)
            data = np.where(load_map(family, k, pm, brain=BRAIN) >= THR, T, 0.0)
            disp[c].append(slices(cols[c][i], data, V) if np.any(data != 0) else None)
            if disp[c][-1] is None:
                cols[c][i].axis("off")
    mark_rl(next(d for d in (disp[0][0], disp[1][0]) if d is not None))
    drawn = [d for col in disp for d in col if d is not None]
    ytop = max(c.ax.get_position().y1 for d in drawn for c in d.axes.values()) + 0.008
    for c, (gc, lab) in enumerate(zip((1, 3), labels)):
        p = gs[0, gc].get_position(fig)
        fig.text(p.x0, ytop, f"{'ab'[c]}   {lab}", fontsize=14, ha="left", va="bottom")
        for i in range(n):
            if disp[c][i] is None:
                fig.text((p.x0 + p.x1) / 2, ycenter(None, cols[c][i]), "n.s.", ha="center", va="center", fontsize=12)
    for i, k in enumerate(keys):
        ref = disp[0][i] or disp[1][i]
        fig.text(gs[0, 1].get_position(fig).x0 - 0.008, ycenter(ref, cols[0][i]), fill(DOMAINS[k], 16),
                 va="center", ha="right", fontsize=12)
    colorbar(fig, gs[0, 1].get_position(fig).x0, gs[0, 3].get_position(fig).x1, V, r"$T$ ($p_{FDR}$ < 0.05)", y=0.030)
    save(fig, outname)


def T_beta_figure(family, outname, surface=False, keys=KEYS):
    """Unthresholded T (a) and beta (b) per domain, on axial slices or cortical surfaces (2x2 views)."""
    keys = [k for k in keys if has_results(family, k)]
    n = len(keys)
    if surface:
        tex = {f: {k: proj(family, f, k) for k in keys} for f in (F_TSTAT, F_COPE)}
        vmax = {f: float(np.round(np.nanpercentile(np.abs(np.concatenate([np.concatenate(tex[f][k]) for k in keys])), 97.5), 2))
                for f in tex}
        fig = plt.figure(figsize=(8.5, 3.0 * n))
        gs = fig.add_gridspec(n, 3, width_ratios=[0.26, 1, 1], wspace=0.0, hspace=0.22,
                              left=0.02, right=0.99, top=0.91, bottom=0.11)
    else:
        vmax = {f: float(np.round(np.nanpercentile(np.abs(np.concatenate(
            [load_map(family, k, f, brain=BRAIN)[BRAIN] for k in keys])), 97.5), 2)) for f in (F_TSTAT, F_COPE)}
        fig = plt.figure(figsize=(16.5, 2.3 * n))
        gs = fig.add_gridspec(n, 3, width_ratios=[0.18, 1.0, 1.0], wspace=0.02, hspace=0.06,
                              left=0.01, right=0.99, top=0.93, bottom=0.085)
    fig.canvas.draw()
    first = None
    for i, k in enumerate(keys):
        for c, f in ((1, F_TSTAT), (2, F_COPE)):
            ax = fig.add_subplot(gs[i, c])
            if surface:
                ax.axis("off"); ax.imshow(surf_img(*tex[f][k], vmax[f], layout=(2, 2)), aspect="equal")
            else:
                d = slices(ax, load_map(family, k, f, brain=BRAIN), vmax[f])
                first = first or d
        p = gs[i, 1].get_position(fig)
        fig.text(p.x0 - 0.006, (p.y0 + p.y1) / 2, fill(DOMAINS[k], 16), va="center", ha="right", fontsize=12)
    if first is not None:
        mark_rl(first)
    for c, (f, lab) in enumerate(((F_TSTAT, r"$T$"), (F_COPE, r"$\beta$")), start=1):
        p = gs[0, c].get_position(fig)
        fig.text(p.x0, p.y1 + 0.012, "ab"[c - 1], fontsize=16, ha="left", va="bottom")
        w = 0.6 * (p.x1 - p.x0)
        cax = fig.add_axes([(p.x0 + p.x1) / 2 - w / 2, 0.04, w, 0.010])
        sm = ScalarMappable(cmap="RdBu_r", norm=Normalize(-vmax[f], vmax[f])); sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both", ticks=[-vmax[f], 0, vmax[f]])
        cb.set_label(lab, fontsize=11); cb.ax.tick_params(labelsize=9)
    save(fig, outname)
