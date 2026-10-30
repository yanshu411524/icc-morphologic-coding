#!/usr/bin/env python3
"""Shared figure style and export routine.

One place to set the typeface so every panel in the paper matches.

FIGURE_FONT
    "times"  Times New Roman, matching the manuscript body text
    "arial"  Arial, for lettering inside figures; panel letters are Arial bold

Export formats
    pdf   vector; the first format the journal lists as preferred and the one
          to upload, since transparency survives and it stays editable
    svg   vector; opens as editable shapes in Illustrator, Inkscape, Affinity,
          and is what the PowerPoint files embed
    png   raster preview, also what the Word documents embed
    tiff  raster fallback at 600 dpi with LZW, only needed if an editor insists

EPS is deliberately not produced: the PostScript backend cannot express alpha,
so the shaded bands used throughout would come out opaque.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURE_FONT = os.environ.get("FIGURE_FONT", "times").lower()

_FAMILIES = {
    "times": ("serif", {"font.serif": ["Times New Roman", "DejaVu Serif"]}),
    "arial": ("sans-serif", {"font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]}),
}

TIFF_DPI = 600
PNG_DPI = 300


def apply(**overrides):
    """Install the shared rcParams; returns the resolved font family name."""
    if FIGURE_FONT not in _FAMILIES:
        raise SystemExit(f"FIGURE_FONT must be one of {sorted(_FAMILIES)}")
    family, extra = _FAMILIES[FIGURE_FONT]
    params = {
        "font.family": family,
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 130,
        "pdf.fonttype": 42,   # embed as TrueType so text stays selectable
        "svg.fonttype": "none",
    }
    params.update(extra)
    params.update(overrides)
    plt.rcParams.update(params)
    return family


def panel_labels(axes, *, x=-0.08, y=1.06):
    """Add panel letters. Axes content is left untouched."""
    import numpy as np

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    wrap = os.environ.get("PANEL_PARENS", "").strip() in {"1", "true", "yes"}
    flat = list(np.asarray(axes, dtype=object).ravel())
    for letter, ax in zip(letters, flat):
        if ax is None:
            continue
        label = f"({letter})" if wrap else letter
        ax.text(x, y, label, transform=ax.transAxes, fontsize=14,
                fontweight="bold", fontname="Arial", va="bottom", ha="right",
                clip_on=False)


def save(fig, name, outdir, tight=True):
    """Write one figure in every delivery format."""
    outdir.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    for ext in ("pdf", "svg", "png", "tiff"):
        kw = {}
        if ext == "tiff":
            kw = {"dpi": TIFF_DPI, "pil_kwargs": {"compression": "tiff_lzw"}}
        elif ext == "png":
            kw = {"dpi": PNG_DPI}
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    return name

