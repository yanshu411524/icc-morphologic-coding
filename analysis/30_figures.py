#!/usr/bin/env python3
"""Display figures.

Reads tables produced by analysis/26 and analysis/27. Joinpoint lines are
continuous segmented log-linear fits at the joinpoint years already selected
in analysis/26, weighted by case counts.

Outputs (figures/paper2/): pdf, svg, png (300 dpi), tiff (600 dpi LZW)
  Fig1_asr_joinpoint
  Fig2_stratified_rise
  Fig3_seer_validation
  FigS1_strat_eapc
  FigS2_seer_icc_mix
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CI5 = ROOT / "data/processed/ci5"
TAB = ROOT / "tables"
OUT = ROOT / "figures" / os.environ.get("FIG_SUBDIR", "paper2") / figstyle.FIGURE_FONT

OBS = "#3B6EA5"
COR = "#C23B3B"
OBS_FILL = "#D6E3F0"
COR_FILL = "#F4D6D6"
GRAY = "#8A8A8A"
SHADE = "#EFEFEF"

Y0, Y1 = 1998, 2017


def csv_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x):
    return None if x in ("", None) else float(x)


def segmented_fit(years, rates, cases, joinpoints):
    """Continuous log-linear segments; weights = sqrt(cases)."""
    t = np.asarray(years, float)
    y = np.log(np.asarray(rates, float))
    w = np.sqrt(np.clip(np.asarray(cases, float), 1.0, None))
    cols = [np.ones(len(t)), t]
    for jp in joinpoints:
        cols.append(np.maximum(t - jp, 0.0))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    return np.exp(X @ beta)


def panel_letters(axes, x=-0.10, y=1.04):
    figstyle.panel_labels(axes, x=x, y=y)


def fig1(pooled, jp_rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55), sharey=False)
    jp_lookup = {}
    for r in jp_rows:
        if r["stratum"] != "All 73 registries":
            continue
        key = (r["sex"], r["version"])
        years = r["joinpoint_years"]
        jps = [] if not years else [int(x) for x in years.split(";") if x]
        jp_lookup.setdefault(key, {"jps": jps, "segs": []})
        jp_lookup[key]["segs"].append(r)

    for ax, sex, title in (
        (axes[0], "Male", "Men"),
        (axes[1], "Female", "Women"),
    ):
        rows = [r for r in pooled if r["sex_label"] == sex]
        years = np.array([int(r["year"]) for r in rows])
        obs = np.array([float(r["icc_asr_obs"]) for r in rows])
        corr = np.array([float(r["icc_asr_corr"]) for r in rows])
        n_obs = np.array([float(r["icc_cases_obs"]) for r in rows])
        n_cor = np.array([float(r["icc_cases_corr"]) for r in rows])

        j_obs = jp_lookup[(sex, "As coded")]["jps"]
        j_cor = jp_lookup[(sex, "Corrected")]["jps"]
        fit_obs = segmented_fit(years, obs, n_obs, j_obs)
        fit_cor = segmented_fit(years, corr, n_cor, j_cor)

        ax.plot(years, fit_obs, color=OBS, lw=1.7, ls="--", zorder=2)
        ax.plot(years, fit_cor, color=COR, lw=1.9, ls="-", zorder=2)
        ax.scatter(years, obs, s=22, facecolors="white", edgecolors=OBS,
                   linewidths=0.9, zorder=3, marker="o")
        ax.scatter(years, corr, s=20, facecolors=COR, edgecolors=COR,
                   linewidths=0.4, zorder=3, marker="s")
        for jp in j_cor:
            ax.axvline(jp, color=COR, lw=0.7, ls=":", alpha=0.85, zorder=1)

        aapc_obs = jp_lookup[(sex, "As coded")]["segs"][0]["aapc"]
        aapc_cor = jp_lookup[(sex, "Corrected")]["segs"][0]["aapc"]
        jp_txt = "none" if not j_cor else str(j_cor[0])
        ax.text(
            0.03, 0.97,
            f"As coded AAPC {float(aapc_obs):.2f}%/y\n"
            f"Corrected AAPC {float(aapc_cor):.2f}%/y\n"
            f"Joinpoint {jp_txt}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.0,
            linespacing=1.35, color="0.15",
        )
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel("Year")
        ax.set_xlim(1997.2, 2017.8)
        ax.set_xticks([1998, 2002, 2006, 2010, 2014, 2017])
        ymax = max(corr.max(), obs.max())
        ax.set_ylim(0, ymax * 1.22)
        ax.yaxis.set_major_locator(MultipleLocator(0.3 if sex == "Male" else 0.2))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="0.92", lw=0.6)

    axes[0].set_ylabel("ICC incidence per 100,000")
    handles = [
        Line2D([0], [0], marker="o", color=OBS, mfc="white", lw=1.7, ls="--",
               ms=5, label="As coded"),
        Line2D([0], [0], marker="s", color=COR, mfc=COR, lw=1.9, ls="-",
               ms=5, label="Corrected"),
    ]
    axes[1].legend(handles=handles, frameon=False, loc="lower right",
                   fontsize=7.5, handlelength=2.4)
    panel_letters(axes, x=-0.08, y=1.05)
    fig.subplots_adjust(wspace=0.28, left=0.08, right=0.98, top=0.88, bottom=0.16)
    figstyle.save(fig, "Fig1_asr_joinpoint", OUT, tight=False)


def fig2(strat):
    both = [r for r in strat if r["sex"] == "Both"]

    def grab(group, name):
        for r in both:
            if r["stratum_group"] == group and r["stratum"] == name:
                return r
        raise KeyError(name)

    blocks = [
        ("Morphologic specificity", [
            ("<10% unspecified", grab("unspecified_share", "<10% (high specificity)")),
            ("10% to <40%", grab("unspecified_share", "10–<40%")),
            ("≥40% unspecified", grab("unspecified_share", "≥40%")),
        ]),
        ("Region", [
            ("Europe", grab("continent", "Europe")),
            ("North America", grab("continent", "North America")),
            ("Oceania", grab("continent", "Oceania")),
            ("Latin America", grab("continent", "Central and South America")),
            ("Asia, excluding fluke", grab(
                "fluke", "Asia excluding 4 Thai fluke registries")),
        ]),
        ("Liver fluke", [
            ("Thai endemic (4 registries)", grab(
                "fluke", "Liver-fluke endemic (4 Thai registries)")),
            ("≥40% unspecified, not fluke", grab(
                "fluke", "≥40% unspecified, excluding 4 Thai fluke registries")),
        ]),
    ]

    n_rows = sum(len(items) for _, items in blocks) + len(blocks)
    fig, ax = plt.subplots(figsize=(7.2, 5.15))
    y = 0
    ticks, labels = [], []
    y_header = []

    for header, items in blocks:
        y_header.append((y, header))
        y -= 1.0
        for lab, r in items:
            obs = float(r["obs_rise_pct"])
            cor = float(r["corr_rise_pct"])
            n = int(r["n_registries"])
            ax.plot([obs, cor], [y, y], color="0.82", lw=3.2, solid_capstyle="round",
                    zorder=1)
            ax.scatter([obs], [y], s=38, color=OBS, zorder=3, marker="o")
            ax.scatter([cor], [y], s=36, color=COR, zorder=3, marker="s")
            art = r["artefact_rise_pct"]
            if lab.startswith("Thai"):
                note = f"n={n}"
            else:
                note = f"n={n}; artefact {float(art):.1f}%"
            ax.text(max(obs, cor) + 7, y, note, va="center", ha="left",
                    fontsize=6.7, color="0.25")
            ticks.append(y)
            labels.append(lab)
            y -= 1.0
        y -= 0.25

    ax.axvline(0, color="0.55", lw=0.7, zorder=0)
    for yh, header in y_header:
        ax.text(0.015, yh - 0.08, header, fontsize=8, fontweight="bold",
                va="center", ha="left", color="0.1",
                transform=ax.get_yaxis_transform(),
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2})

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Change in ICC ASR, 1998 to 2017 (%)")
    ax.set_xlim(-150, 220)
    ax.set_ylim(y + 0.4, 0.7)
    ax.xaxis.grid(True, color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=OBS, lw=0, ms=6, label="As coded"),
            Line2D([0], [0], marker="s", color=COR, lw=0, ms=6, label="Corrected"),
        ],
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.11),
        ncol=2, fontsize=8, handletextpad=0.4, columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.31, right=0.97, top=0.97, bottom=0.16)
    figstyle.save(fig, "Fig2_stratified_rise", OUT, tight=False)


def fig3(year_rows, conf):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.85))

    ax = axes[0]
    ax.axvspan(Y0, Y1, color=SHADE, zorder=0, label="CI5 window")
    for name, ls, lw in (("SEER 8", "-", 1.8), ("SEER 17", "--", 1.6)):
        ser = [r for r in year_rows if r["database"] == name]
        ax.plot([int(r["year"]) for r in ser],
                [100 * float(r["unspec_share"]) for r in ser],
                color=OBS, lw=lw, ls=ls, label=f"{name}, unspecified")
    ax.set_xlabel("Year")
    ax.set_ylabel("Unspecified morphology (%)")
    ax.set_ylim(0, 28)
    ax.set_xlim(1974, 2024)
    ax.set_title("Unspecified share", fontsize=9, pad=4)
    ax.yaxis.grid(True, color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")

    ax = axes[1]
    order = [
        ("Hepatocellular carcinoma", "hcc"),
        ("Cholangiocarcinoma", "icc"),
        ("Unspecified morphology", "unspec_morph"),
        ("Unspecified carcinoma", "unspec_ca"),
    ]
    s17 = {r["morph"]: r for r in conf if r["database"] == "SEER 17"}
    x = np.arange(len(order))
    vals = [100 * float(s17[k]["micro_share"]) for _, k in order]
    ns = [int(float(s17[k]["n"])) for _, k in order]
    colors = [OBS, COR, GRAY, "#6B8F71"]
    bars = ax.bar(x, vals, width=0.62, color=colors, edgecolor="white", lw=0.4)
    for rect, v, n in zip(bars, vals, ns):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 1.8,
                f"{v:.1f}%\nn={n:,}", ha="center", va="bottom", fontsize=6.6,
                color="0.15")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [lab for lab, _ in order],
        fontsize=6.5,
        rotation=45,
        ha="right",
        va="top",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="x", pad=2)
    ax.set_ylabel("Microscopically confirmed (%)")
    ax.set_ylim(0, 112)
    ax.set_title("SEER 17, 2000–2017", fontsize=9, pad=4)
    ax.yaxis.grid(True, color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color="0.4", lw=0.5)

    panel_letters(axes, x=-0.10, y=1.05)
    fig.subplots_adjust(wspace=0.34, left=0.08, right=0.99, top=0.86, bottom=0.32)
    figstyle.save(fig, "Fig3_seer_validation", OUT, tight=False)


def fig_s1(strat):
    both = [r for r in strat if r["sex"] == "Both"]

    def grab(group, name):
        for r in both:
            if r["stratum_group"] == group and r["stratum"] == name:
                return r
        raise KeyError((group, name))

    panels = [
        ("Continent", [
            ("Latin America", grab("continent", "Central and South America")),
            ("North America", grab("continent", "North America")),
            ("Asia", grab("continent", "Asia")),
            ("Europe", grab("continent", "Europe")),
            ("Oceania", grab("continent", "Oceania")),
        ]),
        ("Unspecified share", [
            ("<10%", grab("unspecified_share", "<10% (high specificity)")),
            ("10–<40%", grab("unspecified_share", "10–<40%")),
            ("≥40%", grab("unspecified_share", "≥40%")),
        ]),
        ("Liver fluke", [
            ("Thai fluke", grab("fluke", "Liver-fluke endemic (4 Thai registries)")),
            ("Asia, not fluke", grab("fluke", "Asia excluding 4 Thai fluke registries")),
        ]),
    ]

    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 3.55), sharex=False,
        gridspec_kw={"width_ratios": [1.05, 1.0, 1.12]},
    )
    for ax, (title, items) in zip(axes, panels):
        y = np.arange(len(items))
        obs = [float(r["eapc_obs"]) for _, r in items]
        cor = [float(r["eapc_corr"]) for _, r in items]
        olo = [float(r["eapc_obs_lo"]) for _, r in items]
        ohi = [float(r["eapc_obs_hi"]) for _, r in items]
        clo = [float(r["eapc_corr_lo"]) for _, r in items]
        chi = [float(r["eapc_corr_hi"]) for _, r in items]
        ax.hlines(y + 0.14, olo, ohi, color=OBS, lw=1.5)
        ax.hlines(y - 0.14, clo, chi, color=COR, lw=1.5)
        ax.scatter(obs, y + 0.14, color=OBS, s=20, zorder=3, marker="o")
        ax.scatter(cor, y - 0.14, color=COR, s=18, zorder=3, marker="s")
        ax.axvline(0, color="0.55", lw=0.6)
        labels = [f"{lab} (n={int(r['n_registries'])})" for lab, r in items]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.6)
        ax.tick_params(axis="y", pad=4)
        ax.set_xlabel("EAPC (% per year)")
        ax.set_title(title, fontsize=9, pad=4)
        ax.invert_yaxis()
        xmin = min(olo + clo)
        xmax = max(ohi + chi)
        pad = 0.12 * (xmax - xmin if xmax > xmin else 1)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.xaxis.grid(True, color="0.92", lw=0.6)
        ax.set_axisbelow(True)

    handles = [
        Line2D([0], [0], marker="o", color=OBS, lw=1.5, ms=5, label="As coded"),
        Line2D([0], [0], marker="s", color=COR, lw=1.5, ms=5, label="Corrected"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=7.5,
               handletextpad=0.4, columnspacing=1.8)
    fig.suptitle("EAPC of ICC incidence, 1998–2017", fontsize=10, y=0.98)
    panel_letters(axes, x=-0.18, y=1.06)
    fig.subplots_adjust(wspace=0.62, left=0.11, right=0.99, top=0.82, bottom=0.22)
    figstyle.save(fig, "FigS1_strat_eapc", OUT, tight=False)


def fig_s2(year_rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45))
    hcc_col = "#6B8F71"
    for ax, name in zip(axes, ("SEER 8", "SEER 17")):
        ser = [r for r in year_rows if r["database"] == name]
        years = [int(r["year"]) for r in ser]
        ax.axvspan(Y0, Y1, color=SHADE, zorder=0)
        ax.plot(years, [100 * float(r["icc_among_hcc_icc"]) for r in ser],
                color=COR, lw=1.8, ls="-",
                label="ICC share of specified HCC+ICC")
        ax.plot(years, [100 * float(r["micro_among_icc"]) for r in ser],
                color=COR, lw=1.3, ls="--", label="Microscopic, coded ICC")
        ax.plot(years, [100 * float(r["micro_among_hcc"]) for r in ser],
                color=hcc_col, lw=1.3, ls="--", label="Microscopic, coded HCC")
        ax.set_title(name, fontsize=9, pad=4)
        ax.set_xlabel("Year")
        ax.set_ylabel("Percent")
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, color="0.92", lw=0.6)
        ax.set_axisbelow(True)

    handles = [
        Line2D([0], [0], color=COR, lw=1.8, ls="-",
               label="ICC share of specified HCC+ICC"),
        Line2D([0], [0], color=COR, lw=1.3, ls="--",
               label="Microscopic, coded ICC"),
        Line2D([0], [0], color=hcc_col, lw=1.3, ls="--",
               label="Microscopic, coded HCC"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=6.8,
               handletextpad=0.4, columnspacing=1.2)
    fig.suptitle(
        "SEER: ICC among specified HCC and ICC, and microscopic confirmation",
        fontsize=9, y=0.98)
    panel_letters(axes, x=-0.10, y=1.05)
    fig.subplots_adjust(wspace=0.30, left=0.08, right=0.98, top=0.80, bottom=0.24)
    figstyle.save(fig, "FigS2_seer_icc_mix", OUT, tight=False)


def main():
    figstyle.apply(**{
        "font.size": 8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
    })
    OUT.mkdir(parents=True, exist_ok=True)
    pooled = [r for r in csv_rows(CI5 / "morph_pooled_series.csv")
              if r["scenario"] == "M1_proportional"]
    jp = csv_rows(TAB / "T_morph_joinpoint.csv")
    strat = csv_rows(TAB / "T_morph_stratified.csv")
    year_rows = csv_rows(TAB / "T_seer_year.csv")
    conf = csv_rows(TAB / "T_seer_confirmation.csv")

    fig1(pooled, jp)
    fig2(strat)
    fig3(year_rows, conf)
    fig_s1(strat)
    fig_s2(year_rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
