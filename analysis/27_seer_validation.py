#!/usr/bin/env python3
"""SEER external validation for Paper 2 morphology redistribution.

Question
--------
CI5 M1 reallocates morphologically unspecified liver cancers in proportion to
specified morphologies. SEER cannot reveal the 'true' type of unspecified
cases, but it can test the two mechanisms that would break M1:

  1. Unspecified share fell while microscopic confirmation rose
     (coding/diagnosis quality, the artefact channel).
  2. Unspecified and non-microscopic cases are not a random draw from the
     specified mix. In US data, HCC is often imaging-diagnosed; ICC is more
     often biopsied. If so, the unspecified pool is HCC-enriched in SEER,
     the opposite of the liver-fluke prior (unspecified ~ ICC).

Inputs (data/raw/seer/):
  seer8_1975-2023_hepatobiliary.csv
  seer17_2000-2023_hepatobiliary.csv
  Death-certificate-only cases are Type of Reporting Source
  == "Death certificate only", not a Diagnostic Confirmation code.

Outputs (tables/):
  T_seer_year.csv              year x database, C22 composition
  T_seer_confirmation.csv      morphology x confirmation
  T_seer_site_morph.csv        site x morphology
  seer_validation_summary.json
Figures (figures/paper2/):
  FigP2_seer_unspec_micro
  FigP2_seer_icc_specified_mix
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import figstyle

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/seer"
TAB = ROOT / "tables"
FIG = ROOT / "figures/paper2"

FILES = {
    "SEER 8": RAW / "seer8_1975-2023_hepatobiliary.csv",
    "SEER 17": RAW / "seer17_2000-2023_hepatobiliary.csv",
}

CI5_Y0, CI5_Y1 = 1998, 2017
C_OBS, C_COR = "#4C72B0", "#C44E52"
C_ICC, C_HCC = "#C44E52", "#55A868"

MICRO_LABELS = {
    "Positive histology",
    "Positive exfoliative cytology, no positive histology",
    "Pos hist AND immunophenotyping AND/OR pos genetic studies",
    "Positive microscopic confirm, method not specified",
}

YOUNG_AGES = {
    "00 years", "01-04 years", "05-09 years", "10-14 years", "15-19 years",
    "20-24 years", "25-29 years", "30-34 years", "35-39 years", "40-44 years",
    "45-49 years",
}

figstyle.apply(**{"font.size": 9, "figure.dpi": 150})


def parse_hist(raw: str) -> tuple[int, str]:
    token = raw.split(":")[0].strip()
    code_s, _, beh = token.partition("/")
    try:
        return int(code_s), beh
    except ValueError:
        return -1, ""


def site_group(label: str) -> str:
    if label.startswith("C22.0"):
        return "C22.0_liver"
    if label.startswith("C22.1"):
        return "C22.1_icc_site"
    if label.startswith("C24.0"):
        return "C24.0_ecc"
    if label.startswith("C24.1"):
        return "C24.1_ampulla"
    if label.startswith("C24.8") or label.startswith("C24.9"):
        return "C24_other"
    return "other"


def morph_group(code: int) -> str:
    if 8170 <= code <= 8175:
        return "hcc"
    if code in (8160, 8161):
        return "icc"
    if code == 8180:
        return "combined"
    if code == 8162:
        return "klatskin"
    if 8000 <= code <= 8005:
        return "unspec_morph"
    if code in (8010, 8012, 8020, 8021, 8022, 8140):
        return "unspec_ca"
    return "other_spec"


def is_micro(label: str) -> bool:
    return label in MICRO_LABELS


def is_unspec(g: str) -> bool:
    return g in ("unspec_ca", "unspec_morph")


def eapc_ols(years, values):
    pts = [(y, v) for y, v in zip(years, values) if v is not None and v > 0]
    if len(pts) < 5:
        return None
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.log(np.array([p[1] for p in pts], dtype=float))
    n = len(x)
    mx, my = x.mean(), y.mean()
    sxx = float(((x - mx) ** 2).sum())
    b = float(((x - mx) * (y - my)).sum() / sxx)
    resid = y - (my - b * mx + b * x)
    dof = n - 2
    se = float(np.sqrt((resid @ resid) / dof / sxx))
    t = 1.96 if dof > 60 else 2.101 if dof > 15 else 2.365
    return {
        "eapc": 100 * (math.exp(b) - 1),
        "lo": 100 * (math.exp(b - t * se) - 1),
        "hi": 100 * (math.exp(b + t * se) - 1),
        "n": n,
    }


def or_ci(a, b, c, d):
    """OR for (exposed, case)/(exposed, ctrl) vs unexposed. Woolf interval."""
    aa, bb, cc, dd = a, b, c, d
    if min(a, b, c, d) == 0:
        aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    odds = (aa / bb) / (cc / dd)
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    lo, hi = math.exp(math.log(odds) - 1.96 * se), math.exp(math.log(odds) + 1.96 * se)
    return {"or": odds, "lo": lo, "hi": hi, "a": a, "b": b, "c": c, "d": d}


def load_file(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            year = int(r["Year of diagnosis"])
            code, beh = parse_hist(r["ICD-O-3 Hist/behav"])
            sg = site_group(r["Primary Site - labeled"])
            mg = morph_group(code)
            rows.append({
                "year": year,
                "sex": r["Sex"],
                "age": r["Age recode with <1 year olds and 90+"],
                "young": r["Age recode with <1 year olds and 90+"] in YOUNG_AGES,
                "site": sg,
                "site_raw": r["Primary Site - labeled"],
                "hist_code": code,
                "morph": mg,
                "unspec": is_unspec(mg),
                "micro": is_micro(r["Diagnostic Confirmation"]),
                "dx": r["Diagnostic Confirmation"],
                "dco": r.get("Type of Reporting Source", "") == "Death certificate only",
                "c22": sg in ("C22.0_liver", "C22.1_icc_site"),
            })
    return rows


def c22(rows):
    return [r for r in rows if r["c22"]]


def year_block(rows, y0, y1):
    bag = {}
    for r in rows:
        y = r["year"]
        if y < y0 or y > y1:
            continue
        b = bag.get(y)
        if b is None:
            b = bag[y] = Counter()
        b["n"] += 1
        if r["unspec"]:
            b["unspec"] += 1
            if r["micro"]:
                b["unspec_micro"] += 1
        if r["morph"] == "hcc":
            b["hcc"] += 1
            if r["micro"]:
                b["hcc_micro"] += 1
        elif r["morph"] == "icc":
            b["icc"] += 1
            if r["micro"]:
                b["icc_micro"] += 1
        if r["micro"]:
            b["micro"] += 1
        if r["site"] == "C22.1_icc_site":
            b["c22_1"] += 1
    out = []
    for y in range(y0, y1 + 1):
        b = bag.get(y)
        if not b:
            continue
        n, n_un = b["n"], b["unspec"]
        n_hcc, n_icc = b["hcc"], b["icc"]
        spec = n - n_un
        out.append({
            "year": y, "n": n, "unspec": n_un, "hcc": n_hcc, "icc": n_icc,
            "micro": b["micro"], "c22_1": b["c22_1"],
            "icc_micro": b["icc_micro"], "hcc_micro": b["hcc_micro"],
            "unspec_share": n_un / n if n else None,
            "micro_share": b["micro"] / n if n else None,
            "icc_among_specified": n_icc / spec if spec else None,
            "icc_among_hcc_icc": n_icc / (n_icc + n_hcc) if (n_icc + n_hcc) else None,
            "c22_1_share": b["c22_1"] / n if n else None,
            "micro_among_hcc": b["hcc_micro"] / n_hcc if n_hcc else None,
            "micro_among_icc": b["icc_micro"] / n_icc if n_icc else None,
            "micro_among_unspec": b["unspec_micro"] / n_un if n_un else None,
        })
    return out


def fmt_ci(e):
    if not e:
        return ""
    return f"{e['eapc']:.2f} ({e['lo']:.2f} to {e['hi']:.2f})"


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    TAB.mkdir(exist_ok=True)
    data = {}
    for name, path in FILES.items():
        if not path.exists():
            raise SystemExit(f"missing {path}")
        data[name] = load_file(path)
        print(f"{name}: {len(data[name])} rows, "
              f"{sum(r['c22'] for r in data[name])} C22", flush=True)

    year_rows = []
    summary = {"files": {}, "tests": {}}
    for name, rows in data.items():
        liver = c22(rows)
        years = [r["year"] for r in liver]
        y0, y1 = min(years), max(years)
        series = year_block(liver, y0, y1)
        for rec in series:
            year_rows.append({"database": name, **{
                k: (round(v, 5) if isinstance(v, float) else v)
                for k, v in rec.items()}})
        summary["files"][name] = {
            "n_all": len(rows),
            "n_c22": len(liver),
            "window": [y0, y1],
        }

        # CI5-overlapping window
        ov = [r for r in liver if CI5_Y0 <= r["year"] <= CI5_Y1]
        ys = sorted({r["year"] for r in ov})
        start_y, end_y = ys[0], ys[-1]
        s0 = [r for r in ov if r["year"] == start_y]
        s1 = [r for r in ov if r["year"] == end_y]

        def share_unspec(sub):
            return sum(r["unspec"] for r in sub) / len(sub) if sub else None

        def mix(sub):
            h = sum(r["morph"] == "hcc" for r in sub)
            i = sum(r["morph"] == "icc" for r in sub)
            return i / (i + h) if (i + h) else None

        ser_ov = [r for r in series if start_y <= r["year"] <= end_y]
        e_un = eapc_ols([r["year"] for r in ser_ov],
                        [r["unspec_share"] for r in ser_ov])
        e_icc_cnt = eapc_ols([r["year"] for r in ser_ov],
                             [r["icc"] for r in ser_ov])
        e_icc_mic = eapc_ols([r["year"] for r in ser_ov],
                             [r["icc_micro"] for r in ser_ov])
        e_hcc_cnt = eapc_ols([r["year"] for r in ser_ov],
                             [r["hcc"] for r in ser_ov])

        # confirmation OR: unspecified vs specified, non-micro vs micro
        un_nm = sum(r["unspec"] and not r["micro"] for r in ov)
        un_m = sum(r["unspec"] and r["micro"] for r in ov)
        sp_nm = sum((not r["unspec"]) and not r["micro"] for r in ov)
        sp_m = sum((not r["unspec"]) and r["micro"] for r in ov)
        # OR of being non-microscopic if unspecified
        odds = or_ci(un_nm, un_m, sp_nm, sp_m)

        hcc = [r for r in ov if r["morph"] == "hcc"]
        icc = [r for r in ov if r["morph"] == "icc"]
        hcc_micro = sum(r["micro"] for r in hcc) / len(hcc) if hcc else None
        icc_micro = sum(r["micro"] for r in icc) / len(icc) if icc else None

        summary["tests"][name] = {
            "ci5_window": [start_y, end_y],
            "n_c22": len(ov),
            "unspec_share_start": round(share_unspec(s0), 4),
            "unspec_share_end": round(share_unspec(s1), 4),
            "unspec_share_eapc": fmt_ci(e_un),
            "icc_hcc_mix_start": round(mix(s0), 4) if mix(s0) is not None else None,
            "icc_hcc_mix_end": round(mix(s1), 4) if mix(s1) is not None else None,
            "icc_count_eapc": fmt_ci(e_icc_cnt),
            "icc_microscopic_count_eapc": fmt_ci(e_icc_mic),
            "hcc_count_eapc": fmt_ci(e_hcc_cnt),
            "micro_share_hcc": round(hcc_micro, 4) if hcc_micro is not None else None,
            "micro_share_icc": round(icc_micro, 4) if icc_micro is not None else None,
            "or_unspec_nonmicroscopic": {
                "or": round(odds["or"], 2),
                "lo": round(odds["lo"], 2),
                "hi": round(odds["hi"], 2),
                "unspec_nonmicro": un_nm, "unspec_micro": un_m,
                "spec_nonmicro": sp_nm, "spec_micro": sp_m,
            },
        }

        # site x morph in CI5 window
        sm = Counter((r["site"], r["morph"]) for r in ov)

    write_csv(TAB / "T_seer_year.csv", year_rows)

    conf_rows, site_rows = [], []
    for name, rows in data.items():
        ov = [r for r in c22(rows) if CI5_Y0 <= r["year"] <= CI5_Y1]
        for mg in ("hcc", "icc", "combined", "klatskin", "unspec_ca",
                   "unspec_morph", "other_spec"):
            sub = [r for r in ov if r["morph"] == mg]
            if not sub:
                continue
            n_m = sum(r["micro"] for r in sub)
            conf_rows.append({
                "database": name, "morph": mg, "n": len(sub),
                "microscopic": n_m, "not_microscopic": len(sub) - n_m,
                "micro_share": round(n_m / len(sub), 4),
            })
        for sg in ("C22.0_liver", "C22.1_icc_site"):
            sub = [r for r in ov if r["site"] == sg]
            n = len(sub)
            site_rows.append({
                "database": name, "site": sg, "n": n,
                "hcc": sum(r["morph"] == "hcc" for r in sub),
                "icc": sum(r["morph"] == "icc" for r in sub),
                "unspec": sum(r["unspec"] for r in sub),
                "unspec_share": round(sum(r["unspec"] for r in sub) / n, 4) if n else "",
                "micro_share": round(sum(r["micro"] for r in sub) / n, 4) if n else "",
            })
    write_csv(TAB / "T_seer_confirmation.csv", conf_rows)
    write_csv(TAB / "T_seer_site_morph.csv", site_rows)

    # Death-certificate-only is Type of Reporting Source, not Diagnostic Confirmation.
    dco_out = {}
    for name, rows in data.items():
        ov = [r for r in c22(rows) if CI5_Y0 <= r["year"] <= CI5_Y1]
        buckets = {}
        for r in ov:
            g = r["morph"] if r["morph"] in ("unspec_morph", "unspec_ca", "hcc", "icc") else "other_spec"
            slot = buckets.setdefault(g, {"n": 0, "dco": 0})
            slot["n"] += 1
            slot["dco"] += int(r["dco"])
        dco_out[name] = {
            g: {"n": slot["n"], "dco": slot["dco"],
                "dco_pct": round(100 * slot["dco"] / slot["n"], 1)}
            for g, slot in buckets.items()
        }
    (TAB / "revision_seer_dco.json").write_text(
        json.dumps(dco_out, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["death_certificate_only"] = dco_out

    # age split, SEER 17 (larger)
    se17 = [r for r in c22(data["SEER 17"]) if CI5_Y0 <= r["year"] <= CI5_Y1]
    age_note = {}
    for flag, lab in ((True, "<50"), (False, "50+")):
        sub = [r for r in se17 if r["young"] is flag]
        s0 = [r for r in sub if r["year"] == 2000]
        s1 = [r for r in sub if r["year"] == 2017]
        age_note[lab] = {
            "n": len(sub),
            "unspec_share_2000": round(sum(r["unspec"] for r in s0) / len(s0), 4) if s0 else None,
            "unspec_share_2017": round(sum(r["unspec"] for r in s1) / len(s1), 4) if s1 else None,
            "icc_hcc_mix_2000": round(
                sum(r["morph"] == "icc" for r in s0) /
                max(sum(r["morph"] in ("icc", "hcc") for r in s0), 1), 4),
            "icc_hcc_mix_2017": round(
                sum(r["morph"] == "icc" for r in s1) /
                max(sum(r["morph"] in ("icc", "hcc") for r in s1), 1), 4),
        }
    summary["age_seer17"] = age_note
    summary["morph_definition"] = {
        "hcc": "ICD-O-3 8170-8175",
        "icc": "8160, 8161",
        "unspec_morph": "8000-8005",
        "unspec_ca": "8010, 8012, 8020-8022, 8140 (adenocarcinoma NOS)",
        "universe": "C22.0 + C22.1 only",
        "ci5_window": [CI5_Y0, CI5_Y1],
    }

    (TAB / "seer_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- figures ----
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for ax, name in zip(axes, ("SEER 8", "SEER 17")):
        ser = [r for r in year_rows if r["database"] == name]
        ax.plot([r["year"] for r in ser], [100 * r["unspec_share"] for r in ser],
                color=C_OBS, lw=1.8, label="Unspecified morphology")
        ax.plot([r["year"] for r in ser], [100 * r["micro_share"] for r in ser],
                color=C_COR, lw=1.8, label="Microscopically confirmed")
        ax.axvspan(CI5_Y0, CI5_Y1, color="0.85", zorder=0)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Year")
        ax.set_ylabel("Percent of C22 cases")
        ax.set_ylim(0, 100)
    axes[0].legend(frameon=False, fontsize=7.5, loc="center right")
    figstyle.panel_labels(axes)
    figstyle.save(fig, "FigP2_seer_unspec_micro", FIG)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for ax, name in zip(axes, ("SEER 8", "SEER 17")):
        ser = [r for r in year_rows if r["database"] == name]
        ax.plot([r["year"] for r in ser],
                [100 * r["icc_among_hcc_icc"] if r["icc_among_hcc_icc"] is not None else np.nan
                 for r in ser],
                color=C_ICC, lw=1.8, label="ICC / (ICC+HCC) specified")
        ax.plot([r["year"] for r in ser],
                [100 * r["micro_among_icc"] if r["micro_among_icc"] is not None else np.nan
                 for r in ser],
                color=C_ICC, lw=1.2, ls="--", label="% ICC microscopic")
        ax.plot([r["year"] for r in ser],
                [100 * r["micro_among_hcc"] if r["micro_among_hcc"] is not None else np.nan
                 for r in ser],
                color=C_HCC, lw=1.2, ls="--", label="% HCC microscopic")
        ax.axvspan(CI5_Y0, CI5_Y1, color="0.85", zorder=0)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Year")
        ax.set_ylabel("Percent")
        ax.set_ylim(0, 100)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    figstyle.panel_labels(axes)
    figstyle.save(fig, "FigP2_seer_icc_specified_mix", FIG)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
