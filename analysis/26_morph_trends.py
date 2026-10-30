#!/usr/bin/env python3
"""EAPC, joinpoint, bootstrap UI, and stratification for Paper 2.

Inputs are the M1 morphology-redistribution outputs. Joinpoint and EAPC
are implemented in this file (death-weighted segmented log-linear;
unweighted OLS of ln(ASR) on year).

  EAPC       unweighted OLS of ln(ASR) on calendar year
  Joinpoint  death-weighted segmented log-linear; permutation test on the
             pooled series, BIC3 on registry series
  Bootstrap  Poisson case counts + Beta weight uncertainty from the same
             donor stratum that M1 used (B=1000)

Outputs (tables/):
  T_morph_eapc.csv
  T_morph_joinpoint.csv
  T_morph_stratified.csv
  T_morph_eapc_UI.json
  morph_trends_summary.json
Figures (figures/paper2/):
  FigP2_asr_joinpoint
  FigP2_strat_eapc
"""

from __future__ import annotations

import csv
import importlib.util
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import figstyle

ROOT = Path(__file__).resolve().parents[1]
CI5 = ROOT / "data/processed/ci5"
TAB = ROOT / "tables"
FIG = ROOT / "figures/paper2"

Y0, Y1 = 1998, 2017
SCEN = "M1_proportional"
B = 1000
SEED = 20260813
MAX_JP = 3
MIN_SEG = 4
N_PERM = 999
ALPHA = 0.05

FLUKE = {
    "76400299": "Thailand, Khon Kaen",
    "76400199": "Thailand, Chiang Mai",
    "76400599": "Thailand, Lampang",
    "76400499": "Thailand, Songkhla",
}

AGE_ORDER = [
    "0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39",
    "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74",
    "75-79", "80-84", "85+",
]
YOUNG = AGE_ORDER[:10]
OLD = AGE_ORDER[10:]
SEGI = {
    "0-4": 12000, "5-9": 10000, "10-14": 9000, "15-19": 9000, "20-24": 8000,
    "25-29": 8000, "30-34": 6000, "35-39": 6000, "40-44": 6000, "45-49": 6000,
    "50-54": 5000, "55-59": 4000, "60-64": 4000, "65-69": 3000, "70-74": 2000,
    "75-79": 1000, "80-84": 500, "85+": 500,
}
_TOT = sum(SEGI.values())
STD_W = {k: v / _TOT for k, v in SEGI.items()}

C_OBS, C_COR = "#4C72B0", "#C44E52"
figstyle.apply(**{"font.size": 9, "figure.dpi": 150})


def load_morph25():
    path = ROOT / "analysis/25_morphology_redistribution.py"
    spec = importlib.util.spec_from_file_location("morph25", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------- joinpoint core -----------------------------
def design(x, jps):
    cols = [np.ones_like(x, dtype=float), x.astype(float)]
    for t in jps:
        cols.append(np.maximum(0.0, x - t))
    return np.column_stack(cols)


def wls(X, y, w):
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    return beta, float(resid @ resid), Xw


def candidate_sets(x, k):
    n = len(x)
    idx = range(MIN_SEG - 1, n - MIN_SEG)
    for combo in combinations(idx, k):
        if all(combo[i + 1] - combo[i] >= MIN_SEG for i in range(len(combo) - 1)):
            yield [float(x[i]) for i in combo]


def fit_k(x, y, w, k):
    if k == 0:
        X = design(x, [])
        beta, sse, Xw = wls(X, y, w)
        return {"jps": [], "beta": beta, "sse": sse, "X": X, "Xw": Xw}
    best = None
    for jps in candidate_sets(x, k):
        try:
            beta, sse, Xw = wls(design(x, jps), y, w)
        except np.linalg.LinAlgError:
            continue
        X = design(x, jps)
        if best is None or sse < best["sse"]:
            best = {"jps": jps, "beta": beta, "sse": sse, "X": X, "Xw": Xw}
    return best


def bic3(sse, n, n_params):
    return np.log(sse / n) + n_params * np.log(n) * 3.0 / n


def permutation_test(x, y, w, k0, k1, rng, n_perm=N_PERM):
    f0, f1 = fit_k(x, y, w, k0), fit_k(x, y, w, k1)
    if f0 is None or f1 is None:
        return 1.0, f0, f1
    stat = (f0["sse"] - f1["sse"]) / f1["sse"] if f1["sse"] > 0 else 0.0
    sw = np.sqrt(w)
    fitted0 = f0["X"] @ f0["beta"]
    resid_w = (y - fitted0) * sw
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(resid_w)
        y_star = fitted0 + perm / sw
        g0, g1 = fit_k(x, y_star, w, k0), fit_k(x, y_star, w, k1)
        if g0 is None or g1 is None or g1["sse"] <= 0:
            continue
        if (g0["sse"] - g1["sse"]) / g1["sse"] >= stat:
            count += 1
    return (count + 1) / (n_perm + 1), f0, f1


def segment_stats(x, y, w, fit):
    X, beta = fit["X"], fit["beta"]
    n, p = X.shape
    dof = n - p
    sw = np.sqrt(w)
    resid = (y - X @ beta) * sw
    s2 = float(resid @ resid) / dof if dof > 0 else np.nan
    cov = s2 * np.linalg.pinv((X * w[:, None]).T @ X)
    tcrit = 2.0 if dof > 30 else 2.145 if dof > 12 else 2.571
    jps = fit["jps"]
    bounds = [float(x[0])] + list(jps) + [float(x[-1])]
    segs = []
    for i in range(len(bounds) - 1):
        c = np.zeros(len(beta))
        c[1] = 1.0
        for j in range(i):
            c[2 + j] = 1.0
        slope = float(c @ beta)
        se = float(np.sqrt(max(c @ cov @ c, 0.0)))
        segs.append({
            "start": bounds[i], "end": bounds[i + 1],
            "apc": 100 * (np.exp(slope) - 1),
            "lo": 100 * (np.exp(slope - tcrit * se) - 1),
            "hi": 100 * (np.exp(slope + tcrit * se) - 1),
            "slope": slope, "se": se,
            "significant": (slope - tcrit * se) * (slope + tcrit * se) > 0,
        })
    total = bounds[-1] - bounds[0]
    weights = [(s["end"] - s["start"]) / total for s in segs]
    aapc_slope = sum(wt * s["slope"] for wt, s in zip(weights, segs))
    c = np.zeros(len(beta))
    for i, wt in enumerate(weights):
        c[1] += wt
        for j in range(i):
            c[2 + j] += wt
    se_a = float(np.sqrt(max(c @ cov @ c, 0.0)))
    aapc = {
        "aapc": 100 * (np.exp(aapc_slope) - 1),
        "lo": 100 * (np.exp(aapc_slope - tcrit * se_a) - 1),
        "hi": 100 * (np.exp(aapc_slope + tcrit * se_a) - 1),
        "significant": (aapc_slope - tcrit * se_a) * (aapc_slope + tcrit * se_a) > 0,
    }
    return segs, aapc


def select_model(x, y, w, method, rng=None):
    n = len(x)
    kmax = min(MAX_JP, max(0, n // MIN_SEG - 1))
    fits = {}
    for k in range(kmax + 1):
        f = fit_k(x, y, w, k)
        if f is not None:
            fits[k] = f
    if not fits:
        return None, None
    if method == "bic3":
        best_k = min(fits, key=lambda k: bic3(fits[k]["sse"], n, fits[k]["X"].shape[1] + k))
        return best_k, fits[best_k]
    best_k = 0
    for k in range(1, kmax + 1):
        p, _, _ = permutation_test(x, y, w, best_k, k, rng)
        if p < ALPHA:
            best_k = k
        else:
            break
    return best_k, fits[best_k]


# --------------------------------- EAPC ----------------------------------
def eapc_ols(years, rates):
    ok = np.asarray(rates) > 0
    if ok.sum() < 5:
        return None
    x = np.asarray(years, dtype=float)[ok]
    y = np.log(np.asarray(rates, dtype=float)[ok])
    n = len(x)
    mx, my = x.mean(), y.mean()
    sxx = float(((x - mx) ** 2).sum())
    b = float(((x - mx) * (y - my)).sum() / sxx)
    resid = y - (my - b * mx + b * x)
    dof = n - 2
    se = float(np.sqrt((resid @ resid) / dof / sxx))
    t = 1.96 if dof > 60 else 2.101 if dof > 15 else 2.365
    return {
        "eapc": 100 * (np.exp(b) - 1),
        "lo": 100 * (np.exp(b - t * se) - 1),
        "hi": 100 * (np.exp(b + t * se) - 1),
        "n": int(n),
    }


def eapc_point(years, rates):
    ok = np.asarray(rates) > 0
    if ok.sum() < 3:
        return np.nan
    x = np.asarray(years, dtype=float)[ok]
    y = np.log(np.asarray(rates, dtype=float)[ok])
    b = np.polyfit(x, y, 1)[0]
    return float((np.exp(b) - 1) * 100)


def rise_pct(start, end):
    if not start or start <= 0:
        return None
    return 100.0 * (end / start - 1.0)


def artefact_pct(obs_rise, corr_rise):
    if not obs_rise or corr_rise is None:
        return None
    return 100.0 * (1.0 - corr_rise / obs_rise)


def fmt_ci(a, lo, hi):
    return f"{a:.2f} ({lo:.2f} to {hi:.2f})"


# ------------------------------ ASR pooling ------------------------------
def asr_from_maps(deaths, pop, ages):
    w = {a: STD_W[a] for a in ages}
    tw = sum(w.values())
    total = used = 0.0
    for a in ages:
        p = pop.get(a, 0.0)
        if p <= 0:
            continue
        total += w[a] * deaths.get(a, 0.0) / p
        used += w[a]
    if used < 0.99 * tw:
        return None
    return total / tw * 1e5


def series_from_age_rows(rows, ages=None):
    """rows: iterable of dicts with year, age_group, person_years, icc_obs, icc_corr."""
    ages = ages or AGE_ORDER
    bag = defaultdict(lambda: {"pop": defaultdict(float),
                               "o": defaultdict(float),
                               "c": defaultdict(float),
                               "n_o": 0.0, "n_c": 0.0})
    for r in rows:
        a = r["age_group"]
        if a not in ages:
            continue
        y = r["year"]
        bag[y]["pop"][a] += r["person_years"]
        bag[y]["o"][a] += r["icc_obs"]
        bag[y]["c"][a] += r["icc_corr"]
        bag[y]["n_o"] += r["icc_obs"]
        bag[y]["n_c"] += r["icc_corr"]
    years, obs, corr, n_o, n_c = [], [], [], [], []
    for y in range(Y0, Y1 + 1):
        if y not in bag:
            continue
        ao = asr_from_maps(bag[y]["o"], bag[y]["pop"], ages)
        ac = asr_from_maps(bag[y]["c"], bag[y]["pop"], ages)
        if ao is None or ac is None:
            continue
        years.append(y)
        obs.append(ao)
        corr.append(ac)
        n_o.append(bag[y]["n_o"])
        n_c.append(bag[y]["n_c"])
    return (np.array(years, dtype=float), np.array(obs), np.array(corr),
            np.array(n_o), np.array(n_c))


def trend_block(years, obs, corr, n_o, n_c, jp_method, rng):
    out = {"n_years": int(len(years))}
    if len(years) < 5:
        return out
    def rnd(v, nd):
        return None if v is None else round(float(v), nd)

    out["asr_obs_start"] = rnd(obs[0], 5)
    out["asr_obs_end"] = rnd(obs[-1], 5)
    out["asr_corr_start"] = rnd(corr[0], 5)
    out["asr_corr_end"] = rnd(corr[-1], 5)
    out["obs_rise_pct"] = rnd(rise_pct(obs[0], obs[-1]), 1)
    out["corr_rise_pct"] = rnd(rise_pct(corr[0], corr[-1]), 1)
    out["artefact_rise_pct"] = rnd(artefact_pct(out["obs_rise_pct"],
                                                out["corr_rise_pct"]), 1)
    eo, ec = eapc_ols(years, obs), eapc_ols(years, corr)
    if eo and ec:
        out["eapc_obs"] = round(eo["eapc"], 3)
        out["eapc_obs_lo"] = round(eo["lo"], 3)
        out["eapc_obs_hi"] = round(eo["hi"], 3)
        out["eapc_corr"] = round(ec["eapc"], 3)
        out["eapc_corr_lo"] = round(ec["lo"], 3)
        out["eapc_corr_hi"] = round(ec["hi"], 3)
        out["eapc_diff"] = round(ec["eapc"] - eo["eapc"], 3)
        out["artefact_eapc_pct"] = round(artefact_pct(eo["eapc"], ec["eapc"]), 1)
    jp = {}
    for tag, rate, wt in (("obs", obs, n_o), ("corr", corr, n_c)):
        ylog = np.log(np.maximum(rate, 1e-12))
        w = np.maximum(wt, 1.0)
        k, fit = select_model(years, ylog, w, jp_method, rng)
        if fit is None:
            continue
        segs, aapc = segment_stats(years, ylog, w, fit)
        jp[tag] = {
            "joinpoints": int(k),
            "joinpoint_years": [int(t) for t in fit["jps"]],
            "aapc": round(aapc["aapc"], 2),
            "aapc_lo": round(aapc["lo"], 2),
            "aapc_hi": round(aapc["hi"], 2),
            "segments": [{
                "period": f"{int(s['start'])}-{int(s['end'])}",
                "apc": round(s["apc"], 2),
                "lo": round(s["lo"], 2),
                "hi": round(s["hi"], 2),
                "significant": bool(s["significant"]),
            } for s in segs],
            "fit": fit,
            "ylog": ylog,
        }
    out["joinpoint"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("fit", "ylog")}
                        for k, v in jp.items()}
    out["_jp"] = jp
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# --------------------------------- main ----------------------------------
def main():
    TAB.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    regs = list(csv.DictReader(open(CI5 / "morph_panel_registries.csv", encoding="utf-8")))
    for r in regs:
        r["unspecified_share"] = float(r["unspecified_share"])
        r["liver_cases"] = float(r["liver_cases"])
        r["high_specificity"] = int(r["high_specificity"])
    by_id = {r["id_code"]: r for r in regs}

    def unspec_band(share):
        if share < 0.10:
            return "<10% (high specificity)"
        if share < 0.40:
            return "10–<40%"
        return "≥40%"

    age_rows = []
    with open(CI5 / "morph_corrected_age.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["scenario"] != SCEN:
                continue
            rec = {
                "id_code": r["id_code"],
                "registry": r["registry"],
                "continent": r["continent"],
                "year": int(r["year"]),
                "sex": r["sex"],
                "age_group": r["age_group"],
                "person_years": float(r["person_years"]),
                "icc_obs": float(r["icc_obs"]),
                "icc_corr": float(r["icc_corr"]),
            }
            meta = by_id[rec["id_code"]]
            rec["unspec_band"] = unspec_band(meta["unspecified_share"])
            rec["fluke"] = rec["id_code"] in FLUKE
            age_rows.append(rec)
    print(f"M1 age rows {len(age_rows)} registries {len(by_id)}")

    # ---- pooled + strata (point estimates) ----
    strata = []

    def add_stratum(name, group, subset, sex_label, sex_code, jp_method):
        years, obs, corr, n_o, n_c = series_from_age_rows(subset)
        n_reg = len({r["id_code"] for r in subset})
        blk = trend_block(years, obs, corr, n_o, n_c, jp_method, rng)
        blk.update({
            "stratum_group": group, "stratum": name,
            "sex": sex_label, "sex_code": sex_code,
            "n_registries": n_reg,
        })
        strata.append(blk)
        return blk

    headline = {}
    for sex_code, sex_label in (("1", "Male"), ("2", "Female"), ("both", "Both")):
        if sex_code == "both":
            sub = age_rows
        else:
            sub = [r for r in age_rows if r["sex"] == sex_code]
        method = "permutation" if sex_code != "both" else "permutation"
        blk = add_stratum("All 73 registries", "panel", sub, sex_label, sex_code, method)
        headline[sex_label] = {k: v for k, v in blk.items() if not k.startswith("_")}

        for ages, alab in ((YOUNG, "<50 years"), (OLD, "50+ years")):
            yrs, ob, co, no, nc = series_from_age_rows(sub, ages)
            b = trend_block(yrs, ob, co, no, nc, "bic3", rng)
            b.update({
                "stratum_group": "age", "stratum": alab,
                "sex": sex_label, "sex_code": sex_code,
                "n_registries": len({r["id_code"] for r in sub}),
            })
            strata.append(b)

    continents = sorted({r["continent"] for r in age_rows})
    bands = ["<10% (high specificity)", "10–<40%", "≥40%"]
    for sex_code, sex_label in (("1", "Male"), ("2", "Female"), ("both", "Both")):
        base = age_rows if sex_code == "both" else [r for r in age_rows if r["sex"] == sex_code]
        for cont in continents:
            sub = [r for r in base if r["continent"] == cont]
            if not sub:
                continue
            add_stratum(cont, "continent", sub, sex_label, sex_code, "bic3")
        for band in bands:
            sub = [r for r in base if r["unspec_band"] == band]
            add_stratum(band, "unspecified_share", sub, sex_label, sex_code, "bic3")
        add_stratum("Liver-fluke endemic (4 Thai registries)", "fluke",
                    [r for r in base if r["fluke"]], sex_label, sex_code, "bic3")
        add_stratum("Asia excluding 4 Thai fluke registries", "fluke",
                    [r for r in base if r["continent"] == "Asia" and not r["fluke"]],
                    sex_label, sex_code, "bic3")
        add_stratum("≥40% unspecified, excluding 4 Thai fluke registries", "fluke",
                    [r for r in base if r["unspec_band"] == "≥40%" and not r["fluke"]],
                    sex_label, sex_code, "bic3")

    # scenario sensitivity from the already-pooled file
    scen_eapc = []
    pooled = list(csv.DictReader(open(CI5 / "morph_pooled_series.csv", encoding="utf-8")))
    for r in pooled:
        r["year"] = int(r["year"])
        for k in ("icc_asr_obs", "icc_asr_corr", "icc_cases_obs", "icc_cases_corr"):
            r[k] = float(r[k])
    for scen in sorted({r["scenario"] for r in pooled}):
        for sx, slab in (("1", "Male"), ("2", "Female")):
            ser = sorted([r for r in pooled if r["scenario"] == scen and r["sex"] == sx],
                         key=lambda r: r["year"])
            years = np.array([r["year"] for r in ser], dtype=float)
            obs = np.array([r["icc_asr_obs"] for r in ser])
            corr = np.array([r["icc_asr_corr"] for r in ser])
            eo, ec = eapc_ols(years, obs), eapc_ols(years, corr)
            scen_eapc.append({
                "scenario": scen, "sex": slab,
                "eapc_obs": round(eo["eapc"], 3) if eo else "",
                "eapc_obs_ci": fmt_ci(eo["eapc"], eo["lo"], eo["hi"]) if eo else "",
                "eapc_corr": round(ec["eapc"], 3) if ec else "",
                "eapc_corr_ci": fmt_ci(ec["eapc"], ec["lo"], ec["hi"]) if ec else "",
                "eapc_diff": round(ec["eapc"] - eo["eapc"], 3) if eo and ec else "",
                "artefact_eapc_pct": round(artefact_pct(eo["eapc"], ec["eapc"]), 1)
                if eo and ec else "",
                "obs_rise_pct": round(rise_pct(obs[0], obs[-1]), 1),
                "corr_rise_pct": round(rise_pct(corr[0], corr[-1]), 1),
                "artefact_rise_pct": round(artefact_pct(rise_pct(obs[0], obs[-1]),
                                                        rise_pct(corr[0], corr[-1])), 1),
            })

    # ---- bootstrap UI on the 73-registry panel ----
    print("building M1 donor counts for bootstrap")
    m = load_morph25()
    panel = m.build_panel()
    cells, pys = m.load_cells(panel)
    levels = m.build_weight_levels(cells, panel)

    icc_obs, unspec, a_icc, a_rest, cell_g = [], [], [], [], []
    groups, gidx = {}, []
    pop_g = defaultdict(float)
    for y in range(Y0, Y1 + 1):
        for s in ("1", "2"):
            for a in AGE_ORDER:
                groups[(y, s, a)] = len(groups)
    for (rid, y, s, a), vals in cells.items():
        if rid not in panel or not (Y0 <= y <= Y1):
            continue
        g = groups[(y, s, a)]
        pop_g[g] += pys.get((rid, y, s, a), 0.0)
        p = m.period_of(y)
        cont = panel[rid]["CI5_continent"]
        chain = [
            levels["rys_a"].get((rid, y, s, a)),
            levels["rps_a"].get((rid, p, s, a)),
            levels["rps"].get((rid, p, s)),
            levels["cps_a"].get((cont, p, s, a)),
            levels["gps_a"].get((p, s, a)),
            levels["gps"].get((p, s)),
        ]
        donor = None
        for tbl in chain:
            if not tbl:
                continue
            tot = sum(tbl.get(e, 0.0) for e in m.SPECIFIED)
            if tot >= m.MIN_CELL:
                donor = tbl
                break
        if donor is None:
            ai, ar = 1.0, float(len(m.SPECIFIED) - 1)
        else:
            ai = donor.get(m.ICC, 0.0)
            ar = sum(donor.get(e, 0.0) for e in m.SPECIFIED) - ai
        icc_obs.append(vals.get(m.ICC, 0.0))
        unspec.append(vals.get(m.UNSPEC_CA, 0.0) + vals.get(m.UNSPEC_MORPH, 0.0))
        a_icc.append(ai)
        a_rest.append(ar)
        cell_g.append(g)

    icc_obs = np.asarray(icc_obs, dtype=float)
    unspec = np.asarray(unspec, dtype=float)
    a_icc = np.asarray(a_icc, dtype=float) + 0.5
    a_rest = np.asarray(a_rest, dtype=float) + 0.5
    cell_g = np.asarray(cell_g)
    n_g = len(groups)
    pop_arr = np.array([pop_g.get(i, 0.0) for i in range(n_g)])
    std_w = np.zeros(n_g)
    for (y, s, a), i in groups.items():
        std_w[i] = STD_W[a]
    coef = np.where(pop_arr > 0, std_w / np.where(pop_arr > 0, pop_arr, 1) * 1e5, 0.0)
    ys_keys = [(y, s) for y in range(Y0, Y1 + 1) for s in ("1", "2")]
    ys_idx = {k: i for i, k in enumerate(ys_keys)}
    g_to_ys = np.array([ys_idx[(y, s)] for (y, s, a) in
                        sorted(groups, key=lambda k: groups[k])])
    w_pt = (a_icc - 0.5) / np.maximum(a_icc + a_rest - 1.0, 1e-9)

    def to_series(deaths):
        dg = np.bincount(cell_g, weights=deaths, minlength=n_g)
        return np.bincount(g_to_ys, weights=coef * dg, minlength=len(ys_keys))

    point_obs = to_series(icc_obs)
    point_corr = to_series(icc_obs + unspec * w_pt)
    print(f"bootstrap cells {len(cell_g)} draws {B}")

    draws_o = np.empty((B, len(ys_keys)))
    draws_c = np.empty((B, len(ys_keys)))
    for b in range(B):
        io = rng.poisson(icc_obs)
        us = rng.poisson(unspec)
        ww = rng.beta(a_icc, a_rest)
        draws_o[b] = to_series(io)
        draws_c[b] = to_series(io + us * ww)
        if (b + 1) % 200 == 0:
            print(f"  draw {b + 1}/{B}")

    years = np.array(range(Y0, Y1 + 1), dtype=float)
    ui = {"draws": B, "scenario": SCEN, "window": [Y0, Y1], "by_sex": {}}
    for s, lab in (("1", "Male"), ("2", "Female")):
        sel = np.array([i for i, (y, ss) in enumerate(ys_keys) if ss == s])
        d_o = np.array([eapc_point(years, draws_o[b, sel]) for b in range(B)])
        d_c = np.array([eapc_point(years, draws_c[b, sel]) for b in range(B)])
        diff = d_c - d_o
        # endpoint-to-endpoint rise and artefact on each draw
        i0, i1 = sel[0], sel[-1]
        rise_o = 100 * (draws_o[:, i1] / draws_o[:, i0] - 1)
        rise_c = 100 * (draws_c[:, i1] / draws_c[:, i0] - 1)
        art = 100 * (1 - rise_c / rise_o)
        art_e = 100 * (1 - d_c / d_o)
        lo_o, hi_o = np.nanpercentile(d_o, [2.5, 97.5])
        lo_c, hi_c = np.nanpercentile(d_c, [2.5, 97.5])
        lo_d, hi_d = np.nanpercentile(diff, [2.5, 97.5])
        lo_a, hi_a = np.nanpercentile(art, [2.5, 97.5])
        lo_ae, hi_ae = np.nanpercentile(art_e, [2.5, 97.5])
        po = eapc_point(years, point_obs[sel])
        pc = eapc_point(years, point_corr[sel])
        ro = 100 * (point_obs[i1] / point_obs[i0] - 1)
        rc = 100 * (point_corr[i1] / point_corr[i0] - 1)
        ui["by_sex"][lab] = {
            "eapc_obs": {"eapc": round(po, 3), "lo": round(float(lo_o), 3),
                         "hi": round(float(hi_o), 3)},
            "eapc_corr": {"eapc": round(pc, 3), "lo": round(float(lo_c), 3),
                          "hi": round(float(hi_c), 3)},
            "eapc_difference": {
                "diff": round(pc - po, 3),
                "lo": round(float(lo_d), 3), "hi": round(float(hi_d), 3),
                "excludes_zero": bool(lo_d > 0 or hi_d < 0),
            },
            "obs_rise_pct": {"point": round(ro, 1),
                             "lo": round(float(np.nanpercentile(rise_o, 2.5)), 1),
                             "hi": round(float(np.nanpercentile(rise_o, 97.5)), 1)},
            "corr_rise_pct": {"point": round(rc, 1),
                              "lo": round(float(np.nanpercentile(rise_c, 2.5)), 1),
                              "hi": round(float(np.nanpercentile(rise_c, 97.5)), 1)},
            "artefact_rise_pct": {
                "point": round(100 * (1 - rc / ro), 1),
                "lo": round(float(lo_a), 1), "hi": round(float(hi_a), 1),
            },
            "artefact_eapc_pct": {
                "point": round(100 * (1 - pc / po), 1),
                "lo": round(float(lo_ae), 1), "hi": round(float(hi_ae), 1),
            },
        }

    (TAB / "T_morph_eapc_UI.json").write_text(
        json.dumps(ui, ensure_ascii=False, indent=2), encoding="utf-8")

    eapc_rows = []
    for r in scen_eapc:
        eapc_rows.append(r)
    write_csv(TAB / "T_morph_eapc.csv", eapc_rows)

    jp_rows = []
    strat_rows = []
    for blk in strata:
        rec = {k: v for k, v in blk.items() if not k.startswith("_") and k != "joinpoint"}
        jp = blk.get("joinpoint") or {}
        rec["jp_obs"] = jp.get("obs", {}).get("joinpoints", "")
        rec["jp_years_obs"] = ";".join(str(t) for t in jp.get("obs", {}).get("joinpoint_years", []))
        rec["aapc_obs"] = jp.get("obs", {}).get("aapc", "")
        rec["jp_corr"] = jp.get("corr", {}).get("joinpoints", "")
        rec["jp_years_corr"] = ";".join(str(t) for t in jp.get("corr", {}).get("joinpoint_years", []))
        rec["aapc_corr"] = jp.get("corr", {}).get("aapc", "")
        strat_rows.append(rec)
        for ver in ("obs", "corr"):
            info = jp.get(ver)
            if not info:
                continue
            for seg in info["segments"]:
                jp_rows.append({
                    "stratum_group": blk["stratum_group"],
                    "stratum": blk["stratum"],
                    "sex": blk["sex"],
                    "n_registries": blk["n_registries"],
                    "version": "As coded" if ver == "obs" else "Corrected",
                    "joinpoints": info["joinpoints"],
                    "joinpoint_years": ";".join(str(t) for t in info["joinpoint_years"]),
                    "segment": seg["period"].replace("-", "\u2013"),
                    "apc": seg["apc"],
                    "apc_ci": fmt_ci(seg["apc"], seg["lo"], seg["hi"]),
                    "significant": "yes" if seg["significant"] else "no",
                    "aapc": info["aapc"],
                    "aapc_ci": fmt_ci(info["aapc"], info["aapc_lo"], info["aapc_hi"]),
                })
    write_csv(TAB / "T_morph_joinpoint.csv", jp_rows)
    write_csv(TAB / "T_morph_stratified.csv", strat_rows)

    summary = {
        "window": [Y0, Y1],
        "scenario": SCEN,
        "n_registries": len(by_id),
        "method": ("EAPC: unweighted OLS of ln(ASR) on year. "
                   "Joinpoint: death-weighted segmented log-linear; "
                   "permutation test (999) on the 73-registry panel, BIC3 elsewhere. "
                   f"Bootstrap: {B} draws, Poisson cases + Beta weights from the M1 donor stratum."),
        "headline": headline,
        "bootstrap": ui["by_sex"],
        "scenario_eapc": scen_eapc,
    }
    # drop joinpoint fit objects from headline
    for lab in list(summary["headline"]):
        summary["headline"][lab].pop("joinpoint", None)
    (TAB / "morph_trends_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ---- figures ----
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=False)
    for ax, (sx, slab) in zip(axes, (("1", "Male"), ("2", "Female"))):
        sub = [r for r in age_rows if r["sex"] == sx]
        years, obs, corr, n_o, n_c = series_from_age_rows(sub)
        blk = next(b for b in strata if b["stratum_group"] == "panel"
                   and b["sex"] == slab and b["stratum"] == "All 73 registries")
        jp = blk.get("_jp", {})
        ax.scatter(years, obs, s=12, color=C_OBS, alpha=0.75, zorder=3)
        ax.scatter(years, corr, s=12, color=C_COR, alpha=0.75, zorder=3)
        for ver, col, lab in (("obs", C_OBS, "As coded"), ("corr", C_COR, "Corrected")):
            info = jp.get(ver)
            if not info:
                continue
            fit = info["fit"]
            ax.plot(years, np.exp(fit["X"] @ fit["beta"]), color=col, lw=1.8, label=lab)
            for t in fit["jps"]:
                ax.axvline(t, color=col, lw=0.7, ls=":", alpha=0.8)
        ax.set_title(slab, fontsize=9)
        ax.set_xlabel("Year")
        ax.set_ylabel("ICC ASR(W) per 100 000")
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=7.5)
    figstyle.panel_labels(axes)
    figstyle.save(fig, "FigP2_asr_joinpoint", FIG)

    # stratified EAPC, both sexes
    want_groups = [
        ("continent", ["Africa", "Central and South America", "North America",
                       "Asia", "Europe", "Oceania"]),
        ("unspecified_share", bands),
        ("fluke", ["Liver-fluke endemic (4 Thai registries)",
                   "Asia excluding 4 Thai fluke registries"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.6), sharex=False)
    for ax, (grp, order) in zip(axes, want_groups):
        rows = [b for b in strata if b["stratum_group"] == grp and b["sex"] == "Both"
                and "eapc_obs" in b and b["n_registries"] >= 2]
        rows = [next((r for r in rows if r["stratum"] == name), None) for name in order]
        rows = [r for r in rows if r]
        y = np.arange(len(rows))
        ax.hlines(y + 0.12, [r["eapc_obs_lo"] for r in rows],
                  [r["eapc_obs_hi"] for r in rows], color=C_OBS, lw=1.4)
        ax.hlines(y - 0.12, [r["eapc_corr_lo"] for r in rows],
                  [r["eapc_corr_hi"] for r in rows], color=C_COR, lw=1.4)
        ax.scatter([r["eapc_obs"] for r in rows], y + 0.12, color=C_OBS, s=18, zorder=3,
                   label="As coded")
        ax.scatter([r["eapc_corr"] for r in rows], y - 0.12, color=C_COR, s=18, zorder=3,
                   label="Corrected")
        ax.axvline(0, color="0.6", lw=0.6)
        labels = []
        for r in rows:
            lab = r["stratum"]
            if lab.startswith("Liver-fluke"):
                lab = "Thai fluke (4)"
            elif lab.startswith("Asia excluding"):
                lab = "Asia, not fluke"
            labels.append(f"{lab} (n={r['n_registries']})")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("EAPC (% per year)")
        ax.invert_yaxis()
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    axes[0].set_title("Continent", fontsize=8.5)
    axes[1].set_title("Unspecified share", fontsize=8.5)
    axes[2].set_title("Liver fluke", fontsize=8.5)
    figstyle.panel_labels(axes, x=-0.12)
    figstyle.save(fig, "FigP2_strat_eapc", FIG)

    print(json.dumps({
        "headline": {k: {kk: vv for kk, vv in v.items() if kk not in ("joinpoint",)}
                     for k, v in headline.items()},
        "bootstrap": ui["by_sex"],
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
