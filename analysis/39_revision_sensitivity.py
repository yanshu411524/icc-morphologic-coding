#!/usr/bin/env python3
"""Revision sensitivity: how much of the unspecified-morphology pool to reallocate.

Unspecified carcinoma is always reallocated with the M1 weights. A fraction f
of unspecified morphology is reallocated with the same weights and the rest is
left out of every subtype numerator.

  carcinoma_only   f = 0
  dco_discounted   f = share of SEER 17 unspecified morphology that is not
                   death-certificate only (Beta uncertainty from SEER counts)
  M1_full          f = 1 (reproduces the primary model)

For each scenario: pooled ASR by sex, endpoint-rise artefact, EAPC artefact,
1000-draw bootstrap UI (Poisson counts, Beta donor weights, Beta f), and
joinpoint with the permutation test used for the primary series.

Output: tables/revision_sensitivity.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
TAB = ROOT / "tables"


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m25 = load("morph25", "25_morphology_redistribution.py")
m26 = load("morph26", "26_morph_trends.py")

Y0, Y1, B, SEED = m26.Y0, m26.Y1, m26.B, m26.SEED


def main():
    rng = np.random.default_rng(SEED)
    dco = json.loads((TAB / "revision_seer_dco.json").read_text(encoding="utf-8"))
    um17 = dco["SEER 17"]["unspec_morph"]
    um8 = dco["SEER 8"]["unspec_morph"]
    f17 = 1 - um17["dco"] / um17["n"]
    f8 = 1 - um8["dco"] / um8["n"]
    beta_f = (um17["n"] - um17["dco"] + 0.5, um17["dco"] + 0.5)

    panel = m25.build_panel()
    cells, pys = m25.load_cells(panel)
    levels = m25.build_weight_levels(cells, panel)

    groups = {}
    for y in range(Y0, Y1 + 1):
        for s in ("1", "2"):
            for a in m26.AGE_ORDER:
                groups[(y, s, a)] = len(groups)
    pop_g = defaultdict(float)
    icc, uca, umo, a_icc, a_rest, cell_g = [], [], [], [], [], []
    for (rid, y, s, a), vals in cells.items():
        g = groups[(y, s, a)]
        pop_g[g] += pys.get((rid, y, s, a), 0.0)
        p = m25.period_of(y)
        cont = panel[rid]["CI5_continent"]
        chain = [levels["rys_a"].get((rid, y, s, a)),
                 levels["rps_a"].get((rid, p, s, a)),
                 levels["rps"].get((rid, p, s)),
                 levels["cps_a"].get((cont, p, s, a)),
                 levels["gps_a"].get((p, s, a)),
                 levels["gps"].get((p, s))]
        donor = next((t for t in chain if t and
                      sum(t.get(e, 0.0) for e in m25.SPECIFIED) >= m25.MIN_CELL), None)
        if donor is None:
            ai, ar = 1.0, float(len(m25.SPECIFIED) - 1)
        else:
            ai = donor.get(m25.ICC, 0.0)
            ar = sum(donor.get(e, 0.0) for e in m25.SPECIFIED) - ai
        icc.append(vals.get(m25.ICC, 0.0))
        uca.append(vals.get(m25.UNSPEC_CA, 0.0))
        umo.append(vals.get(m25.UNSPEC_MORPH, 0.0))
        a_icc.append(ai)
        a_rest.append(ar)
        cell_g.append(g)

    icc, uca, umo = (np.asarray(v, dtype=float) for v in (icc, uca, umo))
    a_icc = np.asarray(a_icc) + 0.5
    a_rest = np.asarray(a_rest) + 0.5
    cell_g = np.asarray(cell_g)
    n_g = len(groups)
    pop = np.array([pop_g.get(i, 0.0) for i in range(n_g)])
    std_w = np.zeros(n_g)
    for (y, s, a), i in groups.items():
        std_w[i] = m26.STD_W[a]
    coef = np.where(pop > 0, std_w / np.where(pop > 0, pop, 1) * 1e5, 0.0)
    ys_keys = [(y, s) for y in range(Y0, Y1 + 1) for s in ("1", "2")]
    ys_idx = {k: i for i, k in enumerate(ys_keys)}
    g_to_ys = np.array([ys_idx[(y, s)] for (y, s, a) in sorted(groups, key=groups.get)])
    w_pt = (a_icc - 0.5) / np.maximum(a_icc + a_rest - 1.0, 1e-9)

    def to_series(cases):
        dg = np.bincount(cell_g, weights=cases, minlength=n_g)
        asr = np.bincount(g_to_ys, weights=coef * dg, minlength=len(ys_keys))
        cnt = np.bincount(g_to_ys, weights=dg, minlength=len(ys_keys))
        return asr, cnt

    scenarios = {"carcinoma_only": 0.0, "dco_discounted": f17, "M1_full": 1.0}
    years = np.arange(Y0, Y1 + 1, dtype=float)
    sel = {s: np.array([i for i, (y, ss) in enumerate(ys_keys) if ss == s])
           for s in ("1", "2")}

    point = {}
    for name, f in scenarios.items():
        point[name] = to_series(icc + (uca + f * umo) * w_pt)
    obs_asr, obs_cnt = to_series(icc)

    draws = {name: np.empty((B, len(ys_keys))) for name in scenarios}
    draws_o = np.empty((B, len(ys_keys)))
    for b in range(B):
        io = rng.poisson(icc)
        ca = rng.poisson(uca)
        mo = rng.poisson(umo)
        ww = rng.beta(a_icc, a_rest)
        fb = rng.beta(*beta_f)
        draws_o[b] = to_series(io)[0]
        for name, f in scenarios.items():
            ff = fb if name == "dco_discounted" else f
            draws[name][b] = to_series(io + (ca + ff * mo) * ww)[0]

    def q(v):
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        return round(float(lo), 1), round(float(hi), 1)

    out = {"f_seer17": round(f17, 4), "f_seer8": round(f8, 4), "draws": B,
           "scenarios": {}}
    for name, f in scenarios.items():
        res = {"f": round(f, 4)}
        asr_c, cnt_c = point[name]
        for s, lab in (("1", "Male"), ("2", "Female")):
            ix = sel[s]
            i0, i1 = ix[0], ix[-1]
            ro = 100 * (obs_asr[i1] / obs_asr[i0] - 1)
            rc = 100 * (asr_c[i1] / asr_c[i0] - 1)
            po = m26.eapc_point(years, obs_asr[ix])
            pc = m26.eapc_point(years, asr_c[ix])
            d_o, d_c = draws_o[:, ix], draws[name][:, ix]
            rise_c = 100 * (d_c[:, -1] / d_c[:, 0] - 1)
            rise_o = 100 * (d_o[:, -1] / d_o[:, 0] - 1)
            e_o = np.array([m26.eapc_point(years, r) for r in d_o])
            e_c = np.array([m26.eapc_point(years, r) for r in d_c])
            rec = {
                "asr_corr_start": round(float(asr_c[i0]), 3),
                "asr_corr_end": round(float(asr_c[i1]), 3),
                "corr_rise_pct": round(rc, 1), "corr_rise_ui": q(rise_c),
                "artefact_rise_pct": round(100 * (1 - rc / ro), 1),
                "artefact_rise_ui": q(100 * (1 - rise_c / rise_o)),
                "eapc_corr": round(pc, 2),
                "eapc_corr_ui": tuple(round(float(v), 2) for v in
                                      np.nanpercentile(e_c, [2.5, 97.5])),
                "artefact_eapc_pct": round(100 * (1 - pc / po), 1),
                "artefact_eapc_ui": q(100 * (1 - e_c / e_o)),
            }
            ylog = np.log(asr_c[ix])
            w = np.maximum(cnt_c[ix], 1.0)
            k, fit = m26.select_model(years, ylog, w, "permutation", rng)
            segs, aapc = m26.segment_stats(years, ylog, w, fit)
            rec["joinpoint"] = {
                "joinpoint_years": [int(t) for t in fit["jps"]],
                "segments": [{"period": f"{int(sg['start'])}-{int(sg['end'])}",
                              "apc": round(sg["apc"], 2), "lo": round(sg["lo"], 2),
                              "hi": round(sg["hi"], 2),
                              "significant": bool(sg["significant"])} for sg in segs],
                "aapc": round(aapc["aapc"], 2), "aapc_lo": round(aapc["lo"], 2),
                "aapc_hi": round(aapc["hi"], 2),
            }
            res[lab] = rec
        out["scenarios"][name] = res

    # point artefact across f, for the text
    grid = {}
    for f in (0.0, 0.25, 0.5, f8, f17, 0.75, 1.0):
        asr_c, _ = to_series(icc + (uca + f * umo) * w_pt)
        row = {}
        for s, lab in (("1", "Male"), ("2", "Female")):
            ix = sel[s]
            ro = obs_asr[ix[-1]] / obs_asr[ix[0]] - 1
            rc = asr_c[ix[-1]] / asr_c[ix[0]] - 1
            row[lab] = round(100 * (1 - rc / ro), 1)
        grid[f"{f:.3f}"] = row
    out["artefact_by_f"] = grid

    (TAB / "revision_sensitivity.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
