#!/usr/bin/env python3
"""Redistribute morphologically unspecified liver cancers across subtypes.

Motivation
----------
Registries differ enormously in how often liver cancer is assigned a specific
morphology: the unspecified share ranges from 2% (Hamburg) to 80% (Khon Kaen,
the highest-incidence cholangiocarcinoma population in the world). Because the
unspecified share has fallen steadily over time, a purely artefactual rise in
cholangiocarcinoma (ICC) is expected even with a constant true incidence. This
script quantifies how much of the observed rise survives correction.

Estimand
--------
Cases coded as "unspecified carcinoma" or "unspecified morphology" are
reallocated to specific morphologies. Proportional reallocation within a
stratum cannot change that stratum's ICC:HCC ratio, but it does change the
*trend*, because the unspecified pool shrinks over time and therefore inflates
early years more than late ones. That is the mechanism under test.

Scenarios
  M1_proportional    both unspecified pools -> all specified morphologies,
                     proportional to observed specified cases in the stratum
  M2_carcinoma_split unspecified carcinoma -> carcinomas only;
                     unspecified morphology -> all specified morphologies
  M3_reference       weights taken from high-specificity registries
                     (unspecified share < REF_MAX_UNSPEC) matched on
                     continent-period-sex-age; tests whether ICC is
                     *disproportionately* hidden in the unspecified pool
  M4_max_icc         all unspecified -> ICC (upper bound on ICC)
  M5_max_hcc         all unspecified -> HCC (lower bound on ICC)

Weight hierarchy for M1/M2 (falls back until the stratum has >= MIN_CELL
specified cases):
  registry-year-sex-age -> registry-period-sex-age -> registry-period-sex
  -> continent-period-sex-age -> global-period-sex-age -> global-period-sex

Outputs (data/processed/ci5/):
  morph_panel_registries.csv   the de-duplicated analytic panel
  morph_corrected_age.csv      registry-year-sex-age x scenario, ICC/HCC obs+corr
  morph_corrected_asr.csv      registry-year-sex x scenario, ASR(W) obs+corr
  morph_pooled_series.csv      panel-pooled year-sex x scenario, ASR(W) obs+corr
  morph_redistribution_summary.json
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI5 = ROOT / "data/processed/ci5"
RAW = ROOT / "data/raw/ci5/_extracted/CI5plus_Detailed"

# ---- analytic window and panel rules -------------------------------------
Y0 = int(os.environ.get("CI5_Y0", 1998))
Y1 = int(os.environ.get("CI5_Y1", 2017))
# Which USA registries to keep. NPCR (45 states + DC) and SEER 9 are largely
# complementary but overlap in four metropolitan areas, and Los Angeles sits
# inside California which NPCR covers. Keeping one avoids double counting.
USA_MODE = os.environ.get("USA_MODE", "npcr")  # npcr | seer
MIN_CELL = float(os.environ.get("MIN_CELL", 20))
REF_MAX_UNSPEC = float(os.environ.get("REF_MAX_UNSPEC", 0.10))
PERIOD_LEN = 5

USA_NPCR = "USA, NPCR (45 States and Washington DC)"
USA_SEER9 = "USA, SEER (9 registries)"
USA_LA = "USA, California, Los Angeles (SEER)"

# ---- morphology groups ----------------------------------------------------
HCC, ICC = "hcc", "icc"
CARCINOMAS = [HCC, ICC, "other_spec_ca"]
NON_CARCINOMA = ["hepatoblastoma", "hemangiosarcoma", "other_sarcoma",
                 "other_spec_morph"]
SPECIFIED = CARCINOMAS + NON_CARCINOMA
UNSPEC_CA, UNSPEC_MORPH = "unspec_ca", "unspec_morph"
UNSPECIFIED = [UNSPEC_CA, UNSPEC_MORPH]

SCENARIOS = ["M1_proportional", "M2_carcinoma_split", "M3_reference",
             "M4_max_icc", "M5_max_hcc"]

# Segi world standard population, CI5 18 age groups, per 100 000
SEGI = {
    "0-4": 12000, "5-9": 10000, "10-14": 9000, "15-19": 9000, "20-24": 8000,
    "25-29": 8000, "30-34": 6000, "35-39": 6000, "40-44": 6000, "45-49": 6000,
    "50-54": 5000, "55-59": 4000, "60-64": 4000, "65-69": 3000, "70-74": 2000,
    "75-79": 1000, "80-84": 500, "85+": 500,
}
_TOT = sum(SEGI.values())
STD_W = {k: v / _TOT for k, v in SEGI.items()}
AGE_ORDER = list(SEGI)

CONTINENT = {"1": "Africa", "2": "Central and South America",
             "3": "North America", "4": "Asia", "5": "Europe", "6": "Oceania"}


def period_of(year: int) -> str:
    start = (year // PERIOD_LEN) * PERIOD_LEN
    return f"{start}-{start + PERIOD_LEN - 1}"


# ---------------------------------------------------------------- panel ----
def build_panel() -> dict[str, dict]:
    """Return {id_code: registry metadata} for the de-duplicated analytic panel."""
    meta = {r["id_code"]: r for r in
            csv.DictReader(open(RAW / "id_dict.csv", encoding="utf-8"))}
    cov = {r["id_code"]: r for r in
           csv.DictReader(open(CI5 / "morph_registry_coverage.csv",
                               encoding="utf-8"))}

    keep = {}
    for rid, c in cov.items():
        m = meta.get(rid)
        if not m or m.get("ethnic_code") != "99":
            continue  # ethnic subsets duplicate their parent registry
        if not (int(c["year_min"]) <= Y0 and int(c["year_max"]) >= Y1
                and int(c["n_years"]) >= (Y1 - Y0 + 1)):
            continue
        keep[rid] = {**m, **c}

    # drop subnational registries when the whole country is also present
    national_countries = {v["country_code"] for v in keep.values()
                          if "," not in v["id_label"]}
    for rid in list(keep):
        v = keep[rid]
        if "," in v["id_label"] and v["country_code"] in national_countries:
            del keep[rid]

    # resolve the USA overlap
    usa_drop = ({USA_SEER9, USA_LA} if USA_MODE == "npcr"
                else {USA_NPCR, USA_LA})
    for rid in list(keep):
        if keep[rid]["id_label"] in usa_drop:
            del keep[rid]
    return keep


def load_cells(panel: dict) -> tuple[dict, dict]:
    """Return (cells, pys) keyed by (id_code, year, sex, age_group)."""
    cells: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    pys: dict[tuple, float] = {}
    with open(CI5 / "morph_panel.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rid, year = r["id_code"], int(r["year"])
            if rid not in panel or not (Y0 <= year <= Y1):
                continue
            age = r["age_group"]
            if age == "unknown":
                continue  # 254 cases in total; excluded from age-standardisation
            key = (rid, year, r["sex"], age)
            for e in SPECIFIED + UNSPECIFIED:
                cells[key][e] += float(r[e])
            pys[key] = float(r["person_years"])
    return cells, pys


# -------------------------------------------------------------- weights ----
def build_weight_levels(cells: dict, panel: dict) -> dict:
    levels = {k: defaultdict(lambda: defaultdict(float)) for k in
              ("rys_a", "rps_a", "rps", "cps_a", "gps_a", "gps")}
    for (rid, y, s, a), vals in cells.items():
        p = period_of(y)
        cont = panel[rid]["CI5_continent"]
        for e in SPECIFIED:
            v = vals.get(e, 0.0)
            levels["rys_a"][(rid, y, s, a)][e] += v
            levels["rps_a"][(rid, p, s, a)][e] += v
            levels["rps"][(rid, p, s)][e] += v
            levels["cps_a"][(cont, p, s, a)][e] += v
            levels["gps_a"][(p, s, a)][e] += v
            levels["gps"][(p, s)][e] += v
    return levels


def build_reference_levels(cells: dict, panel: dict) -> dict:
    """Weight tables restricted to registries with high morphological specificity."""
    ref = {k: defaultdict(lambda: defaultdict(float)) for k in
           ("cps_a", "gps_a", "gps")}
    hq = {rid for rid, m in panel.items()
          if float(m["unspecified_share"]) < REF_MAX_UNSPEC}
    for (rid, y, s, a), vals in cells.items():
        if rid not in hq:
            continue
        p = period_of(y)
        cont = panel[rid]["CI5_continent"]
        for e in SPECIFIED:
            v = vals.get(e, 0.0)
            ref["cps_a"][(cont, p, s, a)][e] += v
            ref["gps_a"][(p, s, a)][e] += v
            ref["gps"][(p, s)][e] += v
    return ref, hq


def weights_from(chain, recipients) -> tuple[dict[str, float], str]:
    for tag, tbl in chain:
        if not tbl:
            continue
        tot = sum(tbl.get(e, 0.0) for e in recipients)
        if tot >= MIN_CELL:
            return {e: tbl.get(e, 0.0) / tot for e in recipients}, tag
    n = len(recipients)
    return {e: 1.0 / n for e in recipients}, "equal_split"


def asr_of(deaths_by_age: dict, pop_by_age: dict) -> float | None:
    total = used = 0.0
    for a in AGE_ORDER:
        p = pop_by_age.get(a, 0.0)
        if p <= 0:
            continue
        total += STD_W[a] * deaths_by_age.get(a, 0.0) / p
        used += STD_W[a]
    if used < 0.99:
        return None
    return total / used * 1e5


# ------------------------------------------------------------------ main ----
def main() -> None:
    panel = build_panel()
    cells, pys = load_cells(panel)
    levels = build_weight_levels(cells, panel)
    ref_levels, hq_registries = build_reference_levels(cells, panel)

    level_use: dict[str, int] = defaultdict(int)
    # corrected[(scen, rid, y, s)][age][morph]
    corrected: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    observed: dict = defaultdict(lambda: defaultdict(float))

    for (rid, y, s, a), vals in cells.items():
        p = period_of(y)
        cont = panel[rid]["CI5_continent"]
        u_ca = vals.get(UNSPEC_CA, 0.0)
        u_mo = vals.get(UNSPEC_MORPH, 0.0)

        for e in SPECIFIED + UNSPECIFIED:
            observed[(rid, y, s)][(a, e)] += vals.get(e, 0.0)

        chain_all = [
            ("registry_year_sex_age", levels["rys_a"].get((rid, y, s, a))),
            ("registry_period_sex_age", levels["rps_a"].get((rid, p, s, a))),
            ("registry_period_sex", levels["rps"].get((rid, p, s))),
            ("continent_period_sex_age", levels["cps_a"].get((cont, p, s, a))),
            ("global_period_sex_age", levels["gps_a"].get((p, s, a))),
            ("global_period_sex", levels["gps"].get((p, s))),
        ]
        w_all, tag_all = weights_from(chain_all, SPECIFIED)
        w_ca, _ = weights_from(chain_all, CARCINOMAS)
        level_use[tag_all] += 1

        chain_ref = [
            ("ref_continent_period_sex_age", ref_levels["cps_a"].get((cont, p, s, a))),
            ("ref_global_period_sex_age", ref_levels["gps_a"].get((p, s, a))),
            ("ref_global_period_sex", ref_levels["gps"].get((p, s))),
        ]
        w_ref, _ = weights_from(chain_ref, SPECIFIED)

        for scen in SCENARIOS:
            new = {e: vals.get(e, 0.0) for e in SPECIFIED}
            if scen == "M1_proportional":
                for e, w in w_all.items():
                    new[e] += (u_ca + u_mo) * w
            elif scen == "M2_carcinoma_split":
                for e, w in w_ca.items():
                    new[e] += u_ca * w
                for e, w in w_all.items():
                    new[e] += u_mo * w
            elif scen == "M3_reference":
                for e, w in w_ref.items():
                    new[e] += (u_ca + u_mo) * w
            elif scen == "M4_max_icc":
                new[ICC] += u_ca + u_mo
            elif scen == "M5_max_hcc":
                new[HCC] += u_ca + u_mo
            for e, v in new.items():
                corrected[(scen, rid, y, s)][a][e] = v

    # ------------------------------------------------------ age-level out ----
    age_rows = []
    for (scen, rid, y, s), by_age in corrected.items():
        for a, morphs in by_age.items():
            obs = observed[(rid, y, s)]
            age_rows.append({
                "scenario": scen, "id_code": rid,
                "registry": panel[rid]["id_label"],
                "continent": CONTINENT.get(panel[rid]["CI5_continent"], ""),
                "year": y, "sex": s, "age_group": a,
                "person_years": round(pys.get((rid, y, s, a), 0.0), 1),
                "icc_obs": round(obs.get((a, ICC), 0.0), 4),
                "icc_corr": round(morphs.get(ICC, 0.0), 4),
                "hcc_obs": round(obs.get((a, HCC), 0.0), 4),
                "hcc_corr": round(morphs.get(HCC, 0.0), 4),
                "unspecified_obs": round(obs.get((a, UNSPEC_CA), 0.0)
                                         + obs.get((a, UNSPEC_MORPH), 0.0), 4),
            })
    age_rows.sort(key=lambda r: (r["scenario"], r["id_code"], r["year"],
                                 r["sex"], AGE_ORDER.index(r["age_group"])))
    _write(CI5 / "morph_corrected_age.csv", age_rows)

    # ---------------------------------------------------- registry ASR out ----
    asr_rows = []
    for (scen, rid, y, s), by_age in corrected.items():
        pop = {a: pys.get((rid, y, s, a), 0.0) for a in AGE_ORDER}
        obs = observed[(rid, y, s)]
        icc_o = {a: obs.get((a, ICC), 0.0) for a in AGE_ORDER}
        hcc_o = {a: obs.get((a, HCC), 0.0) for a in AGE_ORDER}
        icc_c = {a: by_age.get(a, {}).get(ICC, 0.0) for a in AGE_ORDER}
        hcc_c = {a: by_age.get(a, {}).get(HCC, 0.0) for a in AGE_ORDER}
        a_icc_o, a_icc_c = asr_of(icc_o, pop), asr_of(icc_c, pop)
        if a_icc_c is None:
            continue
        n_icc_o, n_icc_c = sum(icc_o.values()), sum(icc_c.values())
        n_hcc_o, n_hcc_c = sum(hcc_o.values()), sum(hcc_c.values())
        n_uns = sum(obs.get((a, e), 0.0) for a in AGE_ORDER for e in UNSPECIFIED)
        n_liver = sum(obs.get((a, e), 0.0) for a in AGE_ORDER
                      for e in SPECIFIED + UNSPECIFIED)
        asr_rows.append({
            "scenario": scen, "id_code": rid, "registry": panel[rid]["id_label"],
            "continent": CONTINENT.get(panel[rid]["CI5_continent"], ""),
            "country_code": panel[rid]["country_code"],
            "year": y, "sex": s,
            "high_specificity": int(rid in hq_registries),
            "liver_cases": round(n_liver, 1),
            "unspecified_cases": round(n_uns, 1),
            "unspecified_share": round(n_uns / n_liver, 5) if n_liver else "",
            "icc_cases_obs": round(n_icc_o, 1), "icc_cases_corr": round(n_icc_c, 1),
            "hcc_cases_obs": round(n_hcc_o, 1), "hcc_cases_corr": round(n_hcc_c, 1),
            "icc_asr_obs": round(a_icc_o, 5) if a_icc_o is not None else "",
            "icc_asr_corr": round(a_icc_c, 5),
            "icc_share_specified_obs": round(n_icc_o / (n_icc_o + n_hcc_o), 5)
            if (n_icc_o + n_hcc_o) else "",
            "icc_share_specified_corr": round(n_icc_c / (n_icc_c + n_hcc_c), 5)
            if (n_icc_c + n_hcc_c) else "",
            "uplift": round(n_icc_c / n_icc_o, 4) if n_icc_o > 0 else "",
        })
    asr_rows.sort(key=lambda r: (r["scenario"], r["id_code"], r["year"], r["sex"]))
    _write(CI5 / "morph_corrected_asr.csv", asr_rows)

    # ------------------------------------------------------ pooled series ----
    pooled = []
    for scen in SCENARIOS:
        for y in range(Y0, Y1 + 1):
            for s in ("1", "2"):
                pop = defaultdict(float)
                io = defaultdict(float); ic = defaultdict(float)
                ho = defaultdict(float); hc = defaultdict(float)
                un = defaultdict(float); liv = defaultdict(float)
                for rid in panel:
                    by_age = corrected.get((scen, rid, y, s))
                    if not by_age:
                        continue
                    obs = observed[(rid, y, s)]
                    for a in AGE_ORDER:
                        pop[a] += pys.get((rid, y, s, a), 0.0)
                        io[a] += obs.get((a, ICC), 0.0)
                        ho[a] += obs.get((a, HCC), 0.0)
                        ic[a] += by_age.get(a, {}).get(ICC, 0.0)
                        hc[a] += by_age.get(a, {}).get(HCC, 0.0)
                        for e in UNSPECIFIED:
                            un[a] += obs.get((a, e), 0.0)
                        for e in SPECIFIED + UNSPECIFIED:
                            liv[a] += obs.get((a, e), 0.0)
                n_io, n_ic = sum(io.values()), sum(ic.values())
                n_ho, n_hc = sum(ho.values()), sum(hc.values())
                n_un, n_liv = sum(un.values()), sum(liv.values())
                pooled.append({
                    "scenario": scen, "year": y, "sex": s,
                    "sex_label": "Male" if s == "1" else "Female",
                    "n_registries": len(panel),
                    "person_years": round(sum(pop.values()), 0),
                    "liver_cases": round(n_liv, 0),
                    "unspecified_cases": round(n_un, 0),
                    "unspecified_share": round(n_un / n_liv, 5) if n_liv else "",
                    "icc_cases_obs": round(n_io, 0), "icc_cases_corr": round(n_ic, 0),
                    "hcc_cases_obs": round(n_ho, 0), "hcc_cases_corr": round(n_hc, 0),
                    "icc_asr_obs": round(asr_of(io, pop) or 0, 5),
                    "icc_asr_corr": round(asr_of(ic, pop) or 0, 5),
                    "hcc_asr_obs": round(asr_of(ho, pop) or 0, 5),
                    "hcc_asr_corr": round(asr_of(hc, pop) or 0, 5),
                    "icc_share_liver_obs": round(n_io / n_liv, 5) if n_liv else "",
                    "icc_share_specified_obs": round(n_io / (n_io + n_ho), 5)
                    if (n_io + n_ho) else "",
                    "icc_share_specified_corr": round(n_ic / (n_ic + n_hc), 5)
                    if (n_ic + n_hc) else "",
                })
    _write(CI5 / "morph_pooled_series.csv", pooled)

    _write(CI5 / "morph_panel_registries.csv", [
        {"id_code": rid, "registry": m["id_label"],
         "country_code": m["country_code"],
         "continent": CONTINENT.get(m["CI5_continent"], ""),
         "liver_cases": m["liver_cases"],
         "unspecified_share": m["unspecified_share"],
         "high_specificity": int(rid in hq_registries)}
        for rid, m in sorted(panel.items(),
                             key=lambda kv: -float(kv[1]["liver_cases"]))])

    # ---------------------------------------------------------- summary ----
    def pick(scen, y, s):
        for r in pooled:
            if r["scenario"] == scen and r["year"] == y and r["sex"] == s:
                return r
        return None

    headline = {}
    for s, lab in (("1", "male"), ("2", "female")):
        a, b = pick("M1_proportional", Y0, s), pick("M1_proportional", Y1, s)
        if not (a and b):
            continue
        headline[lab] = {
            "unspecified_share_start": a["unspecified_share"],
            "unspecified_share_end": b["unspecified_share"],
            "icc_asr_obs_start": a["icc_asr_obs"], "icc_asr_obs_end": b["icc_asr_obs"],
            "icc_asr_corr_start": a["icc_asr_corr"], "icc_asr_corr_end": b["icc_asr_corr"],
            "observed_rise_pct": round(100 * (b["icc_asr_obs"] / a["icc_asr_obs"] - 1), 1)
            if a["icc_asr_obs"] else None,
            "corrected_rise_pct": round(100 * (b["icc_asr_corr"] / a["icc_asr_corr"] - 1), 1)
            if a["icc_asr_corr"] else None,
        }
        if headline[lab]["observed_rise_pct"]:
            headline[lab]["share_of_rise_explained_pct"] = round(
                100 * (1 - headline[lab]["corrected_rise_pct"]
                       / headline[lab]["observed_rise_pct"]), 1)

    summary = {
        "window": [Y0, Y1],
        "usa_mode": USA_MODE,
        "min_cell": MIN_CELL,
        "reference_max_unspecified": REF_MAX_UNSPEC,
        "n_registries": len(panel),
        "n_high_specificity_registries": len(hq_registries),
        "weight_level_usage": dict(level_use),
        "headline_M1_proportional": headline,
    }
    (CI5 / "morph_redistribution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise SystemExit(f"no rows to write for {path}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
