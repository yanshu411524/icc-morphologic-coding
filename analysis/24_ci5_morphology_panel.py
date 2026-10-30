#!/usr/bin/env python3
"""Extract a morphology-resolved liver cancer panel from CI5plus Detailed.

CI5plus Detailed resolves liver cancer (C22) into morphological subtypes,
including cholangiocarcinoma, while gallbladder/biliary (C23-24) is only
available as a single aggregate. The analytic question therefore concerns
ascertainment of *intrahepatic* cholangiocarcinoma within liver cancer.

Morphology codes (cancer_dict.csv):
  31 Liver, all morphologies (total; equals the sum of 32-40)
  32 Hepatocellular carcinoma
  33 Cholangiocarcinoma
  34 Other specified carcinoma
  35 Unspecified carcinoma          <- partly unascertained ICC
  36 Hepatoblastoma
  37 Hemangiosarcoma
  38 Other specified sarcoma
  39 Other specified morphology
  40 Unspecified morphology         <- partly unascertained ICC
  41 Gallbladder etc. (C23-24, no morphology detail; carried for context)

Age codes are 1 = 0-4, 2 = 5-9, ... 18 = 85+, 19 = age unknown. Cases with
unknown age are retained in a separate column rather than silently pooled
into the oldest group, because unknown age correlates with poor ascertainment
and would otherwise bias the age-specific analysis.

Outputs (data/processed/ci5/):
  morph_panel.csv            registry-year-sex-age x morphology cases + person-years
  morph_registry_coverage.csv  per-registry year span and completeness
  morph_panel_summary.json
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/ci5/_extracted/CI5plus_Detailed"
OUT = ROOT / "data/processed/ci5"

# morphology code -> short endpoint name
MORPH = {
    32: "hcc",
    33: "icc",
    34: "other_spec_ca",
    35: "unspec_ca",
    36: "hepatoblastoma",
    37: "hemangiosarcoma",
    38: "other_sarcoma",
    39: "other_spec_morph",
    40: "unspec_morph",
}
LIVER_TOTAL = 31
GALLBLADDER = 41
WANTED = set(MORPH) | {LIVER_TOTAL, GALLBLADDER}

# CI5 age code -> label; 19 is "age unknown"
AGE_LABELS = {i: f"{(i - 1) * 5}-{(i - 1) * 5 + 4}" for i in range(1, 18)}
AGE_LABELS[18] = "85+"
AGE_UNKNOWN = 19

SEX_LABELS = {"1": "Male", "2": "Female"}


def load_registries() -> dict[str, dict[str, str]]:
    reg = {}
    with open(RAW / "id_dict.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            reg[r["id_code"]] = r
    return reg


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    registries = load_registries()

    # (id_code, year, sex, age_code) -> {endpoint: cases}
    cells: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    # (id_code, year, sex, age_code) -> person-years
    pys: dict[tuple, float] = {}
    # consistency check: reported liver total vs summed morphologies
    total_reported = 0.0
    total_summed = 0.0

    with open(RAW / "data.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            code = int(r["cancer_code"])
            if code not in WANTED:
                continue
            key = (r["id_code"], int(r["year"]), r["sex"], int(r["age"]))
            cases = float(r["cases"] or 0)
            py = float(r["py"] or 0)
            # person-years are replicated across cancer codes; keep the largest
            # non-zero value seen for the cell
            if py > pys.get(key, 0.0):
                pys[key] = py
            if code == LIVER_TOTAL:
                cells[key]["liver_total_reported"] += cases
                total_reported += cases
            elif code == GALLBLADDER:
                cells[key]["gallbladder_biliary"] += cases
            else:
                cells[key][MORPH[code]] += cases
                total_summed += cases

    endpoints = list(MORPH.values())
    fields = (["id_code", "registry", "country_code", "continent", "year",
               "sex", "sex_label", "age_code", "age_group", "person_years"]
              + endpoints + ["liver_total_reported", "liver_total_summed",
                             "gallbladder_biliary"])

    rows = []
    for key in sorted(cells):
        id_code, year, sex, age_code = key
        meta = registries.get(id_code, {})
        vals = cells[key]
        summed = sum(vals.get(e, 0.0) for e in endpoints)
        rec = {
            "id_code": id_code,
            "registry": meta.get("id_label", ""),
            "country_code": meta.get("country_code", ""),
            "continent": meta.get("CI5_continent", ""),
            "year": year,
            "sex": sex,
            "sex_label": SEX_LABELS.get(sex, sex),
            "age_code": age_code,
            "age_group": AGE_LABELS.get(age_code, "unknown"),
            "person_years": round(pys.get(key, 0.0), 1),
        }
        for e in endpoints:
            rec[e] = vals.get(e, 0.0)
        rec["liver_total_reported"] = vals.get("liver_total_reported", 0.0)
        rec["liver_total_summed"] = summed
        rec["gallbladder_biliary"] = vals.get("gallbladder_biliary", 0.0)
        rows.append(rec)

    with open(OUT / "morph_panel.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---- per-registry coverage, used later to define the analytic panel ----
    span: dict[str, set[int]] = defaultdict(set)
    liver_by_reg: dict[str, float] = defaultdict(float)
    unspec_by_reg: dict[str, float] = defaultdict(float)
    unknown_age_by_reg: dict[str, float] = defaultdict(float)
    for rec in rows:
        rid = rec["id_code"]
        liver = rec["liver_total_summed"]
        if liver > 0:
            span[rid].add(rec["year"])
        liver_by_reg[rid] += liver
        unspec_by_reg[rid] += rec["unspec_morph"] + rec["unspec_ca"]
        if rec["age_code"] == AGE_UNKNOWN:
            unknown_age_by_reg[rid] += liver

    cov_rows = []
    for rid, years in sorted(span.items(), key=lambda kv: -len(kv[1])):
        meta = registries.get(rid, {})
        y0, y1 = min(years), max(years)
        liver = liver_by_reg[rid] or 1.0
        cov_rows.append({
            "id_code": rid,
            "registry": meta.get("id_label", ""),
            "country_code": meta.get("country_code", ""),
            "continent": meta.get("CI5_continent", ""),
            "year_min": y0,
            "year_max": y1,
            "n_years": len(years),
            "span_complete": int(len(years) == (y1 - y0 + 1)),
            "liver_cases": round(liver_by_reg[rid], 0),
            "unspecified_share": round(unspec_by_reg[rid] / liver, 4),
            "unknown_age_share": round(unknown_age_by_reg[rid] / liver, 4),
        })
    with open(OUT / "morph_registry_coverage.csv", "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cov_rows[0].keys()))
        w.writeheader()
        w.writerows(cov_rows)

    summary = {
        "n_registries": len(span),
        "n_rows": len(rows),
        "year_min": min(r["year"] for r in rows),
        "year_max": max(r["year"] for r in rows),
        "liver_cases_reported_total": round(total_reported, 0),
        "liver_cases_summed_morphology": round(total_summed, 0),
        "reported_vs_summed_ratio": round(total_summed / total_reported, 6)
        if total_reported else None,
        "morphology_totals": {
            e: round(sum(r[e] for r in rows), 0) for e in endpoints
        },
        "unknown_age_liver_cases": round(
            sum(r["liver_total_summed"] for r in rows
                if r["age_code"] == AGE_UNKNOWN), 0),
    }
    (OUT / "morph_panel_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
