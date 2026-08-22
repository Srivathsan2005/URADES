"""
inverse_design/search.py
=========================
URADES inverse design engine — grid search over Nb-alloy composition space.

Given a set of property targets (DBTT ceiling, density ceiling, YS floor),
this script screens all compositions in a defined grid, applies the full
URADES pipeline (boundary check → GVI → model), and returns ranked candidates.

The search is case-aware: it covers both Case 1 (wt%, IAS model) and
Case 2 (at%, SR model) composition spaces independently, then merges results.

Default demo targets (propulsion-relevant)
------------------------------------------
  DBTT    ≤ -50°C      (ductile at subzero temperatures)
  Density ≤ 10.0 g/cc  (weight budget for nozzle/thruster hardware)
  YS      ≥ 400 MPa    (minimum structural load-bearing capacity)
  GVI     ≥ 0.5        (phase stability gate)

Outputs
-------
  - Console: ranked table of qualifying candidates
  - CSV: inverse_design/results/candidates.csv
  - Figure: DBTT vs density scatter of candidates, coloured by GVI
    saved to inverse_design/figures/inverse_design_map.png
"""

import sys
import os
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from urades.core import (
    run_URADES, identify_case, check_boundary_conditions,
    calc_GVI, predict_case1, predict_case2,
    atomic_to_weight, weight_to_atomic,
    GVI_THRESHOLD
)

# =============================================================================
# SEARCH TARGETS (modify these for different design objectives)
# =============================================================================

TARGETS = {
    "DBTT_max":    -50.0,    # °C  — upper bound on DBTT
    "density_max":  10.0,    # g/cc — upper bound on density
    "YS_min":      400.0,    # MPa  — lower bound on yield strength
    "GVI_min":     GVI_THRESHOLD,  # phase stability threshold
}

# =============================================================================
# GRID DEFINITION
# =============================================================================

# Case 2 grid — at%
# Nb is the balance; all other elements step in coarse increments
# for fast screening (~seconds). Fine-grained search can be done
# by reducing the step size.
C2_GRID = {
    "W":  range(0, 16, 2),     # 0,2,4,...,14  at%  (limit 15)
    "Mo": range(0, 18, 3),     # 0,3,6,...,15  at%  (limit 18)
    "Hf": range(0, 23, 3),     # 0,3,6,...,21  at%  (limit 22.4)
    "Zr": range(0,  9, 2),     # 0,2,4,...,8   at%  (limit 8.5)
    "Ti": range(0, 11, 2),     # 0,2,4,...,10  at%  (limit 10)
}

# Case 1 grid — wt%  (input as wt%, convert internally for routing)
C1_GRID = {
    "W":  range(0, 21, 3),     # 0,3,...,18  wt%  (limit 20)
    "Mo": range(0, 11, 2),     # 0,2,...,10  wt%  (limit 10)
    "Hf": range(0, 11, 2),     # 0,2,...,10  wt%  (limit 10)
    "Zr": range(0,  6, 1),     # 0,1,...,5   wt%  (limit 5)
    "Ti": range(0, 11, 2),     # 0,2,...,10  wt%  (limit 10)
}

# =============================================================================
# CASE 2 GRID SEARCH
# =============================================================================

def search_case2(targets):
    """Screen the Case 2 composition space (at%)."""
    candidates = []
    screened   = 0
    rejected_bc  = 0
    rejected_gvi = 0

    from itertools import product
    for W, Mo, Hf, Zr, Ti in product(
            C2_GRID["W"], C2_GRID["Mo"],
            C2_GRID["Hf"], C2_GRID["Zr"], C2_GRID["Ti"]):

        Nb = 100 - W - Mo - Hf - Zr - Ti
        if Nb < 50:        # Case 2 requires Nb ≥ 50 at%
            continue
        if Nb > 99.5:      # pure or near-pure Nb — no alloying
            continue

        Buffer = Hf + Zr + Ti
        if Buffer > 22.4:  # combined buffer limit
            continue

        screened += 1
        comp_at = {"Nb": Nb, "W": W, "Mo": Mo, "Hf": Hf, "Zr": Zr, "Ti": Ti}

        # Boundary conditions
        passed, _ = check_boundary_conditions(comp_at, case=2)
        if not passed:
            rejected_bc += 1
            continue

        # GVI
        gvi_data = calc_GVI(comp_at, case=2)
        gvi = gvi_data["GVI"]
        if gvi < targets["GVI_min"]:
            rejected_gvi += 1
            continue

        # Property prediction
        props = predict_case2(comp_at)
        dbtt    = props["DBTT"]
        density = props["density"]
        ys      = props["YS_MPa"]

        # Target filters
        if dbtt > targets["DBTT_max"]:
            continue
        if density > targets["density_max"]:
            continue
        if ys < targets["YS_min"]:
            continue

        candidates.append({
            "case":    2,
            "alloy":   f"Nb{Nb:.0f}-W{W}-Mo{Mo}-Hf{Hf}-Zr{Zr}-Ti{Ti}",
            "Nb": Nb, "W": W, "Mo": Mo, "Hf": Hf, "Zr": Zr, "Ti": Ti,
            "DBTT":    dbtt,
            "density": density,
            "YS_MPa":  ys,
            "GVI":     round(gvi, 4),
            "SR":      props["SR"],
        })

    return candidates, screened, rejected_bc, rejected_gvi


# =============================================================================
# CASE 1 GRID SEARCH
# =============================================================================

def search_case1(targets):
    """Screen the Case 1 composition space (wt%)."""
    from itertools import product
    from urades.core import IAS_BASELINE, IAS_COEFFS, _calc_density, _calc_Tm_ROM

    candidates  = []
    screened    = 0
    rejected_bc = 0
    rejected_gvi = 0

    for W, Mo, Hf, Zr, Ti in product(
            C1_GRID["W"], C1_GRID["Mo"],
            C1_GRID["Hf"], C1_GRID["Zr"], C1_GRID["Ti"]):

        Nb = 100 - W - Mo - Hf - Zr - Ti
        if Nb < 79:        # Case 1 requires Nb ≥ 79 wt%
            continue
        if Nb > 99.5:
            continue

        screened += 1
        comp_wt = {"Nb": Nb, "W": W, "Mo": Mo, "Hf": Hf, "Zr": Zr, "Ti": Ti}

        # Boundary conditions (pass comp as at% — core converts internally)
        comp_at = weight_to_atomic(comp_wt)
        passed, _ = check_boundary_conditions(comp_at, case=1)
        if not passed:
            rejected_bc += 1
            continue

        # GVI (Case 1)
        gvi_data = calc_GVI(comp_at, case=1)
        gvi = gvi_data["GVI"]
        if gvi < targets["GVI_min"]:
            rejected_gvi += 1
            continue

        # IAS prediction (wt%)
        dbtt = IAS_BASELINE
        for el, k in IAS_COEFFS.items():
            dbtt += k * comp_wt.get(el, 0)

        density = _calc_density(comp_at)

        # YS (Case 1 empirical)
        ys = (150.0
              + 15.0 * W + 25.0 * Mo
              + 15.0 * Zr + 8.0 * Hf + 5.0 * Ti)

        if dbtt > targets["DBTT_max"]:
            continue
        if density > targets["density_max"]:
            continue
        if ys < targets["YS_min"]:
            continue

        candidates.append({
            "case":    1,
            "alloy":   f"Nb{Nb:.0f}wt-W{W}-Mo{Mo}-Hf{Hf}-Zr{Zr}-Ti{Ti}",
            "Nb": Nb, "W": W, "Mo": Mo, "Hf": Hf, "Zr": Zr, "Ti": Ti,
            "DBTT":    round(dbtt, 1),
            "density": round(density, 3),
            "YS_MPa":  round(ys, 0),
            "GVI":     round(gvi, 4),
            "SR":      None,
        })

    return candidates, screened, rejected_bc, rejected_gvi

# =============================================================================
# RUN SEARCH
# =============================================================================

print("=" * 70)
print("URADES INVERSE DESIGN — GRID SEARCH")
print("=" * 70)
print()
print("  Design targets:")
for k, v in TARGETS.items():
    print(f"    {k:<15} : {v}")
print()

print("  Searching Case 2 (Nb-matrix RCCAs, at%) ...")
c2_cands, c2_screened, c2_rbc, c2_rgvi = search_case2(TARGETS)
print(f"    Screened   : {c2_screened:>6}")
print(f"    BC rejected: {c2_rbc:>6}")
print(f"    GVI flagged: {c2_rgvi:>6}")
print(f"    Qualified  : {len(c2_cands):>6}")
print()

print("  Searching Case 1 (dilute Nb alloys, wt%) ...")
c1_cands, c1_screened, c1_rbc, c1_rgvi = search_case1(TARGETS)
print(f"    Screened   : {c1_screened:>6}")
print(f"    BC rejected: {c1_rbc:>6}")
print(f"    GVI flagged: {c1_rgvi:>6}")
print(f"    Qualified  : {len(c1_cands):>6}")
print()

# Merge and sort by DBTT (lowest first = most ductile)
all_cands = c2_cands + c1_cands
all_cands.sort(key=lambda x: x["DBTT"])

print(f"  Total qualified candidates: {len(all_cands)}")
print()

# =============================================================================
# KNOWN BENCHMARK ALLOYS (for reference in output)
# =============================================================================

benchmarks = {
    "C-103 (Case 1)":  {"DBTT": -147.0, "density": 8.86, "YS_MPa": 155.0, "GVI": 0.999},
    "Cb-752 (Case 1)": {"DBTT":  -67.5, "density": 9.12, "YS_MPa": 188.0, "GVI": 0.985},
    "V1 (paper alloy)":{"DBTT":  -97.0, "density": 9.20, "YS_MPa": 298.0, "GVI": 0.920},
}

# =============================================================================
# CONSOLE TABLE
# =============================================================================

top_n = min(20, len(all_cands))
print(f"  Top {top_n} candidates (sorted by DBTT, most ductile first):")
print()
print(f"  {'#':<4} {'Case':<6} {'Alloy':<35} {'DBTT':>7} {'ρ(g/cc)':>8} "
      f"{'YS(MPa)':>8} {'GVI':>7}")
print("  " + "-" * 80)

for i, c in enumerate(all_cands[:top_n]):
    unit = "at%" if c["case"] == 2 else "wt%"
    print(f"  {i+1:<4} {unit:<6} {c['alloy']:<35} {c['DBTT']:>7.1f} "
          f"{c['density']:>8.3f} {c['YS_MPa']:>8.0f} {c['GVI']:>7.4f}")

print()
print("  Reference benchmarks:")
for name, b in benchmarks.items():
    print(f"  {'':4} {'':6} {name:<35} {b['DBTT']:>7.1f} "
          f"{b['density']:>8.3f} {b['YS_MPa']:>8.1f} {b['GVI']:>7.4f}")

# =============================================================================
# EXPORT CSV
# =============================================================================

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
csv_path = os.path.join(os.path.dirname(__file__), "results", "candidates.csv")

fieldnames = ["rank", "case", "alloy", "Nb", "W", "Mo", "Hf", "Zr", "Ti",
              "DBTT", "density", "YS_MPa", "GVI", "SR"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i, c in enumerate(all_cands):
        row = {"rank": i + 1, **c}
        writer.writerow(row)

print(f"\n  Full candidate list saved → {csv_path}")

# =============================================================================
# FIGURE
# =============================================================================

if len(all_cands) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.38)

    dbtt_vals    = np.array([c["DBTT"]    for c in all_cands])
    density_vals = np.array([c["density"] for c in all_cands])
    gvi_vals     = np.array([c["GVI"]     for c in all_cands])
    ys_vals      = np.array([c["YS_MPa"]  for c in all_cands])
    case_vals    = np.array([c["case"]    for c in all_cands])

    # ── Left: DBTT vs density, coloured by GVI ──────────────────────────────
    ax1 = axes[0]
    sc = ax1.scatter(dbtt_vals, density_vals,
                     c=gvi_vals, cmap="RdYlGn",
                     vmin=GVI_THRESHOLD, vmax=1.0,
                     s=25, alpha=0.7, edgecolors="none", zorder=3)
    plt.colorbar(sc, ax=ax1, label="GVI Score")

    # Overlay benchmark alloys
    bench_colors = {"C-103 (Case 1)": "blue", "Cb-752 (Case 1)": "navy",
                    "V1 (paper alloy)": "purple"}
    for name, b in benchmarks.items():
        ax1.scatter(b["DBTT"], b["density"], marker="*", s=180,
                    color=bench_colors[name], edgecolors="black",
                    linewidths=0.5, zorder=5, label=name)

    ax1.axvline(TARGETS["DBTT_max"], color="red", linestyle="--",
                linewidth=1.2, label=f"DBTT target ({TARGETS['DBTT_max']}°C)")
    ax1.axhline(TARGETS["density_max"], color="orange", linestyle="--",
                linewidth=1.2, label=f"ρ target ({TARGETS['density_max']} g/cc)")

    ax1.set_xlabel("Predicted DBTT (°C)", fontsize=11)
    ax1.set_ylabel("Density (g/cc)",      fontsize=11)
    ax1.set_title(f"Inverse Design Map\n"
                  f"n={len(all_cands)} qualifying candidates",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=7.5, loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.4)

    # ── Right: DBTT vs YS, coloured by case ─────────────────────────────────
    ax2 = axes[1]
    c2_mask = case_vals == 2
    c1_mask = case_vals == 1

    if c2_mask.any():
        ax2.scatter(dbtt_vals[c2_mask], ys_vals[c2_mask],
                    color="#2b3a8f", s=25, alpha=0.7,
                    edgecolors="none", label="Case 2 (SR model, at%)", zorder=3)
    if c1_mask.any():
        ax2.scatter(dbtt_vals[c1_mask], ys_vals[c1_mask],
                    color="#e74c3c", s=25, alpha=0.7,
                    edgecolors="none", label="Case 1 (IAS model, wt%)", zorder=3)

    for name, b in benchmarks.items():
        ax2.scatter(b["DBTT"], b["YS_MPa"], marker="*", s=180,
                    color=bench_colors[name], edgecolors="black",
                    linewidths=0.5, zorder=5, label=name)

    ax2.axvline(TARGETS["DBTT_max"], color="red", linestyle="--",
                linewidth=1.2, label=f"DBTT target")
    ax2.axhline(TARGETS["YS_min"], color="green", linestyle="--",
                linewidth=1.2, label=f"YS target ({TARGETS['YS_min']:.0f} MPa)")

    ax2.set_xlabel("Predicted DBTT (°C)", fontsize=11)
    ax2.set_ylabel("Predicted YS (MPa)",  fontsize=11)
    ax2.set_title("DBTT vs Yield Strength\nby Model Family",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=7.5, loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.4)

    plt.suptitle("URADES Inverse Design — Composition Space Screening",
                 fontsize=11, fontweight="bold", y=1.01)

    os.makedirs(os.path.join(os.path.dirname(__file__), "figures"), exist_ok=True)
    fig_path = os.path.join(os.path.dirname(__file__), "figures", "inverse_design_map.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"  Figure saved → {fig_path}")
    plt.close()
else:
    print("  No candidates found — consider relaxing targets.")
