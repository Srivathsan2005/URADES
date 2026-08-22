"""
validation/validate_case3.py
=============================
Reproduces the Case 3 EI classifier validation from the paper.

Dataset : 25 Nb-based RHEAs (at%)
Model   : Embrittlement Index (EI) four-zone classifier
          EI = (W + 0.48·Mo) / (Hf+Zr+Ti + 1)
          Zones: Ductile (<0.10) | Transition (<0.50) | Brittle (<15.5) | Confirmed Brittle
Metrics : LOO accuracy = 84%  (21/25)
          alpha = 0.48  (fitted; Mo contributes at 48% of W's weight)

Note on alpha
-------------
alpha=0.48 for Case 3 vs alpha=0 for Case 2 is the central novel result
of URADES. The full alpha sweep is in analysis/alpha_sensitivity.py.

Outputs
-------
  - Console table (all 25 alloys, mismatches flagged)
  - Confusion matrix (console)
  - EI distribution plot saved to validation/figures/case3_ei.png
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from urades.data import CASE3_DATA, DUCTILE, TRANSITION, BRITTLE, CONFIRMED
from urades.core import predict_case3, EI_ALPHA, EI_DUCTILE, EI_TRANSITION, EI_BRITTLE

ZONES       = [DUCTILE, TRANSITION, BRITTLE, CONFIRMED]
ZONE_ABBREV = {DUCTILE: "D", TRANSITION: "T", BRITTLE: "B", CONFIRMED: "CB"}
ZONE_COLOR  = {
    DUCTILE:    "#2ecc71",
    TRANSITION: "#f39c12",
    BRITTLE:    "#e67e22",
    CONFIRMED:  "#e74c3c",
}

# =============================================================================
# RUN PREDICTIONS
# =============================================================================

results = []
for name, comp, exp_zone in CASE3_DATA:
    r = predict_case3(comp)
    correct = (r["zone"] == exp_zone)
    results.append({
        "name":      name,
        "EI":        r["EI"],
        "pred_zone": r["zone"],
        "exp_zone":  exp_zone,
        "correct":   correct,
    })

n_correct = sum(r["correct"] for r in results)
n_total   = len(results)
accuracy  = 100 * n_correct / n_total

# =============================================================================
# CONSOLE OUTPUT
# =============================================================================

print("=" * 80)
print("CASE 3 — EI CLASSIFIER VALIDATION (LOO)")
print("=" * 80)
print(f"  Model   : EI = (W + {EI_ALPHA}·Mo) / (Hf+Zr+Ti+1)   [alpha = {EI_ALPHA}]")
print(f"  Zones   : Ductile (<{EI_DUCTILE}) | Transition (<{EI_TRANSITION}) "
      f"| Brittle (<{EI_BRITTLE}) | Confirmed Brittle")
print(f"  Units   : at%")
print(f"  Dataset : n = {n_total}")
print()

print(f"  {'No.':<4} {'Alloy':<35} {'EI':>7}  "
      f"{'Exp Zone':<22} {'Pred Zone':<22} {'Match'}")
print("  " + "-" * 97)

for i, r in enumerate(results):
    match = "✓" if r["correct"] else "✗  ← mismatch"
    print(f"  {i+1:<4} {r['name']:<35} {r['EI']:>7.4f}  "
          f"{r['exp_zone']:<22} {r['pred_zone']:<22} {match}")

print()
print(f"  Accuracy : {n_correct}/{n_total} = {accuracy:.1f}%   (paper: 84.0% = 21/25)")
print()

# Confusion matrix
print("  Confusion Matrix (rows = Experimental, cols = Predicted)")
print()
header = f"  {'':>22}" + "".join(f"  {ZONE_ABBREV[z]:>4}" for z in ZONES)
print(header)
print("  " + "-" * (22 + 8 * len(ZONES)))

for exp_z in ZONES:
    row = f"  {exp_z:<22}"
    for pred_z in ZONES:
        count = sum(1 for r in results
                    if r["exp_zone"] == exp_z and r["pred_zone"] == pred_z)
        row += f"  {count:>4}"
    print(row)

print()
print(f"  Zone abbreviations: D=Ductile, T=Transition, B=Brittle, CB=Confirmed Brittle")
print()

# Per-zone accuracy
print("  Per-Zone Accuracy")
print("  " + "-" * 40)
for zone in ZONES:
    total_in_zone   = sum(1 for r in results if r["exp_zone"] == zone)
    correct_in_zone = sum(1 for r in results if r["exp_zone"] == zone and r["correct"])
    if total_in_zone > 0:
        pct = 100 * correct_in_zone / total_in_zone
        print(f"  {zone:<25}  {correct_in_zone}/{total_in_zone}  ({pct:.0f}%)")

print()
print("  Known mismatches (discussed in paper):")
for r in results:
    if not r["correct"]:
        print(f"  • {r['name']}: experimental={r['exp_zone']}, "
              f"predicted={r['pred_zone']}, EI={r['EI']:.4f}")

print("=" * 80)

# =============================================================================
# EI DISTRIBUTION FIGURE
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.subplots_adjust(wspace=0.38)

# ── Left: EI values for all alloys, coloured by predicted zone ──────────────
ax1 = axes[0]

ei_vals  = np.array([r["EI"]        for r in results])
names    = [r["name"]               for r in results]
colors   = [ZONE_COLOR[r["pred_zone"]] for r in results]
markers  = ["o" if r["correct"] else "X" for r in results]

# Sort by EI for a clean waterfall
order = np.argsort(ei_vals)

# Plot on log scale because EI spans 0 → 37
ei_plot = ei_vals[order]
ei_plot_safe = np.where(ei_plot == 0, 1e-4, ei_plot)   # avoid log(0)

for idx, orig_idx in enumerate(order):
    r = results[orig_idx]
    ax1.scatter(idx, ei_plot_safe[list(order).index(orig_idx)],
                color=ZONE_COLOR[r["pred_zone"]],
                marker="o" if r["correct"] else "X",
                s=80, edgecolors="black", linewidths=0.5, zorder=3)

# Zone boundary lines
for threshold, label in [(EI_DUCTILE, f"EI={EI_DUCTILE}"),
                          (EI_TRANSITION, f"EI={EI_TRANSITION}"),
                          (EI_BRITTLE, f"EI={EI_BRITTLE}")]:
    ax1.axhline(threshold, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.text(n_total - 0.5, threshold * 1.05, label, fontsize=7.5,
             color="gray", ha="right")

ax1.set_yscale("log")
ax1.set_xlabel("Alloy (sorted by EI)", fontsize=11)
ax1.set_ylabel("Embrittlement Index (EI, log scale)", fontsize=11)
ax1.set_title(f"EI Values — Case 3 Dataset\n"
              f"EI = (W + {EI_ALPHA}·Mo) / (Hf+Zr+Ti+1)",
              fontsize=10, fontweight="bold")
ax1.set_xlim(-0.5, n_total - 0.5)
ax1.grid(True, linestyle=":", alpha=0.4, which="both")

# Legend
patches = [mpatches.Patch(color=ZONE_COLOR[z], label=z) for z in ZONES]
correct_m = plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="gray",
                        markeredgecolor="black", markersize=8, label="Correct")
wrong_m   = plt.Line2D([0],[0], marker="X", color="w", markerfacecolor="gray",
                        markeredgecolor="black", markersize=8, label="Mismatch")
ax1.legend(handles=patches + [correct_m, wrong_m], fontsize=7.5, loc="lower right")

# ── Right: zone classification summary ──────────────────────────────────────
ax2 = axes[1]

zone_counts_exp  = [sum(1 for r in results if r["exp_zone"]  == z) for z in ZONES]
zone_counts_pred = [sum(1 for r in results if r["pred_zone"] == z) for z in ZONES]

x      = np.arange(len(ZONES))
width  = 0.35

bars1 = ax2.bar(x - width/2, zone_counts_exp,  width,
                color=[ZONE_COLOR[z] for z in ZONES],
                edgecolor="black", linewidth=0.5,
                alpha=0.9, label="Experimental")
bars2 = ax2.bar(x + width/2, zone_counts_pred, width,
                color=[ZONE_COLOR[z] for z in ZONES],
                edgecolor="black", linewidth=0.5,
                alpha=0.5, label="Predicted", hatch="//")

ax2.set_xticks(x)
ax2.set_xticklabels([ZONE_ABBREV[z] for z in ZONES])
ax2.set_ylabel("Number of alloys", fontsize=11)
ax2.set_title(f"Zone Distribution\nAccuracy = {n_correct}/{n_total} = {accuracy:.0f}%",
              fontsize=10, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
ax2.set_xlabel("Zone  (D=Ductile, T=Transition, B=Brittle, CB=Confirmed Brittle)",
               fontsize=9)

plt.suptitle("URADES Case 3 — Embrittlement Index Classifier Validation",
             fontsize=11, fontweight="bold", y=1.01)

os.makedirs(os.path.join(os.path.dirname(__file__), "figures"), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), "figures", "case3_ei.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"\n  Figure saved → {out_path}")
plt.close()
