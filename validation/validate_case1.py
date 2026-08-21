"""
validation/validate_case1.py
=============================
Reproduces the Case 1 IAS model validation from the paper.

Dataset : 23 dilute Nb engineering alloys (wt%)
Model   : Independent Alloying Shift (IAS)
          DBTT = -150 + 8·W + 15·Mo - 5·V - 2·Ti + 1·Zr + 0.5·Hf
Metrics : R² = 0.856, MAE = 15.9°C

Outputs
-------
  - Console table (alloy-by-alloy, sorted by absolute error)
  - Console summary (MAE, RMSE, R², within-band counts)
  - Parity plot saved to validation/figures/case1_parity.png
"""

import sys
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from urades.data import CASE1_DATA
from urades.core import predict_case1, IAS_COEFFS, IAS_BASELINE

# =============================================================================
# RUN PREDICTIONS
# =============================================================================

results = []
for name, W, Mo, V, Ti, Zr, Hf, exp_dbtt in CASE1_DATA:
    # IAS model operates in wt%. The dataset is already in wt%.
    comp_wt = {"W": W, "Mo": Mo, "V": V, "Ti": Ti, "Zr": Zr, "Hf": Hf,
               "Nb": 100 - W - Mo - V - Ti - Zr - Hf}
    # call predict_case1 via wt% path (pass as at% with direct wt% values —
    # here we call the formula directly since data IS wt%)
    pred_dbtt = IAS_BASELINE
    for el, k in IAS_COEFFS.items():
        pred_dbtt += k * comp_wt.get(el, 0)

    abs_err = abs(pred_dbtt - exp_dbtt)
    results.append({
        "name":     name,
        "exp":      exp_dbtt,
        "pred":     round(pred_dbtt, 1),
        "abs_err":  round(abs_err, 1),
    })

# =============================================================================
# METRICS
# =============================================================================

exp_vals  = np.array([r["exp"]  for r in results])
pred_vals = np.array([r["pred"] for r in results])
errors    = pred_vals - exp_vals
abs_errs  = np.abs(errors)

SS_res = np.sum((exp_vals - pred_vals) ** 2)
SS_tot = np.sum((exp_vals - exp_vals.mean()) ** 2)
R2     = 1 - SS_res / SS_tot
MAE    = abs_errs.mean()
RMSE   = math.sqrt(np.mean(errors ** 2))
Bias   = errors.mean()
n      = len(results)

# =============================================================================
# CONSOLE OUTPUT
# =============================================================================

print("=" * 70)
print("CASE 1 — IAS MODEL VALIDATION")
print("=" * 70)
print(f"  Model   : DBTT = {IAS_BASELINE} + 8·W + 15·Mo - 5·V - 2·Ti + 1·Zr + 0.5·Hf")
print(f"  Units   : wt%")
print(f"  Dataset : n = {n} dilute Nb engineering alloys")
print()

# Alloy-by-alloy table, sorted by absolute error descending
sorted_res = sorted(results, key=lambda x: x["abs_err"], reverse=True)
print(f"  {'Alloy':<22} {'Exp (°C)':>9} {'Pred (°C)':>10} {'|Error| (°C)':>13}")
print("  " + "-" * 57)
for r in sorted_res:
    flag = "  ← largest error" if r["abs_err"] == sorted_res[0]["abs_err"] else ""
    print(f"  {r['name']:<22} {r['exp']:>9.1f} {r['pred']:>10.1f} {r['abs_err']:>13.1f}{flag}")

print()
print("  Performance Summary")
print("  " + "-" * 40)
print(f"  n           = {n}")
print(f"  R²          = {R2:.4f}   (paper: 0.856)")
print(f"  MAE         = {MAE:.1f} °C   (paper: 15.9°C)")
print(f"  RMSE        = {RMSE:.1f} °C")
print(f"  Mean bias   = {Bias:+.1f} °C")
print()

for band in [15, 20, 30, 50]:
    count = int((abs_errs <= band).sum())
    print(f"  Within ±{band}°C : {count}/{n}  ({100*count/n:.0f}%)")

print("=" * 70)

# =============================================================================
# PARITY PLOT
# =============================================================================

fig = plt.figure(figsize=(13, 5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

# ── Left: parity plot ───────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])

lim_lo = min(exp_vals.min(), pred_vals.min()) - 20
lim_hi = max(exp_vals.max(), pred_vals.max()) + 20

ax1.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
         "k--", linewidth=1.4, label="Perfect agreement (1:1)", zorder=2)
ax1.fill_between([lim_lo, lim_hi],
                 [lim_lo - 30, lim_hi - 30],
                 [lim_lo + 30, lim_hi + 30],
                 alpha=0.08, color="steelblue", label="±30°C band")

ax1.scatter(exp_vals, pred_vals,
            s=70, color="#2b5c8f", edgecolors="black",
            linewidths=0.5, zorder=3)

# Annotate outliers (|error| > 30°C)
for r in results:
    if r["abs_err"] > 30:
        ax1.annotate(r["name"],
                     xy=(r["exp"], r["pred"]),
                     xytext=(r["exp"] + 3, r["pred"] - 10),
                     fontsize=7, color="#555555")

ax1.set_xlabel("Experimental DBTT (°C)", fontsize=11)
ax1.set_ylabel("Predicted DBTT (°C)",    fontsize=11)
ax1.set_title(f"Case 1 — IAS Model Parity Plot\n"
              f"R² = {R2:.3f}  |  MAE = {MAE:.1f}°C  |  n = {n}",
              fontsize=10, fontweight="bold")
ax1.set_xlim(lim_lo, lim_hi)
ax1.set_ylim(lim_lo, lim_hi)
ax1.set_aspect("equal")
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(True, linestyle=":", alpha=0.5)

# ── Right: absolute error bar chart ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])

names_sorted = [r["name"]  for r in sorted_res]
errs_sorted  = [r["abs_err"] for r in sorted_res]
colors       = ["#e74c3c" if e > 30 else "#2b5c8f" for e in errs_sorted]

bars = ax2.barh(names_sorted[::-1], errs_sorted[::-1],
                color=colors[::-1], edgecolor="black", linewidth=0.4, height=0.65)

ax2.axvline(MAE, color="black", linestyle="--", linewidth=1.2,
            label=f"MAE = {MAE:.1f}°C")
ax2.axvline(30,  color="#e74c3c", linestyle=":",  linewidth=1.0,
            label="30°C threshold")

ax2.set_xlabel("|Prediction Error| (°C)", fontsize=11)
ax2.set_title("Absolute Error by Alloy",  fontsize=10, fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(True, axis="x", linestyle=":", alpha=0.5)

plt.suptitle("URADES Case 1 — Independent Alloying Shift Model Validation",
             fontsize=11, fontweight="bold", y=1.01)

os.makedirs(os.path.join(os.path.dirname(__file__), "figures"), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), "figures", "case1_parity.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"\n  Figure saved → {out_path}")
plt.close()
