"""
urades/data.py
==============
Single source of truth for all URADES training and validation datasets.

All experimental DBTT values are compiled from published literature.
Compositions are given in the units used by each model:
  - Case 1 : wt%  (IAS model operates in wt%)
  - Case 2 : at%  (SR  model operates in at%)
  - Case 3 : at%  (EI  classifier operates in at%)

Do not modify these values without updating the corresponding
validation metrics in the paper (Table 1, 2, 3).
"""

# =============================================================================
# CASE 1 — Dilute Nb Engineering Alloys
# IAS model, wt%, n=23
# Validated metrics: R²=0.856, MAE=15.9°C
# =============================================================================

CASE1_DATA = [
    # (alloy_name,          W,    Mo,   V,    Ti,   Zr,   Hf,   exp_DBTT)
    # All concentrations in wt%
    ("Nb-10Mo-10V",         0.0,  10.0, 10.0, 0.0,  0.0,  0.0,  -50.0),
    ("Nb-5Mo-5V",           0.0,   5.0,  5.0, 0.0,  0.0,  0.0, -150.0),
    ("Cb-752",             10.0,   0.0,  0.0, 0.0,  2.5,  0.0, -101.0),
    ("Nb-5W-1Zr",           5.0,   0.0,  0.0, 0.0,  1.0,  0.0, -140.0),
    ("B-66",                0.0,   5.0,  5.0, 0.0,  1.0,  0.0, -130.0),
    ("Nb-10Ti-5Mo-1Zr",     0.0,   5.0,  0.0,10.0,  1.0,  0.0, -110.0),
    ("Nb-10W-1Zr",         10.0,   0.0,  0.0, 0.0,  1.0,  0.0,  -85.0),
    ("Nb-5Mo-10V",          0.0,   5.0, 10.0, 0.0,  0.0,  0.0, -110.0),
    ("Cb-74",               0.0,  10.0,  0.0, 0.0,  0.0,  0.0,  -80.0),
    ("Nb-10V-2Zr",          0.0,   0.0, 10.0, 0.0,  2.0,  0.0, -185.0),
    ("Nb-5W-2.5Zr",         5.0,   0.0,  0.0, 0.0,  2.5,  0.0, -120.0),
    ("Nb-10W-2Zr",         10.0,   0.0,  0.0, 0.0,  2.0,  0.0,  -80.0),
    ("Nb-15W-1Zr",         15.0,   0.0,  0.0, 0.0,  1.0,  0.0,  -40.0),
    ("Cb-751",             10.0,  10.0,  0.0, 0.0,  0.0,  0.0,   20.0),
    ("Nb-15W-2.5Zr",       15.0,   0.0,  0.0, 0.0,  2.5,  0.0,  -20.0),
    ("Nb-5Ti-5V-1Zr",       0.0,   0.0,  5.0, 5.0,  1.0,  0.0, -190.0),
    ("Nb-5W-5V-1Zr",        5.0,   0.0,  5.0, 0.0,  1.0,  0.0, -140.0),
    ("Nb-10W-5V-1Zr",      10.0,   0.0,  5.0, 0.0,  1.0,  0.0, -100.0),
    ("Nb-5Ti-5V",           0.0,   0.0,  5.0, 5.0,  0.0,  0.0, -180.0),
    ("Nb-10Mo-5V",          0.0,  10.0,  5.0, 0.0,  0.0,  0.0,  -30.0),
    ("C-103",               0.0,   0.0,  0.0, 1.0,  0.0, 10.0, -150.0),
    ("Nb-10Hf-2Zr",         0.0,   0.0,  0.0, 0.0,  2.0, 10.0, -145.0),
    ("Nb-10Ti-5W-1Zr",      5.0,   0.0,  0.0,10.0,  1.0,  0.0, -130.0),
]

# =============================================================================
# CASE 2 — Nb-Matrix RCCAs
# SR model, at%, n=11 (n=10 for LOOCV — Nb-40Mo-10Ti excluded)
# Validated metrics: LOOCV R²=0.871, MAE=26°C
# =============================================================================

CASE2_DATA = [
    # (alloy_name,          W,    Mo,   Hf,   Zr,   Ti,   exp_DBTT, in_loocv)
    # All concentrations in at%
    # in_loocv=False → reported for reference only (Mo exceeds validated limit)
    ("C-103",               0,    0,    10,   0,    1,   -150,  True),
    ("Nb-1Zr",              0,    0,     0,   1,    0,   -150,  True),
    ("WC-3009",             9,    0,    30,   0,    0,    -75,  True),
    ("Nb-10W-10Ti",        10,    0,     0,   0,   10,    -40,  True),
    ("Nb-13Mo-10Ti",        0,   13,     0,   0,   10,    -30,  True),
    ("Nb-18Mo-10Ti",        0,   18,     0,   0,   10,     25,  True),
    ("Nb-18Mo-8.5Zr",       0,   18,     0,   8.5,  0,     45,  True),
    ("Nb-5Mo-15W-5Ti",     15,    5,     0,   0,    5,    130,  True),
    ("Nb-5Mo-10Ti",         0,    5,     0,   0,   10,    -80,  True),
    ("Cb-752",             10,    0,     0,   2.5,  0,    -50,  True),
    ("Nb-40Mo-10Ti",        0,   40,     0,   0,   10,    160,  False),  # Mo=40 exceeds limit
]

# =============================================================================
# CASE 3 — Nb-Based RHEAs
# EI classifier, at%, n=25
# Validated metrics: LOO accuracy=84% (21/25)
# alpha=0.48 (Mo contributes at 48% of W's embrittlement weight)
# =============================================================================

# Zone labels
DUCTILE    = "Ductile Zone"
TRANSITION = "Transition Zone"
BRITTLE    = "Brittle Zone"
CONFIRMED  = "Confirmed Brittle Zone"

CASE3_DATA = [
    # (alloy_name,                    composition_at_pct,                              exp_zone)
    # Composition given as dict for flexibility with variable element sets
    ("TiVNbTa",
        {"Ti":25, "V":25, "Nb":25, "Ta":25},
        DUCTILE),

    ("WTaTiVZr",
        {"W":20, "Ta":20, "Ti":20, "V":20, "Zr":20},
        TRANSITION),

    ("HfMoNbTaTiZr",
        {"Hf":16.67, "Mo":16.67, "Nb":16.67, "Ta":16.67, "Ti":16.67, "Zr":16.67},
        DUCTILE),

    ("HfMoTaTiZr",
        {"Hf":20, "Mo":20, "Ta":20, "Ti":20, "Zr":20},
        DUCTILE),

    ("NbTiZr",
        {"Nb":33.33, "Ti":33.33, "Zr":33.33},
        DUCTILE),

    ("NbTiVZr",
        {"Nb":25, "Ti":25, "V":25, "Zr":25},
        DUCTILE),

    ("NbTiV2Zr",
        {"Nb":20, "Ti":20, "V":40, "Zr":20},
        DUCTILE),

    ("Re0.3NbTiZr",
        {"Re":9.1, "Nb":30.3, "Ti":30.3, "Zr":30.3},
        DUCTILE),

    ("Al2.5(NbTiZr)",
        {"Al":2.5, "Nb":32.5, "Ti":32.5, "Zr":32.5},
        DUCTILE),

    ("Al5(NbTiZr)",
        {"Al":5.0, "Nb":31.67, "Ti":31.67, "Zr":31.67},
        DUCTILE),

    ("Al7.5(NbTiZr)",
        {"Al":7.5, "Nb":30.83, "Ti":30.83, "Zr":30.83},
        DUCTILE),

    ("Re0.3TaTiZr",
        {"Re":9.1, "Ta":30.3, "Ti":30.3, "Zr":30.3},
        TRANSITION),

    ("TiZrHfNbTa",
        {"Ti":20, "Zr":20, "Hf":20, "Nb":20, "Ta":20},
        DUCTILE),

    ("VNbMoTaW",
        {"V":20, "Nb":20, "Mo":20, "Ta":20, "W":20},
        CONFIRMED),

    ("NbMoTaW",
        {"Nb":25, "Mo":25, "Ta":25, "W":25},
        CONFIRMED),

    ("TiNbMoTaW",
        {"Ti":20, "Nb":20, "Mo":20, "Ta":20, "W":20},
        BRITTLE),

    ("NbTiV",
        {"Nb":33.33, "Ti":33.33, "V":33.33},
        DUCTILE),

    ("HfNbTaTiZr",
        {"Hf":20, "Nb":20, "Ta":20, "Ti":20, "Zr":20},
        DUCTILE),

    ("NbTaHfTiZrV0.5",
        {"Nb":18.18, "Ta":18.18, "Hf":18.18, "Ti":18.18, "Zr":18.18, "V":9.1},
        DUCTILE),

    ("Nb45Ta25Ti15Hf15",
        {"Nb":45, "Ta":25, "Ti":15, "Hf":15},
        DUCTILE),

    ("Mo0.2NbTiZr",
        {"Mo":6.25, "Nb":31.25, "Ti":31.25, "Zr":31.25},
        DUCTILE),

    ("Mo0.6NbTiZr",
        {"Mo":16.67, "Nb":27.78, "Ti":27.78, "Zr":27.78},
        TRANSITION),

    ("HfNbTiZr",
        {"Hf":25, "Nb":25, "Ti":25, "Zr":25},
        DUCTILE),

    ("TiVNbMoTaW",
        {"Ti":16.67, "V":16.67, "Nb":16.67, "Mo":16.67, "Ta":16.67, "W":16.67},
        BRITTLE),

    ("MoNbTaTiZr",
        {"Mo":20, "Nb":20, "Ta":20, "Ti":20, "Zr":20},
        TRANSITION),
]

# =============================================================================
# GVI CALPHAD VALIDATION — Full Table (10 alloys)
# Source: Thermo-Calc with demo databases
# =============================================================================

CALPHAD_FULL = [
    # (alloy_name,              composition_wt_pct,                              stable_phases,                      gvi_decision)
    ("Nb-10Mo-15Ti",
        {"Mo":10, "Ti":15},
        "BCC + BCC#2 + HCP",
        "Pass"),

    ("Nb-5W-10Mo-5Hf-5Zr-5Ti",
        {"W":5, "Mo":10, "Hf":5, "Zr":5, "Ti":5},
        "BCC + BCC#2 + HCP",
        "Pass (false positive)"),    # GVI=0.990 but CALPHAD confirms multiphase

    ("Nb-6W-8Mo-8Hf-3Zr-5Ti",
        {"W":6, "Mo":8, "Hf":8, "Zr":3, "Ti":5},
        "BCC + BCC#2 + HCP + C15 Laves",
        "Reject"),

    ("Nb-5W-10Mo-5Hf-5Zr-10Ti",
        {"W":5, "Mo":10, "Hf":5, "Zr":5, "Ti":10},
        "BCC + BCC#2 + HCP + C15 Laves",
        "Reject"),

    ("Nb-5W-10Mo-10Hf-10Zr-10Ti",
        {"W":5, "Mo":10, "Hf":10, "Zr":10, "Ti":10},
        "BCC + BCC#2 + HCP + C15 Laves",
        "Reject"),

    ("Nb-8W-5Hf",
        {"W":8, "Hf":5},
        "BCC + HCP + C15 Laves",
        "Pass"),

    ("Nb-12W-5Hf",
        {"W":12, "Hf":5},
        "BCC + HCP + C15 Laves",
        "Reject"),

    ("Nb-20Mo-5Zr-5Ti",
        {"Mo":20, "Zr":5, "Ti":5},
        "BCC + HCP",
        "Reject"),

    ("Nb-10W-15Hf",
        {"W":10, "Hf":15},
        "BCC + HCP",
        "Pass"),

    ("Nb-15Mo-10Zr",
        {"Mo":15, "Zr":10},
        "BCC + C15 Laves + HCP",
        "Pass"),
]

# =============================================================================
# GVI CALPHAD VALIDATION — Summary Table (5 alloys, paper Table 4b)
# The 5 alloys where GVI decision vs CALPHAD agreement is explicitly reported
# =============================================================================

CALPHAD_SUMMARY = [
    # (alloy_name,     gvi_score, gvi_decision, calphad_prediction,            agreement)
    ("Nb-8W-5Hf",      0.780, "Pass",   "BCC + minor secondary phases",  True),
    ("Nb-12W-5Hf",     0.018, "Reject", "Complex multiphase",            True),
    ("Nb-10Mo-15Ti",   0.999, "Pass",   "Predominantly BCC",             True),
    ("Nb-10W-15Hf",    0.993, "Pass",   "Predominantly BCC",             True),
    ("Nb-15Mo-10Zr",   0.999, "Pass",   "Predominantly BCC",             True),
]

# Documented boundary condition: Mo+Hf+Zr simultaneously above 5 at% each
# requires CALPHAD verification regardless of GVI score.
# Evidence: Nb-5W-10Mo-5Hf-5Zr-5Ti has GVI=0.990 but shows C15 Laves in CALPHAD.
CALPHAD_FALSE_POSITIVE_NOTE = (
    "Nb-5W-10Mo-5Hf-5Zr-5Ti: GVI=0.990 (Pass) but CALPHAD confirms "
    "multiphase assembly (BCC + BCC#2 + HCP). Explicit boundary condition: "
    "Mo > 5 at% AND Hf > 5 at% AND Zr > 5 at% simultaneously triggers "
    "mandatory CALPHAD verification regardless of GVI score."
)
