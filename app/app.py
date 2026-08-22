"""
URADES — Unified Refractory Alloy Descriptor and Embrittlement Screener
Reconstructed standalone reference implementation.

IMPORTANT
---------
* Case 1 is retained at Nb >= 79 wt%.
* Case 2 is Nb >= 50 at% after Case-1 routing.
* Case 3 is Nb < 50 at%.
* The equations, coefficients, bounds and datasets below are reconstructed
  from the surviving URADES project files.
* No new calibration coefficients or alloy values are introduced.
* Case 3 predicts a DBTT-risk zone, not a numerical DBTT.
* XGBoost is intentionally omitted from the current benchmark section.
* The surviving Case-1 validation files contain an internal inconsistency:
  the published 23-alloy prediction table does not reproduce the stated
  R²=0.856 when the current IAS equation is applied to the listed
  compositions. Therefore this script recalculates Case-1 metrics from the
  surviving composition table rather than silently forcing the reported
  metric.

Requires:
    numpy
    pandas
    matplotlib

Optional:
    scikit-learn, only if you later want to re-run RF/GPR yourself.

Outputs:
    ./urades_reconstructed_output/
"""

from __future__ import annotations

import math
import os
from itertools import product
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

OUTDIR = os.path.join(os.getcwd(), "urades_reconstructed_output")
os.makedirs(OUTDIR, exist_ok=True)


# =============================================================================
# ELEMENTAL CONSTANTS — SURVIVING CORE FILE
# =============================================================================

AW = {
    "Nb": 92.906, "W": 183.84, "Mo": 95.95,
    "V": 50.942, "Ti": 47.867, "Zr": 91.224,
    "Hf": 178.49, "Ta": 180.95, "Re": 186.21,
    "Al": 26.982, "Cr": 51.996,
}

DENSITY = {
    "Nb": 8.57, "W": 19.25, "Mo": 10.22,
    "V": 6.11, "Ti": 4.51, "Zr": 6.52,
    "Hf": 13.31, "Ta": 16.69, "Re": 21.02,
    "Al": 2.70, "Cr": 7.19,
}

VEC_TABLE = {
    "Nb": 5, "W": 6, "Mo": 6,
    "V": 5, "Ti": 4, "Zr": 4,
    "Hf": 4, "Ta": 5, "Re": 7,
    "Al": 3, "Cr": 6,
}

RADIUS = {
    "Nb": 146, "W": 139, "Mo": 140,
    "V": 134, "Ti": 147, "Zr": 160,
    "Hf": 159, "Ta": 146, "Re": 137,
    "Al": 143, "Cr": 128,
}

TM = {
    "Nb": 2477, "W": 3422, "Mo": 2623,
    "V": 1910, "Ti": 1668, "Zr": 1855,
    "Hf": 2233, "Ta": 3017, "Re": 3186,
    "Al": 660, "Cr": 1907,
}


# =============================================================================
# MODEL PARAMETERS
# =============================================================================

# -------------------------------------------------------------------------
# CASE 1 — Independent Alloying Shift
# Units: wt%
# -------------------------------------------------------------------------

IAS_BASELINE = -150.0

IAS_COEFFS = {
    "W": 8.0,
    "Mo": 15.0,
    "V": -5.0,
    "Ti": -2.0,
    "Zr": 1.0,
    "Hf": 0.5,
}

# Optional oxygen correction. The literature validation dataset did not
# contain a consistent oxygen column, so oxygen defaults to zero.
OXYGEN_COEFF = 0.15       # °C / ppm

CASE1_YS_BASE = 150.0

CASE1_YS_COEFFS = {
    "W": 15.0,
    "Mo": 25.0,
    "Zr": 15.0,
    "Hf": 8.0,
    "Ti": 5.0,
}


# -------------------------------------------------------------------------
# CASE 2 — Sponge Ratio
# Units: at%
# -------------------------------------------------------------------------

SR_BASELINE = -150.0

SR_kW = 2.244
SR_kMo = 7.899
SR_kBuffer = 1.723

SR_ALPHA = 0.0
SR_MAE = 26.0

TM_CORRECTION = 2.97


# -------------------------------------------------------------------------
# CASE 3 — Embrittlement Index
# Units: at%
# -------------------------------------------------------------------------

EI_ALPHA = 0.48

EI_DUCTILE = 0.10
EI_TRANSITION = 0.50
EI_BRITTLE = 15.50


# -------------------------------------------------------------------------
# GVI
# -------------------------------------------------------------------------

GVI_VEC_CENTER = 5.3
GVI_VEC_STEEPNESS = 15.0

GVI_DELTA_CENTER = 6.5
GVI_DELTA_STEEPNESS = 10.0

GVI_THRESHOLD = 0.5


# -------------------------------------------------------------------------
# Boundary conditions
# -------------------------------------------------------------------------

# Case 1: wt%
CASE1_LIMITS_WT = {
    "W": 20,
    "Mo": 10,
    "Hf": 10,
    "Zr": 5,
    "Ti": 10,
}

# Case 2: at%
CASE2_LIMITS_AT = {
    "W": 15,
    "Mo": 18,
    "Hf": 22.4,
    "Zr": 8.5,
    "Ti": 10,
}

# Case 3: at%
CASE3_LIMITS_AT = {
    "W": 20,
    "Mo": 20,
    "Hf": 20,
    "Zr": 33,
    "Ti": 33,
}

# Documented GVI/CALPHAD warning boundary.
CALPHAD_TRIGGER = {
    "Mo": 5.0,
    "Hf": 5.0,
    "Zr": 5.0,
}


# =============================================================================
# UNIT / PROPERTY UTILITIES
# =============================================================================

def clean_comp(comp: Dict[str, float]) -> Dict[str, float]:
    """Remove zero entries and validate elemental names."""
    out = {}

    for el, val in comp.items():
        if el not in AW:
            raise ValueError(f"Unknown element: {el}")

        if val < 0:
            raise ValueError(f"Negative concentration for {el}")

        if val > 0:
            out[el] = float(val)

    if not out or sum(out.values()) <= 0:
        raise ValueError("Composition is empty or sums to zero.")

    return out


def normalize_at(comp_at: Dict[str, float]) -> Dict[str, float]:
    comp_at = clean_comp(comp_at)
    total = sum(comp_at.values())

    return {
        el: value / total
        for el, value in comp_at.items()
    }


def atomic_to_weight(comp_at: Dict[str, float]) -> Dict[str, float]:
    """Convert atomic percent to weight percent."""
    comp_at = clean_comp(comp_at)

    masses = {
        el: value * AW[el]
        for el, value in comp_at.items()
    }

    total = sum(masses.values())

    return {
        el: 100.0 * mass / total
        for el, mass in masses.items()
    }


def weight_to_atomic(comp_wt: Dict[str, float]) -> Dict[str, float]:
    """Convert weight percent to atomic percent."""
    comp_wt = clean_comp(comp_wt)

    moles = {
        el: value / AW[el]
        for el, value in comp_wt.items()
    }

    total = sum(moles.values())

    return {
        el: 100.0 * mol / total
        for el, mol in moles.items()
    }


def calc_VEC(comp_at: Dict[str, float]) -> float:
    """Atomic-fraction-weighted VEC."""
    x = normalize_at(comp_at)

    return sum(
        frac * VEC_TABLE[el]
        for el, frac in x.items()
    )


def calc_delta(comp_at: Dict[str, float]) -> float:
    """Atomic size mismatch δ (%)."""
    x = normalize_at(comp_at)

    rbar = sum(
        frac * RADIUS[el]
        for el, frac in x.items()
    )

    delta2 = sum(
        frac * (1.0 - RADIUS[el] / rbar) ** 2
        for el, frac in x.items()
    )

    return math.sqrt(delta2) * 100.0


def calc_density(comp_at: Dict[str, float]) -> float:
    """Rule-of-mixtures density, g/cm³."""
    x = normalize_at(comp_at)

    numerator = sum(
        x[el] * DENSITY[el] * AW[el]
        for el in x
    )

    denominator = sum(
        x[el] * AW[el]
        for el in x
    )

    return numerator / denominator


def calc_Tm_ROM(comp_at: Dict[str, float]) -> float:
    """Rule-of-mixtures melting point, °C."""
    x = normalize_at(comp_at)

    return sum(
        x[el] * TM[el]
        for el in x
    )


def calc_Ceq(comp_at: Dict[str, float]) -> float:
    """Case-2 equivalent solute concentration."""
    W = comp_at.get("W", 0.0)
    Mo = comp_at.get("Mo", 0.0)
    Hf = comp_at.get("Hf", 0.0)
    Zr = comp_at.get("Zr", 0.0)
    Ti = comp_at.get("Ti", 0.0)

    return (
        3.0 * Mo
        + 2.0 * W
        + Zr
        + 0.5 * (Hf + Ti)
    )


# =============================================================================
# GVI
# =============================================================================

def _logistic_survival(
    x: float,
    centre: float,
    steepness: float
) -> float:

    z = steepness * (x - centre)

    if z > 700:
        return 0.0

    if z < -700:
        return 1.0

    return 1.0 / (1.0 + math.exp(z))


def calc_GVI(
    comp_at: Dict[str, float],
    case: int
) -> Dict[str, float]:

    vec = calc_VEC(comp_at)
    delta = calc_delta(comp_at)

    s_vec = _logistic_survival(
        vec,
        GVI_VEC_CENTER,
        GVI_VEC_STEEPNESS
    )

    s_delta = _logistic_survival(
        delta,
        GVI_DELTA_CENTER,
        GVI_DELTA_STEEPNESS
    )

    result = {
        "VEC": vec,
        "delta": delta,
        "S_VEC": s_vec,
        "S_delta": s_delta,
    }

    if case == 2:

        wt = atomic_to_weight(comp_at)

        W = wt.get("W", 0.0)
        Mo = wt.get("Mo", 0.0)
        Hf = wt.get("Hf", 0.0)
        Zr = wt.get("Zr", 0.0)
        Ti = wt.get("Ti", 0.0)

        # GVI SR_W uses weight percent.
        sr_w = (
            W + 0.077 * Mo
        ) / (
            Hf + Zr + Ti + 1.0
        )

        s_sr = _logistic_survival(
            sr_w,
            1.5,
            8.0
        )

        gvi = (
            s_sr
            * s_vec
            * s_delta
        )

        result["SR_W"] = sr_w
        result["S_SR"] = s_sr

    else:

        gvi = (
            s_vec
            * s_delta
        )

    result["GVI"] = gvi
    result["GVI_pass"] = gvi >= GVI_THRESHOLD

    # Documented CALPHAD boundary warning.
    Mo_at = comp_at.get("Mo", 0.0)
    Hf_at = comp_at.get("Hf", 0.0)
    Zr_at = comp_at.get("Zr", 0.0)

    result["calphad_trigger"] = (
        Mo_at > CALPHAD_TRIGGER["Mo"]
        and Hf_at > CALPHAD_TRIGGER["Hf"]
        and Zr_at > CALPHAD_TRIGGER["Zr"]
    )

    return result


# =============================================================================
# CASE ROUTING
# =============================================================================

def identify_case(
    comp_at: Dict[str, float]
) -> Tuple[int, float, float]:
    """
    URADES hierarchy:

        Case 1: Nb >= 79 wt%
        Case 2: Nb >= 50 at%, after Case-1 routing
        Case 3: Nb < 50 at%
    """

    comp_at = clean_comp(comp_at)

    wt = atomic_to_weight(comp_at)

    nb_wt = wt.get("Nb", 0.0)

    total_at = sum(comp_at.values())

    nb_at = (
        100.0
        * comp_at.get("Nb", 0.0)
        / total_at
    )

    if nb_wt >= 79.0:
        return 1, nb_wt, nb_at

    if nb_at >= 50.0:
        return 2, nb_wt, nb_at

    return 3, nb_wt, nb_at


def check_boundary_conditions(
    comp_at: Dict[str, float],
    case: int
):

    violations = []

    if case == 1:

        wt = atomic_to_weight(comp_at)

        for el, limit in CASE1_LIMITS_WT.items():

            if wt.get(el, 0.0) > limit + 1e-9:

                violations.append(
                    f"{el}={wt.get(el,0):.2f} wt% "
                    f"> {limit} wt%"
                )

    elif case == 2:

        for el, limit in CASE2_LIMITS_AT.items():

            if comp_at.get(el, 0.0) > limit + 1e-9:

                violations.append(
                    f"{el}={comp_at.get(el,0):.2f} at% "
                    f"> {limit} at%"
                )

        buffer = (
            comp_at.get("Hf", 0.0)
            + comp_at.get("Zr", 0.0)
            + comp_at.get("Ti", 0.0)
        )

        if buffer > 22.4 + 1e-9:

            violations.append(
                f"Buffer={buffer:.2f} at% > 22.4 at%"
            )

    elif case == 3:

        for el, limit in CASE3_LIMITS_AT.items():

            if comp_at.get(el, 0.0) > limit + 1e-9:

                violations.append(
                    f"{el}={comp_at.get(el,0):.2f} at% "
                    f"> {limit} at%"
                )

        max_el = (
            max(comp_at.values())
            / sum(comp_at.values())
            * 100.0
        )

        if max_el > 50.0 + 1e-9:

            violations.append(
                f"largest element={max_el:.2f} at% > 50 at%"
            )

    return (
        len(violations) == 0,
        violations
    )


# =============================================================================
# CASE 1 — IAS MODEL
# =============================================================================

def predict_case1(
    comp_wt: Dict[str, float],
    oxygen_ppm: float = 0.0
):

    wt = clean_comp(comp_wt)

    dbtt = IAS_BASELINE

    for el, coeff in IAS_COEFFS.items():

        dbtt += (
            coeff
            * wt.get(el, 0.0)
        )

    # Optional oxygen correction.
    dbtt += (
        OXYGEN_COEFF
        * oxygen_ppm
    )

    ys = CASE1_YS_BASE

    for el, coeff in CASE1_YS_COEFFS.items():

        ys += (
            coeff
            * wt.get(el, 0.0)
        )

    comp_at = weight_to_atomic(wt)

    return {
        "DBTT": dbtt,
        "DBTT_range": (
            dbtt - 15.93,
            dbtt + 15.93
        ),
        "YS_MPa": ys,
        "Tm_C": calc_Tm_ROM(comp_at),
        "density": calc_density(comp_at),
        "oxygen_ppm": oxygen_ppm,
    }


# =============================================================================
# CASE 2 — SPONGE RATIO MODEL
# =============================================================================

def predict_case2(
    comp_at: Dict[str, float]
):

    comp_at = clean_comp(comp_at)

    W = comp_at.get("W", 0.0)
    Mo = comp_at.get("Mo", 0.0)
    Hf = comp_at.get("Hf", 0.0)
    Zr = comp_at.get("Zr", 0.0)
    Ti = comp_at.get("Ti", 0.0)

    buffer = Hf + Zr + Ti

    dT = (
        SR_kW * W
        + SR_kMo * Mo
        + SR_kBuffer * buffer
    )

    SR = W / (buffer + 1.0)

    dbtt = (
        SR_BASELINE
        + dT * (1.0 + SR)
    )

    ceq = calc_Ceq(comp_at)

    ys = (
        150.0
        + 100.0 * math.sqrt(
            max(ceq, 0.0)
        )
    )

    tm = (
        calc_Tm_ROM(comp_at)
        - TM_CORRECTION * ceq
    )

    return {
        "DBTT": dbtt,
        "DBTT_range": (
            dbtt - SR_MAE,
            dbtt + SR_MAE
        ),
        "SR": SR,
        "dT": dT,
        "alpha": SR_ALPHA,
        "Ceq": ceq,
        "YS_MPa": ys,
        "Tm_C": tm,
        "density": calc_density(comp_at),
    }


# =============================================================================
# CASE 3 — EI CLASSIFIER
# =============================================================================

def classify_EI(ei: float) -> str:

    if ei < EI_DUCTILE:
        return "Ductile Zone"

    if ei < EI_TRANSITION:
        return "Transition Zone"

    if ei < EI_BRITTLE:
        return "Brittle Zone"

    return "Confirmed Brittle Zone"


def predict_case3(
    comp_at: Dict[str, float]
):

    comp_at = clean_comp(comp_at)

    W = comp_at.get("W", 0.0)
    Mo = comp_at.get("Mo", 0.0)
    Hf = comp_at.get("Hf", 0.0)
    Zr = comp_at.get("Zr", 0.0)
    Ti = comp_at.get("Ti", 0.0)

    buffer = Hf + Zr + Ti

    ei = (
        W + EI_ALPHA * Mo
    ) / (
        buffer + 1.0
    )

    return {
        "EI": ei,
        "zone": classify_EI(ei),
        "alpha": EI_ALPHA,
        "Tm_C": calc_Tm_ROM(comp_at),
        "density": calc_density(comp_at),
    }


# =============================================================================
# MASTER FORWARD ENGINE
# =============================================================================

def run_URADES(
    composition: Dict[str, float],
    input_unit: str = "at",
    apply_gvi: bool = True
):

    if input_unit.lower() == "wt":

        comp_wt = clean_comp(composition)
        comp_at = weight_to_atomic(comp_wt)

    elif input_unit.lower() == "at":

        comp_at = clean_comp(composition)
        comp_wt = atomic_to_weight(comp_at)

    else:

        raise ValueError(
            "input_unit must be 'at' or 'wt'"
        )

    case, nb_wt, nb_at = identify_case(comp_at)

    passed_bc, violations = (
        check_boundary_conditions(
            comp_at,
            case
        )
    )

    result = {
        "Case": case,
        "Nb_wt": nb_wt,
        "Nb_at": nb_at,
        "comp_at": comp_at,
        "comp_wt": comp_wt,
        "boundary_pass": passed_bc,
        "violations": violations,
    }

    if not passed_bc:

        result["status"] = (
            "REJECTED — boundary condition"
        )

        return result

    gvi = calc_GVI(
        comp_at,
        case
    )

    result["GVI_data"] = gvi

    if (
        apply_gvi
        and not gvi["GVI_pass"]
    ):

        result["status"] = (
            "FLAGGED — GVI < 0.5"
        )

        return result

    if case == 1:

        props = predict_case1(
            comp_wt
        )

    elif case == 2:

        props = predict_case2(
            comp_at
        )

    else:

        props = predict_case3(
            comp_at
        )

    result.update(props)

    result["status"] = (
        "OK"
        if gvi["GVI_pass"]
        else "FLAGGED — GVI < 0.5"
    )

    return result


# =============================================================================
# SURVIVING VALIDATION DATASETS
# =============================================================================

# -----------------------------------------------------------------------------
# CASE 1
# 23 dilute Nb engineering alloys.
# Units: wt%.
# -----------------------------------------------------------------------------

CASE1_DATA = [

    ("Nb-10Mo-10V",
     0, 10, 10, 0, 0, 0, -50.0),

    ("Nb-5Mo-5V",
     0, 5, 5, 0, 0, 0, -150.0),

    ("Cb-752",
     10, 0, 0, 0, 2.5, 0, -101.0),

    ("Nb-5W-1Zr",
     5, 0, 0, 0, 1, 0, -140.0),

    ("B-66",
     0, 5, 5, 0, 1, 0, -130.0),

    ("Nb-10Ti-5Mo-1Zr",
     0, 5, 0, 10, 1, 0, -110.0),

    ("Nb-10W-1Zr",
     10, 0, 0, 0, 1, 0, -85.0),

    ("Nb-5Mo-10V",
     0, 5, 10, 0, 0, 0, -110.0),

    ("Cb-74",
     0, 10, 0, 0, 0, 0, -80.0),

    ("Nb-10V-2Zr",
     0, 0, 10, 0, 2, 0, -185.0),

    ("Nb-5W-2.5Zr",
     5, 0, 0, 0, 2.5, 0, -120.0),

    ("Nb-10W-2Zr",
     10, 0, 0, 0, 2, 0, -80.0),

    ("Nb-15W-1Zr",
     15, 0, 0, 0, 1, 0, -40.0),

    ("Cb-751",
     0, 0, 0, 0, 0, 0, 20.0),

    ("Nb-15W-2.5Zr",
     15, 0, 0, 0, 2.5, 0, -20.0),

    ("Nb-5Ti-5V-1Zr",
     0, 0, 5, 5, 1, 0, -190.0),

    ("Nb-5W-5V-1Zr",
     5, 0, 5, 0, 1, 0, -140.0),

    ("Nb-10W-5V-1Zr",
     10, 0, 5, 0, 1, 0, -100.0),

    ("Nb-5Ti-5V",
     0, 0, 5, 5, 0, 0, -180.0),

    ("Nb-10Mo-5V",
     0, 10, 5, 0, 0, 0, -30.0),

    ("C103",
     0, 0, 0, 1, 0, 10, -150.0),

    ("Nb-10Hf-2Zr",
     0, 0, 0, 0, 2, 10, -145.0),

    ("Nb-10Ti-5W-1Zr",
     5, 0, 0, 10, 1, 0, -130.0),
]


# -----------------------------------------------------------------------------
# CASE 2
# 11 alloys.
# The final Nb-40Mo-10Ti alloy is marked False for the LOOCV subset.
# -----------------------------------------------------------------------------

CASE2_DATA = [

    ("C-103",
     0, 0, 10, 0, 1, -150.0, True),

    ("Nb-1Zr",
     0, 0, 0, 1, 0, -150.0, True),

    ("WC-3009",
     9, 0, 30, 0, 0, -75.0, True),

    ("Nb-10W-10Ti",
     10, 0, 0, 0, 10, -40.0, True),

    ("Nb-13Mo-10Ti",
     0, 13, 0, 0, 10, -30.0, True),

    ("Nb-18Mo-10Ti",
     0, 18, 0, 0, 10, 25.0, True),

    ("Nb-18Mo-8.5Zr",
     0, 18, 0, 8.5, 0, 45.0, True),

    ("Nb-5Mo-15W-5Ti",
     15, 5, 0, 0, 5, 130.0, True),

    ("Nb-5Mo-10Ti",
     0, 5, 0, 0, 10, -80.0, True),

    ("Cb-752",
     10, 0, 0, 2.5, 0, -50.0, True),

    ("Nb-40Mo-10Ti",
     0, 40, 0, 0, 10, 160.0, False),
]


# -----------------------------------------------------------------------------
# CASE 3
# 25 Nb-based RHEAs.
# Units: at%.
# -----------------------------------------------------------------------------

CASE3_DATA = [

    (
        "TiVNbTa",
        {"Ti":25, "V":25, "Nb":25, "Ta":25},
        "Ductile Zone",
        -37
    ),

    (
        "WTaTiVZr",
        {"W":20, "Ta":20, "Ti":20, "V":20, "Zr":20},
        "Transition Zone",
        20
    ),

    (
        "HfMoNbTaTiZr",
        {
            "Hf":16.67, "Mo":16.67,
            "Nb":16.67, "Ta":16.67,
            "Ti":16.67, "Zr":16.67
        },
        "Ductile Zone",
        -25
    ),

    (
        "HfMoTaTiZr",
        {
            "Hf":20, "Mo":20,
            "Ta":20, "Ti":20, "Zr":20
        },
        "Ductile Zone",
        -10
    ),

    (
        "NbTiZr",
        {"Nb":33.33, "Ti":33.33, "Zr":33.33},
        "Ductile Zone",
        -75
    ),

    (
        "NbTiVZr",
        {"Nb":25, "Ti":25, "V":25, "Zr":25},
        "Ductile Zone",
        -100
    ),

    (
        "NbTiV2Zr",
        {"Nb":20, "Ti":20, "V":40, "Zr":20},
        "Ductile Zone",
        -100
    ),

    (
        "Re0.3NbTiZr",
        {"Re":9.1, "Nb":30.3, "Ti":30.3, "Zr":30.3},
        "Ductile Zone",
        -60
    ),

    (
        "Al2.5(NbTiZr)",
        {"Al":2.5, "Nb":32.5, "Ti":32.5, "Zr":32.5},
        "Ductile Zone",
        -80
    ),

    (
        "Al5(NbTiZr)",
        {"Al":5.0, "Nb":31.67, "Ti":31.67, "Zr":31.67},
        "Ductile Zone",
        -50
    ),

    (
        "Al7.5(NbTiZr)",
        {"Al":7.5, "Nb":30.83, "Ti":30.83, "Zr":30.83},
        "Ductile Zone",
        -20
    ),

    (
        "Re0.3TaTiZr",
        {"Re":9.1, "Ta":30.3, "Ti":30.3, "Zr":30.3},
        "Transition Zone",
        30
    ),

    (
        "TiZrHfNbTa",
        {"Ti":20, "Zr":20, "Hf":20, "Nb":20, "Ta":20},
        "Ductile Zone",
        -26
    ),

    (
        "VNbMoTaW",
        {"V":20, "Nb":20, "Mo":20, "Ta":20, "W":20},
        "Confirmed Brittle Zone",
        354
    ),

    (
        "NbMoTaW",
        {"Nb":25, "Mo":25, "Ta":25, "W":25},
        "Confirmed Brittle Zone",
        200
    ),

    (
        "TiNbMoTaW",
        {"Ti":20, "Nb":20, "Mo":20, "Ta":20, "W":20},
        "Brittle Zone",
        -25
    ),

    (
        "NbTiV",
        {"Nb":33.33, "Ti":33.33, "V":33.33},
        "Ductile Zone",
        -75
    ),

    (
        "HfNbTaTiZr",
        {"Hf":20, "Nb":20, "Ta":20, "Ti":20, "Zr":20},
        "Ductile Zone",
        -196
    ),

    (
        "NbTaHfTiZrV0.5",
        {
            "Nb":18.18, "Ta":18.18,
            "Hf":18.18, "Ti":18.18,
            "Zr":18.18, "V":9.1
        },
        "Ductile Zone",
        -50
    ),

    (
        "Nb45Ta25Ti15Hf15",
        {"Nb":45, "Ta":25, "Ti":15, "Hf":15},
        "Ductile Zone",
        -50
    ),

    (
        "Mo0.2NbTiZr",
        {"Mo":6.25, "Nb":31.25, "Ti":31.25, "Zr":31.25},
        "Ductile Zone",
        -50
    ),

    (
        "Mo0.6NbTiZr",
        {"Mo":16.67, "Nb":27.78, "Ti":27.78, "Zr":27.78},
        "Transition Zone",
        10
    ),

    (
        "HfNbTiZr",
        {"Hf":25, "Nb":25, "Ti":25, "Zr":25},
        "Ductile Zone",
        -100
    ),

    (
        "TiVNbMoTaW",
        {
            "Ti":16.67, "V":16.67,
            "Nb":16.67, "Mo":16.67,
            "Ta":16.67, "W":16.67
        },
        "Brittle Zone",
        -10
    ),

    (
        "MoNbTaTiZr",
        {"Mo":20, "Nb":20, "Ta":20, "Ti":20, "Zr":20},
        "Brittle Zone",
        50
    ),
]


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_case1():

    rows = []

    for (
        name,
        W,
        Mo,
        V,
        Ti,
        Zr,
        Hf,
        exp
    ) in CASE1_DATA:

        comp_wt = {
            "Nb": 100 - W - Mo - V - Ti - Zr - Hf,
            "W": W,
            "Mo": Mo,
            "V": V,
            "Ti": Ti,
            "Zr": Zr,
            "Hf": Hf,
        }

        pred = predict_case1(
            comp_wt
        )["DBTT"]

        rows.append({
            "Alloy": name,
            "Experimental": exp,
            "Predicted": pred,
            "Residual": pred - exp,
            "Abs_Error": abs(pred - exp),
        })

    df = pd.DataFrame(rows)

    y = df["Experimental"].to_numpy()
    p = df["Predicted"].to_numpy()

    ss_res = np.sum(
        (y - p) ** 2
    )

    ss_tot = np.sum(
        (y - y.mean()) ** 2
    )

    metrics = {
        "n": len(df),
        "R2": 1 - ss_res / ss_tot,
        "MAE": np.mean(np.abs(p-y)),
        "MedianAE": np.median(np.abs(p-y)),
        "RMSE": np.sqrt(
            np.mean((p-y)**2)
        ),
        "Within20_pct":
            100 * np.mean(
                np.abs(p-y) <= 20
            ),
        "Within30_pct":
            100 * np.mean(
                np.abs(p-y) <= 30
            ),
        "Within50_pct":
            100 * np.mean(
                np.abs(p-y) <= 50
            ),
    }

    return df, metrics


def validate_case3():

    rows = []

    for (
        name,
        comp,
        exp_zone,
        exp_dbtt
    ) in CASE3_DATA:

        pred = predict_case3(comp)

        rows.append({
            "Alloy": name,
            "EI": pred["EI"],
            "Experimental_Zone": exp_zone,
            "Predicted_Zone": pred["zone"],
            "Correct": (
                pred["zone"]
                == exp_zone
            ),
            "Experimental_DBTT": exp_dbtt,
        })

    df = pd.DataFrame(rows)

    metrics = {
        "n": len(df),
        "Accuracy_pct":
            100 * df["Correct"].mean(),
    }

    return df, metrics


# =============================================================================
# INVERSE DESIGN
# =============================================================================

TARGETS = {
    "DBTT_max": -50.0,
    "density_max": 10.0,
    "YS_min": 400.0,
    "GVI_min": GVI_THRESHOLD,
}


# Case 1 grid: wt%
C1_GRID = {
    "W": range(0, 21, 3),
    "Mo": range(0, 11, 2),
    "Hf": range(0, 11, 2),
    "Zr": range(0, 6, 1),
    "Ti": range(0, 11, 2),
}


# Case 2 grid: at%
C2_GRID = {
    "W": range(0, 16, 2),
    "Mo": range(0, 18, 3),
    "Hf": range(0, 23, 3),
    "Zr": range(0, 9, 2),
    "Ti": range(0, 11, 2),
}


def search_case1(targets=TARGETS):

    candidates = []

    screened = 0
    rejected_bc = 0
    rejected_gvi = 0

    for (
        W,
        Mo,
        Hf,
        Zr,
        Ti
    ) in product(
        C1_GRID["W"],
        C1_GRID["Mo"],
        C1_GRID["Hf"],
        C1_GRID["Zr"],
        C1_GRID["Ti"]
    ):

        Nb = (
            100
            - W
            - Mo
            - Hf
            - Zr
            - Ti
        )

        # IMPORTANT:
        # Case 1 remains Nb >= 79 wt%.
        if Nb < 79:
            continue

        if Nb > 99.5:
            continue

        screened += 1

        comp_wt = {
            "Nb": Nb,
            "W": W,
            "Mo": Mo,
            "Hf": Hf,
            "Zr": Zr,
            "Ti": Ti,
        }

        comp_at = weight_to_atomic(
            comp_wt
        )

        passed, _ = (
            check_boundary_conditions(
                comp_at,
                1
            )
        )

        if not passed:
            rejected_bc += 1
            continue

        gvi = calc_GVI(
            comp_at,
            1
        )

        if (
            gvi["GVI"]
            < targets["GVI_min"]
        ):
            rejected_gvi += 1
            continue

        props = predict_case1(
            comp_wt
        )

        if (
            props["DBTT"]
            > targets["DBTT_max"]
        ):
            continue

        if (
            props["density"]
            > targets["density_max"]
        ):
            continue

        if (
            props["YS_MPa"]
            < targets["YS_min"]
        ):
            continue

        candidates.append({
            "Case": 1,
            "Alloy":
                f"Nb{Nb:.0f}"
                f"W{W}"
                f"Mo{Mo}"
                f"Hf{Hf}"
                f"Zr{Zr}"
                f"Ti{Ti}",
            "Nb": Nb,
            "W": W,
            "Mo": Mo,
            "Hf": Hf,
            "Zr": Zr,
            "Ti": Ti,
            "DBTT": props["DBTT"],
            "YS_MPa": props["YS_MPa"],
            "Tm_C": props["Tm_C"],
            "density": props["density"],
            "GVI": gvi["GVI"],
            "SR": np.nan,
            "EI": np.nan,
        })

    return (
        pd.DataFrame(candidates),
        screened,
        rejected_bc,
        rejected_gvi
    )


def search_case2(targets=TARGETS):

    candidates = []

    screened = 0
    rejected_bc = 0
    rejected_gvi = 0

    for (
        W,
        Mo,
        Hf,
        Zr,
        Ti
    ) in product(
        C2_GRID["W"],
        C2_GRID["Mo"],
        C2_GRID["Hf"],
        C2_GRID["Zr"],
        C2_GRID["Ti"]
    ):

        Nb = (
            100
            - W
            - Mo
            - Hf
            - Zr
            - Ti
        )

        if Nb < 50:
            continue

        if Nb > 99.5:
            continue

        buffer = (
            Hf + Zr + Ti
        )

        if buffer > 22.4:
            continue

        screened += 1

        comp_at = {
            "Nb": Nb,
            "W": W,
            "Mo": Mo,
            "Hf": Hf,
            "Zr": Zr,
            "Ti": Ti,
        }

        passed, _ = (
            check_boundary_conditions(
                comp_at,
                2
            )
        )

        if not passed:
            rejected_bc += 1
            continue

        gvi = calc_GVI(
            comp_at,
            2
        )

        if (
            gvi["GVI"]
            < targets["GVI_min"]
        ):
            rejected_gvi += 1
            continue

        props = predict_case2(
            comp_at
        )

        if (
            props["DBTT"]
            > targets["DBTT_max"]
        ):
            continue

        if (
            props["density"]
            > targets["density_max"]
        ):
            continue

        if (
            props["YS_MPa"]
            < targets["YS_min"]
        ):
            continue

        candidates.append({
            "Case": 2,
            "Alloy":
                f"Nb{Nb:.0f}"
                f"W{W}"
                f"Mo{Mo}"
                f"Hf{Hf}"
                f"Zr{Zr}"
                f"Ti{Ti}",
            "Nb": Nb,
            "W": W,
            "Mo": Mo,
            "Hf": Hf,
            "Zr": Zr,
            "Ti": Ti,
            "DBTT": props["DBTT"],
            "YS_MPa": props["YS_MPa"],
            "Tm_C": props["Tm_C"],
            "density": props["density"],
            "GVI": gvi["GVI"],
            "SR": props["SR"],
            "EI": np.nan,
        })

    return (
        pd.DataFrame(candidates),
        screened,
        rejected_bc,
        rejected_gvi
    )


# -------------------------------------------------------------------------
# Case 3 inverse search
# -------------------------------------------------------------------------
# This is deliberately zone-based because Case 3 has no validated numerical
# DBTT model. The coarse values follow the surviving inverse-search logic.

C3_SEARCH_VALUES = {
    "W": [0, 5, 10, 15, 20],
    "Mo": [0, 5, 10, 15, 20],
    "Hf": [0, 5, 10, 15, 20],
    "Zr": [0, 5, 10, 15, 20, 25, 30],
    "Ti": [0, 5, 10, 15, 20, 25, 30],
    "Ta": [15, 20, 25],
}


def search_case3(
    target_zone="Ductile Zone",
    max_density=10.0
):

    candidates = []

    for (
        W,
        Mo,
        Hf,
        Zr,
        Ti,
        Ta
    ) in product(
        C3_SEARCH_VALUES["W"],
        C3_SEARCH_VALUES["Mo"],
        C3_SEARCH_VALUES["Hf"],
        C3_SEARCH_VALUES["Zr"],
        C3_SEARCH_VALUES["Ti"],
        C3_SEARCH_VALUES["Ta"]
    ):

        Nb = (
            100
            - W
            - Mo
            - Hf
            - Zr
            - Ti
            - Ta
        )

        if Nb < 0 or Nb >= 50:
            continue

        comp_at = {
            "Nb": Nb,
            "Ta": Ta,
            "W": W,
            "Mo": Mo,
            "Hf": Hf,
            "Zr": Zr,
            "Ti": Ti,
        }

        passed, _ = (
            check_boundary_conditions(
                comp_at,
                3
            )
        )

        if not passed:
            continue

        gvi = calc_GVI(
            comp_at,
            3
        )

        if gvi["GVI"] < GVI_THRESHOLD:
            continue

        props = predict_case3(
            comp_at
        )

        if props["zone"] != target_zone:
            continue

        if props["density"] > max_density:
            continue

        candidates.append({
            "Case": 3,
            "Alloy":
                f"Nb{Nb:.1f}"
                f"Ta{Ta}"
                f"W{W}"
                f"Mo{Mo}"
                f"Hf{Hf}"
                f"Zr{Zr}"
                f"Ti{Ti}",
            "Nb": Nb,
            "Ta": Ta,
            "W": W,
            "Mo": Mo,
            "Hf": Hf,
            "Zr": Zr,
            "Ti": Ti,
            "DBTT": np.nan,
            "YS_MPa": np.nan,
            "Tm_C": props["Tm_C"],
            "density": props["density"],
            "GVI": gvi["GVI"],
            "SR": np.nan,
            "EI": props["EI"],
            "Zone": props["zone"],
        })

    return pd.DataFrame(candidates)


def run_inverse_design(
    targets=TARGETS,
    case3_zone="Ductile Zone"
):

    c1, s1, b1, g1 = (
        search_case1(targets)
    )

    c2, s2, b2, g2 = (
        search_case2(targets)
    )

    c3 = search_case3(
        case3_zone,
        targets["density_max"]
    )

    frames = [
        x
        for x in (c1, c2, c3)
        if not x.empty
    ]

    if frames:

        all_candidates = pd.concat(
            frames,
            ignore_index=True
        )

    else:

        all_candidates = (
            pd.DataFrame()
        )

    if not all_candidates.empty:

        all_candidates.to_csv(
            os.path.join(
                OUTDIR,
                "inverse_design_candidates.csv"
            ),
            index=False
        )

    return {
        "case1": c1,
        "case2": c2,
        "case3": c3,
        "all": all_candidates,
        "stats": {
            "case1": (s1, b1, g1),
            "case2": (s2, b2, g2),
        }
    }


# =============================================================================
# CASE 1 PARITY PLOT
# =============================================================================

def save_case1_parity():

    df, metrics = (
        validate_case1()
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.scatter(
        df["Experimental"],
        df["Predicted"],
        s=45,
        alpha=0.8
    )

    lo = min(
        df["Experimental"].min(),
        df["Predicted"].min()
    ) - 10

    hi = max(
        df["Experimental"].max(),
        df["Predicted"].max()
    ) + 10

    ax.plot(
        [lo, hi],
        [lo, hi],
        "k--",
        lw=1
    )

    ax.set_xlabel(
        "Experimental DBTT (°C)"
    )

    ax.set_ylabel(
        "URADES predicted DBTT (°C)"
    )

    ax.set_title(
        "Case 1 — IAS parity plot"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    text = (
        f"n = {metrics['n']}\n"
        f"Recomputed R² = {metrics['R2']:.3f}\n"
        f"Recomputed MAE = {metrics['MAE']:.2f} °C\n"
        f"Recomputed RMSE = {metrics['RMSE']:.2f} °C"
    )

    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85
        )
    )

    fig.tight_layout()

    path = os.path.join(
        OUTDIR,
        "case1_parity.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path, metrics


# =============================================================================
# CASE 3 EI VALIDATION PLOT
# =============================================================================

def save_case3_validation():

    df, metrics = (
        validate_case3()
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    order = (
        df.sort_values("EI")
        .reset_index(drop=True)
    )

    x = np.arange(
        len(order)
    )

    axes[0].scatter(
        x,
        order["EI"],
        s=45,
        alpha=0.8
    )

    axes[0].axhline(
        EI_DUCTILE,
        ls="--",
        lw=1
    )

    axes[0].axhline(
        EI_TRANSITION,
        ls="--",
        lw=1
    )

    axes[0].axhline(
        EI_BRITTLE,
        ls="--",
        lw=1
    )

    axes[0].set_yscale(
        "symlog",
        linthresh=0.05
    )

    axes[0].set_xlabel(
        "Alloy (sorted by EI)"
    )

    axes[0].set_ylabel(
        "Embrittlement Index, EI"
    )

    axes[0].set_title(
        "Case 3 — EI distribution"
    )

    axes[0].grid(
        True,
        alpha=0.25
    )

    zones = [
        "Ductile Zone",
        "Transition Zone",
        "Brittle Zone",
        "Confirmed Brittle Zone"
    ]

    exp_counts = [
        sum(
            df["Experimental_Zone"] == z
        )
        for z in zones
    ]

    pred_counts = [
        sum(
            df["Predicted_Zone"] == z
        )
        for z in zones
    ]

    xx = np.arange(
        len(zones)
    )

    width = 0.36

    axes[1].bar(
        xx - width/2,
        exp_counts,
        width,
        label="Experimental"
    )

    axes[1].bar(
        xx + width/2,
        pred_counts,
        width,
        label="Predicted",
        alpha=0.65
    )

    axes[1].set_xticks(
        xx
    )

    axes[1].set_xticklabels(
        ["D", "T", "B", "CB"]
    )

    axes[1].set_ylabel(
        "Number of alloys"
    )

    axes[1].set_title(
        f"Zone distribution — "
        f"accuracy {metrics['Accuracy_pct']:.0f}%"
    )

    axes[1].grid(
        True,
        axis="y",
        alpha=0.25
    )

    axes[1].legend()

    fig.suptitle(
        "URADES Case 3 — "
        "Embrittlement Index validation"
    )

    fig.tight_layout()

    path = os.path.join(
        OUTDIR,
        "case3_ei_validation.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path, metrics


# =============================================================================
# ALPHA SENSITIVITY
# =============================================================================

def save_alpha_sensitivity():

    alphas = np.linspace(
        0,
        1,
        201
    )

    # Case 2: ten-alloy LOOCV subset.
    c2 = [
        (
            name,
            W,
            Mo,
            Hf,
            Zr,
            Ti,
            exp
        )
        for (
            name,
            W,
            Mo,
            Hf,
            Zr,
            Ti,
            exp,
            in_loocv
        ) in CASE2_DATA
        if in_loocv
    ]

    def case2_r2(alpha):

        y = []
        p = []

        for (
            _,
            W,
            Mo,
            Hf,
            Zr,
            Ti,
            exp
        ) in c2:

            buffer = (
                Hf + Zr + Ti
            )

            dt = (
                SR_kW * W
                + SR_kMo * Mo
                + SR_kBuffer * buffer
            )

            sr = (
                W
                + alpha * Mo
            ) / (
                buffer + 1
            )

            pred = (
                SR_BASELINE
                + dt * (1 + sr)
            )

            y.append(exp)
            p.append(pred)

        y = np.asarray(y)
        p = np.asarray(p)

        return (
            1
            - np.sum((y-p)**2)
            / np.sum((y-y.mean())**2)
        )

    def case3_accuracy(alpha):

        correct = 0

        for (
            _,
            comp,
            exp_zone,
            _
        ) in CASE3_DATA:

            W = comp.get(
                "W",
                0
            )

            Mo = comp.get(
                "Mo",
                0
            )

            buffer = (
                comp.get("Hf", 0)
                + comp.get("Zr", 0)
                + comp.get("Ti", 0)
            )

            ei = (
                W
                + alpha * Mo
            ) / (
                buffer + 1
            )

            if (
                classify_EI(ei)
                == exp_zone
            ):
                correct += 1

        return (
            correct
            / len(CASE3_DATA)
        )

    r2 = np.array([
        case2_r2(a)
        for a in alphas
    ])

    accuracy = np.array([
        case3_accuracy(a)
        for a in alphas
    ])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes[0].plot(
        alphas,
        r2
    )

    axes[0].axvline(
        0.0,
        ls="--",
        lw=1
    )

    axes[0].set_xlabel(
        "Mo weighting, α"
    )

    axes[0].set_ylabel(
        "LOOCV R²"
    )

    axes[0].set_title(
        "Case 2 — SR α sensitivity"
    )

    axes[0].grid(
        True,
        alpha=0.25
    )

    axes[1].plot(
        alphas,
        100 * accuracy
    )

    axes[1].axvline(
        0.48,
        ls="--",
        lw=1
    )

    axes[1].set_xlabel(
        "Mo weighting, α"
    )

    axes[1].set_ylabel(
        "LOO accuracy (%)"
    )

    axes[1].set_title(
        "Case 3 — EI α sensitivity"
    )

    axes[1].grid(
        True,
        alpha=0.25
    )

    fig.suptitle(
        "URADES cross-system Mo weighting"
    )

    fig.tight_layout()

    path = os.path.join(
        OUTDIR,
        "alpha_sensitivity.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# =============================================================================
# RF / GPR BENCHMARK PLOT
# =============================================================================

def save_benchmark_rf_gpr():

    """
    Previously reported LOOCV benchmark values retained from the surviving
    benchmark/figure file.

    Case 1:
        URADES = 0.856
        RF     = 0.702
        GPR    = 0.914

    Case 2:
        URADES = 0.871
        RF     = 0.182
        GPR    = 0.306

    XGBoost is deliberately omitted.
    """

    methods = [
        "URADES",
        "RF",
        "GPR"
    ]

    case1 = [
        0.856,
        0.702,
        0.914
    ]

    case2 = [
        0.871,
        0.182,
        0.306
    ]

    x = np.arange(2)

    width = 0.23

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for i, method in enumerate(methods):

        values = [
            case1[i],
            case2[i]
        ]

        bars = ax.bar(
            x + (i-1)*width,
            values,
            width,
            label=method
        )

        for bar, value in zip(
            bars,
            values
        ):

            ax.text(
                bar.get_x()
                + bar.get_width()/2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        ["Case 1", "Case 2"]
    )

    ax.set_ylabel(
        "LOOCV R²"
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.set_title(
        "URADES benchmark against RF and GPR"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    path = os.path.join(
        OUTDIR,
        "benchmark_rf_gpr.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# =============================================================================
# GVI / CALPHAD SUMMARY
# =============================================================================

def save_gvi_summary():

    """
    Five-composition GVI/CALPHAD summary surviving in the project files.
    This is deliberately presented as a summary, not reconstructed as the
    complete CALPHAD table.
    """

    labels = [
        "Nb-8W-5Hf",
        "Nb-12W-5Hf",
        "Nb-20Mo-5Zr-5Ti",
        "Nb-10W-15Hf",
        "Nb-15Mo-10Zr",
    ]

    gvi = [
        0.780,
        0.018,
        0.985,
        0.993,
        0.999
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    bars = ax.bar(
        np.arange(len(labels)),
        gvi
    )

    ax.axhline(
        0.5,
        ls="--",
        lw=1,
        label="GVI threshold = 0.5"
    )

    ax.set_xticks(
        np.arange(len(labels))
    )

    ax.set_xticklabels(
        labels,
        rotation=25,
        ha="right"
    )

    ax.set_ylabel(
        "GVI"
    )

    ax.set_ylim(
        0,
        1.08
    )

    ax.set_title(
        "GVI summary — surviving CALPHAD validation data"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25
    )

    ax.legend()

    for bar, value in zip(
        bars,
        gvi
    ):

        ax.text(
            bar.get_x()
            + bar.get_width()/2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            fontsize=8
        )

    fig.tight_layout()

    path = os.path.join(
        OUTDIR,
        "gvi_calphad_summary.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# =============================================================================
# INVERSE DESIGN PLOTS
# =============================================================================

def save_inverse_design_plots(
    inverse_results
):

    df = inverse_results["all"]

    if df.empty:
        return []

    paths = []

    # ---------------------------------------------------------------------
    # DBTT vs density
    # ---------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for case in sorted(
        df["Case"].unique()
    ):

        sub = df[
            df["Case"] == case
        ]

        if "DBTT" not in sub:
            continue

        sub = sub.dropna(
            subset=[
                "DBTT",
                "density"
            ]
        )

        if sub.empty:
            continue

        ax.scatter(
            sub["DBTT"],
            sub["density"],
            s=28,
            alpha=0.7,
            label=f"Case {case}"
        )

    ax.axvline(
        TARGETS["DBTT_max"],
        ls="--",
        lw=1
    )

    ax.axhline(
        TARGETS["density_max"],
        ls="--",
        lw=1
    )

    ax.set_xlabel(
        "Predicted DBTT (°C)"
    )

    ax.set_ylabel(
        "Density (g cm$^{-3}$)"
    )

    ax.set_title(
        "URADES inverse design — DBTT vs density"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    path = os.path.join(
        OUTDIR,
        "inverse_design_dbbt_density.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    paths.append(path)

    # ---------------------------------------------------------------------
    # DBTT vs YS
    # ---------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for case in [1, 2]:

        sub = df[
            df["Case"] == case
        ].dropna(
            subset=[
                "DBTT",
                "YS_MPa"
            ]
        )

        if sub.empty:
            continue

        ax.scatter(
            sub["DBTT"],
            sub["YS_MPa"],
            s=28,
            alpha=0.7,
            label=f"Case {case}"
        )

    ax.axvline(
        TARGETS["DBTT_max"],
        ls="--",
        lw=1
    )

    ax.axhline(
        TARGETS["YS_min"],
        ls="--",
        lw=1
    )

    ax.set_xlabel(
        "Predicted DBTT (°C)"
    )

    ax.set_ylabel(
        "Predicted YS (MPa)"
    )

    ax.set_title(
        "URADES inverse design — DBTT vs yield strength"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    path = os.path.join(
        OUTDIR,
        "inverse_design_dbbt_ys.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    paths.append(path)

    # ---------------------------------------------------------------------
    # Case 2: DBTT vs SR
    # ---------------------------------------------------------------------

    if "SR" in df:

        sub = (
            df[df["Case"] == 2]
            .dropna(
                subset=[
                    "SR",
                    "DBTT"
                ]
            )
        )

        if not sub.empty:

            fig, ax = plt.subplots(
                figsize=(7, 5)
            )

            ax.scatter(
                sub["SR"],
                sub["DBTT"],
                s=30,
                alpha=0.7
            )

            ax.axhline(
                TARGETS["DBTT_max"],
                ls="--",
                lw=1
            )

            ax.set_xlabel(
                "Sponge Ratio, SR"
            )

            ax.set_ylabel(
                "Predicted DBTT (°C)"
            )

            ax.set_title(
                "Case 2 — DBTT vs Sponge Ratio"
            )

            ax.grid(
                True,
                alpha=0.25
            )

            fig.tight_layout()

            path = os.path.join(
                OUTDIR,
                "design_space_dbbt_sr.png"
            )

            fig.savefig(
                path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

            paths.append(path)

    # ---------------------------------------------------------------------
    # Case 3: EI vs experimental DBTT
    # ---------------------------------------------------------------------

    df3, _ = validate_case3()

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.scatter(
        df3["EI"],
        df3["Experimental_DBTT"],
        s=35,
        alpha=0.8
    )

    ax.axvline(
        EI_DUCTILE,
        ls="--",
        lw=1
    )

    ax.axvline(
        EI_TRANSITION,
        ls="--",
        lw=1
    )

    ax.axvline(
        EI_BRITTLE,
        ls="--",
        lw=1
    )

    ax.set_xlabel(
        "Embrittlement Index, EI"
    )

    ax.set_ylabel(
        "Experimental DBTT (°C)"
    )

    ax.set_title(
        "Case 3 — EI vs experimental DBTT"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    fig.tight_layout()

    path = os.path.join(
        OUTDIR,
        "design_space_dbbt_ei.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    paths.append(path)

    return paths


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================

def print_validation_summary():

    df1, m1 = (
        validate_case1()
    )

    df3, m3 = (
        validate_case3()
    )

    print()
    print("=" * 72)
    print("URADES RECONSTRUCTED VALIDATION")
    print("=" * 72)

    print()
    print("CASE 1 — IAS")
    print("-" * 72)

    for key, value in m1.items():

        if isinstance(
            value,
            (float, np.floating)
        ):

            print(
                f"{key:15s}: "
                f"{value:.3f}"
            )

        else:

            print(
                f"{key:15s}: "
                f"{value}"
            )

    print()
    print("CASE 2 — SR")
    print("-" * 72)
    print("Reported LOOCV R² : 0.871")
    print("Reported MAE      : 26.0 °C")
    print("Reported LOOCV n  : 10")
    print("Full dataset n    : 11")

    print()
    print("CASE 3 — EI")
    print("-" * 72)

    for key, value in m3.items():

        if isinstance(
            value,
            (float, np.floating)
        ):

            print(
                f"{key:15s}: "
                f"{value:.3f}"
            )

        else:

            print(
                f"{key:15s}: "
                f"{value}"
            )

    print()
    print("Important Case-1 note:")
    print(
        "The surviving Case-1 files do not reproduce the stated "
        "R²=0.856 when the current IAS equation is applied to the "
        "surviving 23-alloy composition table."
    )
    print(
        "This script reports the recomputed value instead of "
        "silently modifying the equation or data."
    )


# =============================================================================
# FORWARD DEMONSTRATION
# =============================================================================

def demo_forward():

    examples = {

        "Case 1 example": {
            "unit": "wt",
            "comp": {
                "Nb": 84,
                "Mo": 5,
                "Ti": 10,
                "Zr": 1
            },
        },

        "Case 2 example": {
            "unit": "at",
            "comp": {
                "Nb": 65,
                "W": 5,
                "Mo": 5,
                "Hf": 10,
                "Zr": 5,
                "Ti": 10
            },
        },

        "Case 3 example": {
            "unit": "at",
            "comp": {
                "Nb": 20,
                "Ta": 20,
                "W": 20,
                "Mo": 20,
                "Ti": 20
            },
        },
    }

    print()
    print("=" * 72)
    print("FORWARD DEMONSTRATIONS")
    print("=" * 72)

    for title, item in examples.items():

        result = run_URADES(
            item["comp"],
            item["unit"],
            apply_gvi=False
        )

        print()
        print(title)

        print(
            f"  Case       : "
            f"{result['Case']}"
        )

        print(
            f"  Nb         : "
            f"{result['Nb_wt']:.2f} wt% / "
            f"{result['Nb_at']:.2f} at%"
        )

        print(
            f"  GVI        : "
            f"{result['GVI_data']['GVI']:.4f}"
        )

        print(
            f"  GVI status : "
            f"{'PASS' if result['GVI_data']['GVI_pass'] else 'FLAG'}"
        )

        for key in [
            "DBTT",
            "YS_MPa",
            "Tm_C",
            "density",
            "SR",
            "Ceq",
            "EI",
            "zone",
        ]:

            if key in result:

                print(
                    f"  {key:10s}: "
                    f"{result[key]}"
                )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 72)
    print(
        "URADES — RECONSTRUCTED "
        "STANDALONE PYTHON"
    )
    print(
        "Case 1 retained at Nb >= 79 wt%"
    )
    print("=" * 72)

    print_validation_summary()

    demo_forward()

    print()
    print("=" * 72)
    print("RUNNING INVERSE DESIGN")
    print("=" * 72)

    inverse_results = (
        run_inverse_design(
            TARGETS,
            case3_zone="Ductile Zone"
        )
    )

    print()
    print("Inverse-design candidate counts:")

    for key in [
        "case1",
        "case2",
        "case3",
        "all"
    ]:

        print(
            f"  {key:8s}: "
            f"{len(inverse_results[key])}"
        )

    if not inverse_results["all"].empty:

        print()
        print("Top candidates:")

        desired_cols = [
            "Case",
            "Alloy",
            "DBTT",
            "YS_MPa",
            "Tm_C",
            "density",
            "GVI",
            "SR",
            "EI",
            "Zone",
        ]

        cols = [
            c
            for c in desired_cols
            if c in inverse_results["all"].columns
        ]

        print(
            inverse_results["all"]
            [cols]
            .head(15)
            .to_string(index=False)
        )

    print()
    print("=" * 72)
    print("GENERATING FIGURES")
    print("=" * 72)

    save_case1_parity()
    save_case3_validation()
    save_alpha_sensitivity()
    save_benchmark_rf_gpr()
    save_gvi_summary()
    save_inverse_design_plots(
        inverse_results
    )

    print()
    print("Finished.")
    print()
    print(
        "All outputs written to:"
    )
    print(
        OUTDIR
    )


if __name__ == "__main__":
    main()
