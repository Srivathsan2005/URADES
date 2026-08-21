"""
urades/core.py
==============
URADES — Unified Refractory Alloy Descriptor and Embrittlement Screener

Physics-based hierarchical framework for DBTT prediction and embrittlement
screening of Nb-based refractory alloys. Three alloy families are handled
by three distinct models, selected automatically based on composition.

Framework overview
------------------
  Input composition (at%)
        │
        ▼
  [Boundary condition check] ── fail ──► rejected, reason returned
        │ pass
        ▼
  [GVI phase stability gate] ── GVI < 0.5 ──► flagged (not hard-rejected)
        │
        ▼
  [Case classifier]
        │
        ├── Case 1 (Nb ≥ 79 wt%)  ──► IAS model  → DBTT (°C)
        ├── Case 2 (Nb ≥ 50 at%)  ──► SR  model  → DBTT (°C)
        └── Case 3 (Nb  < 50 at%) ──► EI  model  → Zone classification

Key finding
-----------
The Mo embrittlement weight α transitions from 0 (Case 2, RCCA) to 0.48
(Case 3, RHEA), providing the first quantitative evidence that Group VI
solute embrittlement mechanisms are not transferable across alloy families.

References
----------
See paper for full derivation, LOOCV procedure, and CALPHAD validation.
"""

import math

# =============================================================================
# ELEMENTAL CONSTANTS
# =============================================================================

# Atomic weights (g/mol)
AW = {
    "Nb": 92.906, "W": 183.84, "Mo": 95.95,
    "V":  50.942, "Ti": 47.867, "Zr": 91.224,
    "Hf": 178.49, "Ta": 180.95, "Re": 186.21,
    "Al":  26.982, "Cr":  51.996,
}

# Densities (g/cc)
DENSITY = {
    "Nb":  8.57, "W": 19.25, "Mo": 10.22,
    "V":   6.11, "Ti":  4.51, "Zr":  6.52,
    "Hf": 13.31, "Ta": 16.69, "Re": 21.02,
    "Al":  2.70, "Cr":  7.19,
}

# Valence electron concentrations
VEC_TABLE = {
    "Nb": 5, "W": 6, "Mo": 6,
    "V":  5, "Ti": 4, "Zr":  4,
    "Hf": 4, "Ta": 5, "Re":  7,
    "Al": 3, "Cr": 6,
}

# Goldschmidt atomic radii (pm)
RADIUS = {
    "Nb": 146, "W": 139, "Mo": 140,
    "V":  134, "Ti": 147, "Zr": 160,
    "Hf": 159, "Ta": 146, "Re": 137,
    "Al": 143, "Cr": 128,
}

# Melting points (°C)
TM = {
    "Nb": 2477, "W": 3422, "Mo": 2623,
    "V":  1910, "Ti": 1668, "Zr": 1855,
    "Hf": 2233, "Ta": 3017, "Re": 3186,
    "Al":  660, "Cr": 1907,
}

# =============================================================================
# MODEL PARAMETERS  (validated, do not modify without re-running LOOCV)
# =============================================================================

# Case 1 — Independent Alloying Shift (IAS) model
# Operates in wt%. R²=0.856, MAE=15.9°C, n=23.
IAS_BASELINE = -150.0       # Pure Nb DBTT (°C)
IAS_COEFFS = {              # °C per wt% of each element
    "W":   8.0,
    "Mo": 15.0,
    "V":  -5.0,
    "Ti": -2.0,
    "Zr":  1.0,
    "Hf":  0.5,
}

# Case 2 — Sponge Ratio (SR) model
# Operates in at%. LOOCV R²=0.871, MAE=26°C, n=10.
# alpha=0: Mo does NOT enter the SR multiplier in RCCA regime.
SR_kW       = 2.244         # °C / at% W
SR_kMo      = 7.899         # °C / at% Mo
SR_kBuffer  = 1.723         # °C / at% (Hf+Zr+Ti combined)
SR_BASELINE = -150.0        # °C
SR_ALPHA    = 0.0           # Mo weight in SR denominator (RCCA regime)
SR_MAE      = 26.0          # °C (reported uncertainty)

# Case 3 — Embrittlement Index (EI) classifier
# Operates in at%. LOO accuracy=84%, n=25.
# alpha=0.48: Mo contributes at 48% of W's embrittlement weight in RHEA.
EI_ALPHA      = 0.48
EI_DUCTILE    = 0.10        # EI < 0.10  → Ductile Zone
EI_TRANSITION = 0.50        # EI < 0.50  → Transition Zone
EI_BRITTLE    = 15.50       # EI < 15.50 → Brittle Zone (else Confirmed Brittle)

# GVI logistic parameters
GVI_VEC_CENTER    = 5.3     # VEC threshold for BCC stability
GVI_VEC_STEEPNESS = 15.0
GVI_DELTA_CENTER  = 6.5     # δ threshold for phase separation risk
GVI_DELTA_STEEPNESS = 10.0
GVI_THRESHOLD     = 0.5     # Below this → secondary phase risk flag

# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================

# Case 1 limits — wt%
CASE1_LIMITS_WT = {"W": 20, "Mo": 10, "Hf": 10, "Zr": 5, "Ti": 10}

# Case 2 limits — at%
CASE2_LIMITS_AT = {
    "W": 15, "Mo": 18, "Hf": 22.4, "Zr": 8.5, "Ti": 10,
    "Hf+Zr+Ti": 22.4,          # combined buffer limit
}

# Case 3 limits — at%
CASE3_LIMITS_AT = {"W": 20, "Mo": 20, "Hf": 20, "Zr": 33, "Ti": 33}

# CALPHAD false-positive boundary condition (documented from validation)
# If ALL three are simultaneously exceeded, GVI Pass should be verified by CALPHAD.
CALPHAD_TRIGGER = {"Mo": 5.0, "Hf": 5.0, "Zr": 5.0}   # at%

# =============================================================================
# UNIT CONVERSION
# =============================================================================

def atomic_to_weight(comp_at: dict) -> dict:
    """
    Convert atomic percent composition to weight percent.

    Parameters
    ----------
    comp_at : dict
        {element: atomic_percent}, need not sum to 100 (will be normalised).

    Returns
    -------
    dict
        {element: weight_percent}, summing to 100.
    """
    total_at = sum(comp_at.values())
    mass = {el: (pct / total_at) * AW.get(el, 0) for el, pct in comp_at.items()}
    total_mass = sum(mass.values())
    if total_mass == 0:
        return {el: 0.0 for el in comp_at}
    return {el: 100.0 * m / total_mass for el, m in mass.items()}


def weight_to_atomic(comp_wt: dict) -> dict:
    """
    Convert weight percent composition to atomic percent.

    Parameters
    ----------
    comp_wt : dict
        {element: weight_percent}, need not sum to 100 (will be normalised).

    Returns
    -------
    dict
        {element: atomic_percent}, summing to 100.
    """
    total_wt = sum(comp_wt.values())
    moles = {el: (pct / total_wt) / AW.get(el, 1) for el, pct in comp_wt.items()}
    total_moles = sum(moles.values())
    if total_moles == 0:
        return {el: 0.0 for el in comp_wt}
    return {el: 100.0 * m / total_moles for el, m in moles.items()}

# =============================================================================
# DESCRIPTOR UTILITIES
# =============================================================================

def _calc_VEC(comp_at: dict) -> float:
    """Valence Electron Concentration (atomic-fraction weighted)."""
    total = sum(comp_at.values())
    return sum((pct / total) * VEC_TABLE.get(el, 0) for el, pct in comp_at.items())


def _calc_delta(comp_at: dict) -> float:
    """Atomic size mismatch δ (%)."""
    total  = sum(comp_at.values())
    ci     = {el: pct / total for el, pct in comp_at.items()}
    r_bar  = sum(c * RADIUS.get(el, 145) for el, c in ci.items())
    delta2 = sum(c * (1 - RADIUS.get(el, 145) / r_bar) ** 2 for el, c in ci.items())
    return math.sqrt(delta2) * 100


def _calc_density(comp_at: dict) -> float:
    """Rule-of-mixtures density (g/cc)."""
    total = sum(comp_at.values())
    num   = sum((pct / total) * DENSITY.get(el, 10) * AW.get(el, 100) for el, pct in comp_at.items())
    den   = sum((pct / total) * AW.get(el, 100)                        for el, pct in comp_at.items())
    return num / den if den > 0 else 0.0


def _calc_Tm_ROM(comp_at: dict) -> float:
    """Rule-of-mixtures melting point (°C)."""
    total = sum(comp_at.values())
    return sum((pct / total) * TM.get(el, 2000) for el, pct in comp_at.items())

# =============================================================================
# GVI — GLOBAL VIABILITY INDEX
# =============================================================================

def calc_GVI(comp_at: dict, case: int) -> dict:
    """
    Compute the Global Viability Index for phase stability screening.

    GVI uses logistic survival functions on VEC and δ. Case 2 additionally
    includes a Sponge Ratio survival term (S_SR) because high W/Mo relative
    to buffers is a known driver of secondary phase formation in RCCAs.

    Cases 1 and 3:  GVI = S_VEC × S_delta
    Case 2       :  GVI = S_SR  × S_VEC × S_delta

    Parameters
    ----------
    comp_at : dict
        Composition in atomic percent.
    case : int
        Alloy family (1, 2, or 3).

    Returns
    -------
    dict with keys: VEC, delta, GVI, S_VEC, S_delta, [S_SR], flag, calphad_trigger
    """
    vec   = _calc_VEC(comp_at)
    delta = _calc_delta(comp_at)

    s_vec   = 1 / (1 + math.exp(GVI_VEC_STEEPNESS   * (vec   - GVI_VEC_CENTER)))
    s_delta = 1 / (1 + math.exp(GVI_DELTA_STEEPNESS * (delta - GVI_DELTA_CENTER)))

    result = {"VEC": round(vec, 3), "delta": round(delta, 3),
              "S_VEC": round(s_vec, 4), "S_delta": round(s_delta, 4)}

    if case == 2:
        # SR_W uses weight percent
        comp_wt = atomic_to_weight(comp_at)
        W_wt    = comp_wt.get("W",  0)
        Mo_wt   = comp_wt.get("Mo", 0)
        Hf_wt   = comp_wt.get("Hf", 0)
        Zr_wt   = comp_wt.get("Zr", 0)
        Ti_wt   = comp_wt.get("Ti", 0)
        sr_w    = (W_wt + 0.077 * Mo_wt) / (Hf_wt + Zr_wt + Ti_wt + 1)
        s_sr    = 1 / (1 + math.exp(8 * (sr_w - 1.5)))
        gvi     = s_sr * s_vec * s_delta
        result.update({"S_SR": round(s_sr, 4), "SR_W": round(sr_w, 4)})
    else:
        gvi = s_vec * s_delta

    result["GVI"] = round(gvi, 4)
    result["flag"] = "PASS" if gvi >= GVI_THRESHOLD else "SECONDARY PHASE RISK — verify with CALPHAD"

    # Check documented CALPHAD false-positive trigger
    Mo_at = comp_at.get("Mo", 0)
    Hf_at = comp_at.get("Hf", 0)
    Zr_at = comp_at.get("Zr", 0)
    calphad_trigger = (Mo_at > CALPHAD_TRIGGER["Mo"] and
                       Hf_at > CALPHAD_TRIGGER["Hf"] and
                       Zr_at > CALPHAD_TRIGGER["Zr"])
    result["calphad_trigger"] = calphad_trigger
    if calphad_trigger:
        result["flag"] += " | Mo>5%, Hf>5%, Zr>5% simultaneously — CALPHAD verification required"

    return result

# =============================================================================
# CASE CLASSIFICATION & BOUNDARY CHECKS
# =============================================================================

def identify_case(comp_at: dict) -> tuple:
    """
    Identify which URADES case applies to a given composition.

    Classification hierarchy:
      1. Convert at% to wt% and check Nb content in wt%.
         If Nb ≥ 79 wt% → Case 1 (dilute engineering alloy, IAS model).
      2. Check Nb content in at%.
         If Nb ≥ 50 at% → Case 2 (Nb-matrix RCCA, SR model).
      3. Otherwise → Case 3 (multi-principal RHEA, EI classifier).

    Parameters
    ----------
    comp_at : dict
        Composition in atomic percent.

    Returns
    -------
    (case, Nb_wt_pct, Nb_at_pct)
    """
    comp_wt  = atomic_to_weight(comp_at)
    Nb_wt    = comp_wt.get("Nb", 0)
    total_at = sum(comp_at.values())
    Nb_at    = comp_at.get("Nb", 0) / total_at * 100

    if Nb_wt >= 79.0:
        return 1, Nb_wt, Nb_at
    elif Nb_at >= 50.0:
        return 2, Nb_wt, Nb_at
    else:
        return 3, Nb_wt, Nb_at


def check_boundary_conditions(comp_at: dict, case: int) -> tuple:
    """
    Enforce element concentration limits for each case.

    Case 1 limits are in wt% (Case 1 operates in wt%).
    Case 2 and 3 limits are in at%.

    Parameters
    ----------
    comp_at : dict
        Composition in atomic percent.
    case : int
        Alloy family (1, 2, or 3).

    Returns
    -------
    (passed: bool, violations: list of str)
    """
    violations = []

    if case == 1:
        comp_wt = atomic_to_weight(comp_at)
        for el, limit in CASE1_LIMITS_WT.items():
            val = comp_wt.get(el, 0)
            if val > limit:
                violations.append(f"{el} = {val:.1f} wt% exceeds Case 1 limit of {limit} wt%")

    elif case == 2:
        for el, limit in CASE2_LIMITS_AT.items():
            if el == "Hf+Zr+Ti":
                val = comp_at.get("Hf", 0) + comp_at.get("Zr", 0) + comp_at.get("Ti", 0)
                if val > limit:
                    violations.append(f"Hf+Zr+Ti = {val:.1f} at% exceeds Case 2 combined buffer limit of {limit} at%")
            else:
                val = comp_at.get(el, 0)
                if val > limit:
                    violations.append(f"{el} = {val:.1f} at% exceeds Case 2 limit of {limit} at%")

    elif case == 3:
        for el, limit in CASE3_LIMITS_AT.items():
            val = comp_at.get(el, 0)
            if val > limit:
                violations.append(f"{el} = {val:.1f} at% exceeds Case 3 limit of {limit} at%")

    return (len(violations) == 0), violations

# =============================================================================
# CASE 1 — IAS MODEL
# =============================================================================

def predict_case1(comp_at: dict) -> dict:
    """
    Independent Alloying Shift (IAS) model for dilute Nb engineering alloys.

    DBTT = -150 + Σ(k_i × C_i)

    where C_i is the concentration of element i in wt%, and k_i is the
    embrittlement coefficient (°C/wt%) fitted to n=23 literature alloys.

    Coefficients (°C/wt%):
        W=+8, Mo=+15, V=−5, Ti=−2, Zr=+1, Hf=+0.5

    Parameters
    ----------
    comp_at : dict
        Composition in atomic percent.

    Returns
    -------
    dict with keys: DBTT, YS, Tm, density, wt_comp
    """
    comp_wt = atomic_to_weight(comp_at)

    dbtt = IAS_BASELINE
    for el, k in IAS_COEFFS.items():
        dbtt += k * comp_wt.get(el, 0)

    # Yield strength (solid-solution strengthening, empirical)
    ys_base = 150.0
    ys = (ys_base
          + 15.0 * comp_wt.get("W",  0)
          + 25.0 * comp_wt.get("Mo", 0)
          + 15.0 * comp_wt.get("Zr", 0)
          +  8.0 * comp_wt.get("Hf", 0)
          +  5.0 * comp_wt.get("Ti", 0))

    return {
        "DBTT":      round(dbtt, 1),
        "DBTT_range": f"{round(dbtt - 15.9, 1)} to {round(dbtt + 15.9, 1)} °C",
        "YS_MPa":    round(ys, 0),
        "Tm_C":      round(_calc_Tm_ROM(comp_at), 0),
        "density":   round(_calc_density(comp_at), 3),
        "wt_comp":   {el: round(v, 2) for el, v in comp_wt.items() if v > 0},
    }

# =============================================================================
# CASE 2 — SR MODEL
# =============================================================================

def predict_case2(comp_at: dict) -> dict:
    """
    Sponge Ratio (SR) model for Nb-matrix RCCAs.

    The SR model uses a 3-parameter linear-times-multiplicative form:

        dT   = kW·W + kMo·Mo + kBuffer·(Hf+Zr+Ti)
        SR   = W / (Hf + Zr + Ti + 1)          [alpha=0: Mo absent from SR]
        DBTT = -150 + dT × (1 + SR)

    The key result is alpha=0: Mo does not enter the SR multiplier in the
    RCCA regime. This contrasts with Case 3 where alpha=0.48.

    All concentrations in at%.
    LOOCV R²=0.871, MAE=26°C, n=10.

    Parameters
    ----------
    comp_at : dict
        Composition in atomic percent.

    Returns
    -------
    dict with keys: DBTT, SR, dT, YS, Tm, density
    """
    W      = comp_at.get("W",  0)
    Mo     = comp_at.get("Mo", 0)
    Hf     = comp_at.get("Hf", 0)
    Zr     = comp_at.get("Zr", 0)
    Ti     = comp_at.get("Ti", 0)
    Buffer = Hf + Zr + Ti

    dT   = SR_kW * W + SR_kMo * Mo + SR_kBuffer * Buffer
    SR   = W / (Buffer + 1)
    dbtt = SR_BASELINE + dT * (1 + SR)

    # Equivalent solute concentration for YS and Tm correction
    C_eq = 3*Mo + 2*W + 1*Zr + 0.5*(Hf + Ti)
    ys   = 150.0 + 100.0 * (C_eq ** 0.5)
    tm   = _calc_Tm_ROM(comp_at) - 2.97 * C_eq

    return {
        "DBTT":       round(dbtt, 1),
        "DBTT_range": f"{round(dbtt - SR_MAE, 1)} to {round(dbtt + SR_MAE, 1)} °C",
        "SR":         round(SR, 4),
        "dT":         round(dT, 2),
        "alpha":      SR_ALPHA,
        "YS_MPa":     round(ys, 0),
        "Tm_C":       round(tm, 0),
        "density":    round(_calc_density(comp_at), 3),
    }

# =============================================================================
# CASE 3 — EI CLASSIFIER
# =============================================================================

def predict_case3(comp_at: dict) -> dict:
    """
    Embrittlement Index (EI) classifier for Nb-based RHEAs.

    EI = (W + alpha·Mo) / (Hf + Zr + Ti + 1)    [alpha = 0.48]

    Four-zone classification:
        EI < 0.10  → Ductile Zone
        EI < 0.50  → Transition Zone
        EI < 15.5  → Brittle Zone
        EI ≥ 15.5  → Confirmed Brittle Zone

    The alpha=0.48 value is the central novel result of URADES: Mo
    contributes at 48% of W's embrittlement weight in the RHEA regime,
    whereas alpha=0 in Case 2. This reflects the changing role of Mo
    with increasing compositional complexity.

    LOO accuracy=84%, n=25.

    Parameters
    ----------
    comp_at : dict
        Composition in atomic percent.

    Returns
    -------
    dict with keys: EI, zone, alpha, Tm, density
    """
    W      = comp_at.get("W",  0)
    Mo     = comp_at.get("Mo", 0)
    Hf     = comp_at.get("Hf", 0)
    Zr     = comp_at.get("Zr", 0)
    Ti     = comp_at.get("Ti", 0)
    Buffer = Hf + Zr + Ti

    EI = (W + EI_ALPHA * Mo) / (Buffer + 1)

    if EI < EI_DUCTILE:
        zone = "Ductile Zone"
    elif EI < EI_TRANSITION:
        zone = "Transition Zone"
    elif EI < EI_BRITTLE:
        zone = "Brittle Zone"
    else:
        zone = "Confirmed Brittle Zone"

    return {
        "EI":      round(EI, 4),
        "zone":    zone,
        "alpha":   EI_ALPHA,
        "Tm_C":    round(_calc_Tm_ROM(comp_at), 0),
        "density": round(_calc_density(comp_at), 3),
    }

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_URADES(composition: dict, input_unit: str = "at",
               gvi_gate: bool = True, verbose: bool = True) -> dict:
    """
    Run the full URADES pipeline on a composition.

    Pipeline:
        1. Identify case (1/2/3) from Nb content
        2. Check boundary conditions — hard reject if violated
        3. Compute GVI — flag if below threshold (not hard reject)
        4. Run the appropriate property model
        5. Return structured result dict

    Parameters
    ----------
    composition : dict
        Alloy composition as {element: percent}.

    input_unit : str
        "at" (default) — composition given in atomic percent.
        "wt"           — composition given in weight percent.
        Case 1 model operates in wt%; Cases 2 and 3 operate in at%.
        Conversion is handled internally regardless of input_unit.

    gvi_gate : bool
        If True (default), include GVI screening.

    verbose : bool
        If True, print a formatted result to stdout.

    Returns
    -------
    dict
        Full result. "status" is "OK", "FLAGGED" (GVI below threshold),
        or "REJECTED" (boundary condition violated).
    """
    # Ensure we have both unit representations internally
    if input_unit == "wt":
        comp_wt = composition
        comp_at = weight_to_atomic(composition)
    else:
        comp_at = composition
        comp_wt = atomic_to_weight(composition)

    # Step 1: classify (always uses at% internally)
    case, Nb_wt, Nb_at = identify_case(comp_at)

    # Step 2: boundary conditions (always passes comp_at; function handles unit internally)
    passed, violations = check_boundary_conditions(comp_at, case)
    if not passed:
        result = {
            "status":     "REJECTED",
            "case":       case,
            "Nb_wt_pct":  round(Nb_wt, 2),
            "Nb_at_pct":  round(Nb_at, 2),
            "violations": violations,
        }
        if verbose:
            _print_result(result, composition)
        return result

    # Step 3: GVI
    gvi_result = calc_GVI(comp_at, case) if gvi_gate else {"GVI": None, "flag": "GVI gate bypassed"}

    # Step 4: property model
    # Case 1 operates internally in wt%; predict_case1 handles conversion.
    # Cases 2 and 3 operate in at%.
    case_labels = {1: "Case 1 — Dilute Nb Alloy (IAS)",
                   2: "Case 2 — Nb-Matrix RCCA (SR)",
                   3: "Case 3 — Nb-Based RHEA (EI)"}

    if case == 1:
        props = predict_case1(comp_at)   # converts to wt% internally
    elif case == 2:
        props = predict_case2(comp_at)
    else:
        props = predict_case3(comp_at)

    # Step 5: assemble result
    gvi_val = gvi_result.get("GVI")
    status  = "FLAGGED" if (gvi_val is not None and gvi_val < GVI_THRESHOLD) else "OK"

    result = {
        "status":       status,
        "case":         case,
        "case_label":   case_labels[case],
        "Nb_wt_pct":    round(Nb_wt, 2),
        "Nb_at_pct":    round(Nb_at, 2),
        "GVI":          gvi_result,
        "properties":   props,
    }

    if verbose:
        _print_result(result, composition)

    return result


def _print_result(result: dict, comp_at: dict) -> None:
    """Pretty-print a URADES result to stdout."""
    sep = "─" * 60

    print(f"\n{sep}")
    print("URADES — Unified Refractory Alloy Descriptor & Embrittlement Screener")
    print(sep)

    # Composition
    comp_str = "  ".join(f"{el}{v:.1f}" for el, v in comp_at.items() if v > 0)
    print(f"Composition (at%):  {comp_str}")
    print(f"Status:             {result['status']}")

    if result["status"] == "REJECTED":
        print(f"\nREJECTED — boundary condition violations:")
        for v in result["violations"]:
            print(f"  • {v}")
        print(sep)
        return

    print(f"Case:               {result.get('case_label', result['case'])}")
    print(f"Nb content:         {result['Nb_at_pct']:.1f} at%  /  {result['Nb_wt_pct']:.1f} wt%")

    gvi = result.get("GVI", {})
    if gvi and gvi.get("GVI") is not None:
        print(f"\nPhase Stability (GVI)")
        print(f"  GVI   = {gvi['GVI']:.4f}  [{gvi['flag']}]")
        print(f"  VEC   = {gvi['VEC']:.3f}   δ = {gvi['delta']:.3f}%")
        if gvi.get("calphad_trigger"):
            print("  ⚠  Mo>5, Hf>5, Zr>5 at% simultaneously — CALPHAD verification recommended")

    props = result.get("properties", {})
    if props:
        print(f"\nProperty Predictions")
        if "DBTT" in props:
            print(f"  DBTT  = {props['DBTT']:+.1f} °C   [{props.get('DBTT_range', '')}]")
        if "SR" in props:
            print(f"  SR    = {props['SR']:.4f}   (α = {props['alpha']})")
        if "EI" in props:
            print(f"  EI    = {props['EI']:.4f}   (α = {props['alpha']})")
            print(f"  Zone  = {props['zone']}")
        if "YS_MPa" in props:
            print(f"  YS    ≈ {props['YS_MPa']:.0f} MPa")
        if "Tm_C" in props:
            print(f"  Tm    ≈ {props['Tm_C']:.0f} °C")
        if "density" in props:
            print(f"  ρ     ≈ {props['density']:.3f} g/cc")

    print(sep)
