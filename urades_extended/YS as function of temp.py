"""
URADES EXTENDED — Temperature-Dependent Yield Strength
========================================================

Extended analytical module for URADES high-temperature yield-strength
screening of Nb-based refractory alloys.

This module implements the equations documented in the URADES extended
temperature-variation study:

    Case 1 — Dilute Nb engineering alloys
        Nb > 79 wt%
        Temperature range: 20–1500 °C
        Sponge-Ratio coupling valve formulation

    Case 3 — Concentrated refractory high-entropy alloys
        Equiatomic / near-equiatomic BCC RHEAs
        Temperature range: 24–1400 °C
        VEC + H_mix athermal strength with G_mix-controlled
        thermal efficiency

IMPORTANT
---------
This file reproduces the documented extended formulations. It does not
invent elemental descriptor constants that were not specified in the
extended formulation document.

For Case 3, VEC, G_mix, T_m and H_mix may therefore be supplied directly
from the documented dataset, or calculated externally from the appropriate
elemental-property databases.

Case 3 predicts high-temperature YS, unlike the main URADES Case 3 EI
model, which is an embrittlement-risk classifier.

Dependencies
------------
numpy
pandas
matplotlib
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CASE 1 — TEMPERATURE-DEPENDENT YIELD STRENGTH
# =============================================================================

def case1_sponge_ratio(
    composition_wt: Mapping[str, float],
) -> float:
    """
    Calculate the Case-1 Sponge Ratio.

    SR = (W + Mo) / (Zr + Hf + Ti + Ta)

    The extended formulation defines the ratio using wt% concentrations.

    If the denominator is zero, SR is mathematically undefined. The
    function returns infinity, corresponding to no available buffer.
    """

    W = float(composition_wt.get("W", 0.0))
    Mo = float(composition_wt.get("Mo", 0.0))
    Zr = float(composition_wt.get("Zr", 0.0))
    Hf = float(composition_wt.get("Hf", 0.0))
    Ti = float(composition_wt.get("Ti", 0.0))
    Ta = float(composition_wt.get("Ta", 0.0))

    denominator = Zr + Hf + Ti + Ta

    if denominator == 0:
        return math.inf

    return (W + Mo) / denominator


def case1_phi_sr(
    sr: float,
) -> float:
    """
    High-temperature Sponge-Ratio efficiency.

    Phi_SR = 1 / [1 + exp(1.026 * (SR - 7.0))]
    """

    if math.isinf(sr):
        return 0.0

    exponent = 1.026 * (sr - 7.0)

    # Numerically stable logistic evaluation.
    if exponent >= 0:
        return math.exp(-exponent) / (1.0 + math.exp(-exponent))

    return 1.0 / (1.0 + math.exp(exponent))


def case1_activation(
    temperature_c: float,
) -> float:
    """
    Thermal activation term.

    A(T) = 1 / [1 + exp(-0.008 * (T - 400))]
    """

    exponent = -0.008 * (temperature_c - 400.0)

    if exponent >= 0:
        e = math.exp(-exponent)
        return e / (1.0 + e)

    return 1.0 / (1.0 + math.exp(exponent))


def case1_coupling_valve(
    sr: float,
    temperature_c: float,
) -> float:
    """
    Sponge-Ratio / temperature coupling valve.

    Phi(SR,T) = 1 - [(1 - Phi_SR) * A(T)]
    """

    phi_sr = case1_phi_sr(sr)
    activation = case1_activation(temperature_c)

    return 1.0 - ((1.0 - phi_sr) * activation)


def alpha_nb(
    temperature_c: float,
) -> float:
    """
    Temperature-dependent Nb coefficient, MPa per wt%.
    """

    T = float(temperature_c)

    if T < 200.0:
        return 0.90 - 0.00097 * (T - 20.0)

    return 0.835 - 0.00055 * T


def alpha_zr(
    temperature_c: float,
) -> float:
    """
    Temperature-dependent Zr coefficient, MPa per wt%.
    """

    T = float(temperature_c)

    return 17.0 + 68.9 / (
        1.0 + math.exp(0.033 * (T - 700.0))
    )


def alpha_c(
    temperature_c: float,
) -> float:
    """
    Temperature-dependent C coefficient, MPa per wt%.
    """

    T = float(temperature_c)

    return 1011.0 - 0.507 * T


def alpha_w0(
    temperature_c: float,
) -> float:
    """
    Uncoupled W coefficient, MPa per wt%.
    """

    T = float(temperature_c)

    if T <= 1000.0:
        return 13.8 + 0.00418 * (T - 20.0)

    return 17.9 - 0.0294 * (T - 1000.0)


def alpha_mo0(
    temperature_c: float,
) -> float:
    """
    Uncoupled Mo coefficient, MPa per wt%.
    """

    T = float(temperature_c)

    return 68.65 + 0.0421 * (T - 20.0)


def alpha_v(
    temperature_c: float,
) -> float:
    """
    V weakening coefficient, MPa per wt%.
    """

    T = float(temperature_c)

    return -27.1 - 0.0315 * (T - 20.0)


def case1_ys(
    composition_wt: Mapping[str, float],
    temperature_c: float,
) -> float:
    """
    Predict Case-1 yield strength.

    YS(T)
      = Base(T)
      + Phi(SR,T) * Aggressors(T)
      + Weakeners(T)

    Base(T)
      = Nb*alpha_Nb + Zr*alpha_Zr + C*alpha_C

    Aggressors(T)
      = W*alpha_W0 + Mo*alpha_Mo0

    Weakeners(T)
      = V*alpha_V
    """

    Nb = float(composition_wt.get("Nb", 0.0))
    Zr = float(composition_wt.get("Zr", 0.0))
    C = float(composition_wt.get("C", 0.0))
    W = float(composition_wt.get("W", 0.0))
    Mo = float(composition_wt.get("Mo", 0.0))
    V = float(composition_wt.get("V", 0.0))

    sr = case1_sponge_ratio(composition_wt)
    phi = case1_coupling_valve(
        sr,
        temperature_c,
    )

    base = (
        Nb * alpha_nb(temperature_c)
        + Zr * alpha_zr(temperature_c)
        + C * alpha_c(temperature_c)
    )

    aggressors = (
        W * alpha_w0(temperature_c)
        + Mo * alpha_mo0(temperature_c)
    )

    weakeners = (
        V * alpha_v(temperature_c)
    )

    return (
        base
        + phi * aggressors
        + weakeners
    )


def case1_profile(
    composition_wt: Mapping[str, float],
    temperatures_c: Sequence[float],
) -> pd.DataFrame:
    """
    Generate a Case-1 temperature/YS profile.
    """

    rows = []

    sr = case1_sponge_ratio(
        composition_wt
    )

    for temperature in temperatures_c:

        rows.append(
            {
                "Temperature_C": float(temperature),
                "SR": sr,
                "Phi_SR": case1_phi_sr(sr),
                "A_T": case1_activation(temperature),
                "Phi": case1_coupling_valve(
                    sr,
                    temperature,
                ),
                "YS_MPa": case1_ys(
                    composition_wt,
                    temperature,
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# CASE 3 — TEMPERATURE-DEPENDENT RHEA YIELD STRENGTH
# =============================================================================

def case3_athermal_strength(
    vec: float,
    h_mix_kj_mol: float,
) -> float:
    """
    Athermal strength anchor.

    YS_athermal = 265.2*VEC + 55.5*|H_mix|

    H_mix is supplied in kJ/mol as documented.
    """

    return (
        265.2 * float(vec)
        + 55.5 * abs(float(h_mix_kj_mol))
    )


def case3_p(
    g_mix_gpa: float,
) -> float:
    """
    Thermal exponent p.

    p = 1.410 + 0.021*G_mix
    """

    return (
        1.410
        + 0.021 * float(g_mix_gpa)
    )


def case3_q(
    g_mix_gpa: float,
) -> float:
    """
    Thermal exponent q.

    q = 7.944 - 0.067*G_mix
    """

    return (
        7.944
        - 0.067 * float(g_mix_gpa)
    )


def case3_thermal_efficiency(
    temperature_c: float,
    melting_temperature_k: float,
    g_mix_gpa: float,
) -> float:
    """
    Calculate the Case-3 thermal efficiency.

    Psi(T,G_mix)
      = [1 - (T/Tm)^p]^q

    Temperature is supplied in °C.
    Melting temperature is supplied in K.
    """

    T_K = float(temperature_c) + 273.15
    Tm_K = float(melting_temperature_k)

    if Tm_K <= 0:
        raise ValueError(
            "Melting temperature must be positive."
        )

    homologous_temperature = T_K / Tm_K

    if homologous_temperature < 0:
        raise ValueError(
            "Homologous temperature cannot be negative."
        )

    p = case3_p(g_mix_gpa)
    q = case3_q(g_mix_gpa)

    if homologous_temperature >= 1.0:
        return 0.0

    value = 1.0 - (
        homologous_temperature ** p
    )

    return max(
        0.0,
        value ** q,
    )


def case3_ys(
    temperature_c: float,
    vec: float,
    h_mix_kj_mol: float,
    g_mix_gpa: float,
    melting_temperature_k: float,
) -> float:
    """
    Predict Case-3 high-temperature yield strength.

    YS(T)
      = YS_athermal * Psi(T/Tm,G_mix)
    """

    anchor = case3_athermal_strength(
        vec,
        h_mix_kj_mol,
    )

    efficiency = case3_thermal_efficiency(
        temperature_c,
        melting_temperature_k,
        g_mix_gpa,
    )

    return anchor * efficiency


def case3_profile(
    vec: float,
    h_mix_kj_mol: float,
    g_mix_gpa: float,
    melting_temperature_k: float,
    temperatures_c: Sequence[float],
) -> pd.DataFrame:
    """
    Generate a Case-3 temperature/YS profile.
    """

    anchor = case3_athermal_strength(
        vec,
        h_mix_kj_mol,
    )

    p = case3_p(g_mix_gpa)
    q = case3_q(g_mix_gpa)

    rows = []

    for temperature in temperatures_c:

        T_K = (
            float(temperature)
            + 273.15
        )

        homologous_temperature = (
            T_K / melting_temperature_k
        )

        psi = case3_thermal_efficiency(
            temperature,
            melting_temperature_k,
            g_mix_gpa,
        )

        rows.append(
            {
                "Temperature_C": float(
                    temperature
                ),
                "T_over_Tm": homologous_temperature,
                "G_mix_GPa": g_mix_gpa,
                "p": p,
                "q": q,
                "Psi": psi,
                "YS_athermal_MPa": anchor,
                "YS_MPa": anchor * psi,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# CASE 3 DESCRIPTOR UTILITIES
# =============================================================================

def calculate_vec(
    atomic_fractions: Mapping[str, float],
    vec_values: Mapping[str, float],
) -> float:
    """
    Calculate VEC from atomic fractions and elemental VEC values.

    This uses the rule of mixtures documented for the extended model.
    """

    total = sum(
        float(value)
        for value in atomic_fractions.values()
    )

    if total <= 0:
        raise ValueError(
            "Atomic fractions must contain a positive total."
        )

    return sum(
        (
            float(c) / total
        )
        * float(vec_values[element])
        for element, c
        in atomic_fractions.items()
    )


def calculate_gmix(
    atomic_fractions: Mapping[str, float],
    shear_moduli_gpa: Mapping[str, float],
) -> float:
    """
    Calculate G_mix using the documented rule of mixtures.
    """

    total = sum(
        float(value)
        for value in atomic_fractions.values()
    )

    if total <= 0:
        raise ValueError(
            "Atomic fractions must contain a positive total."
        )

    return sum(
        (
            float(c) / total
        )
        * float(shear_moduli_gpa[element])
        for element, c
        in atomic_fractions.items()
    )


def calculate_tm_rom(
    atomic_fractions: Mapping[str, float],
    melting_points_k: Mapping[str, float],
) -> float:
    """
    Calculate T_m by the documented rule of mixtures.
    """

    total = sum(
        float(value)
        for value in atomic_fractions.values()
    )

    if total <= 0:
        raise ValueError(
            "Atomic fractions must contain a positive total."
        )

    return sum(
        (
            float(c) / total
        )
        * float(melting_points_k[element])
        for element, c
        in atomic_fractions.items()
    )


def calculate_hmix(
    atomic_fractions: Mapping[str, float],
    interaction_enthalpy_kj_mol: Mapping[tuple, float],
) -> float:
    """
    Calculate H_mix using the documented Takeuchi-Inoue form:

        H_mix = sum_{i != j} 4*c_i*c_j*DeltaH_ij

    Parameters
    ----------
    atomic_fractions:
        Element -> atomic fraction.

    interaction_enthalpy_kj_mol:
        Dictionary keyed by (element_i, element_j).

    Notes
    -----
    The extended formulation specifies the equation but the full
    elemental interaction matrix is not contained in the temperature-
    variation document. Therefore the matrix must be supplied by the
    caller rather than invented here.
    """

    elements = list(
        atomic_fractions.keys()
    )

    total = sum(
        float(value)
        for value in atomic_fractions.values()
    )

    if total <= 0:
        raise ValueError(
            "Atomic fractions must contain a positive total."
        )

    c = {
        element: float(value) / total
        for element, value
        in atomic_fractions.items()
    }

    h_mix = 0.0

    for i in elements:

        for j in elements:

            if i == j:
                continue

            key = (i, j)

            if key not in interaction_enthalpy_kj_mol:

                reverse_key = (j, i)

                if reverse_key in interaction_enthalpy_kj_mol:
                    key = reverse_key
                else:
                    raise KeyError(
                        f"Missing interaction enthalpy for {i}-{j}."
                    )

            h_mix += (
                4.0
                * c[i]
                * c[j]
                * float(
                    interaction_enthalpy_kj_mol[key]
                )
            )

    return h_mix


# =============================================================================
# DOCUMENTED CASE 1 DATASET
# =============================================================================

CASE1_DATA = [
    {
        "Alloy": "Nb-1Zr",
        "Composition": {"Nb": 99.0, "Zr": 1.0},
        "SR": 0.0,
        "Points": [
            (20, 175),
            (600, 130),
            (1000, 45),
            (1200, 35),
        ],
    },
    {
        "Alloy": "PWC-11",
        "Composition": {
            "Nb": 98.9,
            "Zr": 1.0,
            "C": 0.1,
        },
        "SR": 0.0,
        "Points": [
            (20, 275),
            (1000, 86),
            (1100, 85),
        ],
    },
    {
        "Alloy": "D-43",
        "Composition": {
            "Nb": 88.9,
            "W": 10.0,
            "Zr": 1.0,
            "C": 0.1,
        },
        "SR": 10.0,
        "Points": [
            (20, 400),
            (538, 230),
            (1000, 100),
            (1204, 78),
            (1316, 50),
        ],
    },
    {
        "Alloy": "Cb-752",
        "Composition": {
            "Nb": 87.5,
            "W": 10.0,
            "Zr": 2.5,
        },
        "SR": 4.0,
        "Points": [
            (20, 435),
            (1000, 240),
            (1204, 172),
        ],
    },
    {
        "Alloy": "Nb-521",
        "Composition": {
            "Nb": 92.0,
            "W": 5.0,
            "Mo": 2.0,
            "Zr": 1.0,
        },
        "SR": 7.0,
        "Points": [
            (20, 375),
            (1000, 220),
            (1500, 129),
        ],
    },
    {
        "Alloy": "B-66",
        "Composition": {
            "Nb": 89.0,
            "Mo": 5.0,
            "V": 5.0,
            "Zr": 1.0,
        },
        "SR": None,
        "Points": [
            (20, 372),
            (1200, 230),
            (1500, 106),
        ],
    },
]


def case1_validation_table() -> pd.DataFrame:
    """
    Build the documented Case-1 experimental/predicted comparison table.
    """

    rows = []

    for alloy in CASE1_DATA:

        composition = alloy["Composition"]

        for temperature, experimental in alloy["Points"]:

            predicted = case1_ys(
                composition,
                temperature,
            )

            if (
                alloy["Alloy"] == "D-43"
                and temperature == 20
            ):
                experimental_display = "400–500"
                error = np.nan

            elif (
                alloy["Alloy"] == "D-43"
                and temperature == 538
            ):
                experimental_display = "230–280"
                error = np.nan

            else:
                experimental_display = experimental
                error = abs(
                    float(experimental)
                    - predicted
                )

            rows.append(
                {
                    "Alloy": alloy["Alloy"],
                    "SR": alloy["SR"],
                    "Temperature_C": temperature,
                    "Experimental_Y S_MPa": experimental_display,
                    "Predicted_YS_MPa": predicted,
                    "Absolute_Error_MPa": error,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# DOCUMENTED CASE 3 DATASET
# =============================================================================

CASE3_DATA = [
    {
        "Alloy": "HfMoNbTaTiZr",
        "VEC": 4.67,
        "G_mix_GPa": 56.7,
        "Tm_K": 2585,
        "H_mix_kJ_mol": -3.00,
        "Points": [
            (24, 1512),
            (800, 1007),
            (1000, 814),
            (1200, 556),
        ],
    },
    {
        "Alloy": "HfNbTaTiZr",
        "VEC": 4.40,
        "G_mix_GPa": 42.8,
        "Tm_K": 2523,
        "H_mix_kJ_mol": 2.72,
        "Points": [
            (24, 929),
            (800, 535),
            (1000, 295),
            (1200, 92),
        ],
    },
    {
        "Alloy": "HfMoNbTiZr",
        "VEC": 4.60,
        "G_mix_GPa": 54.2,
        "Tm_K": 2444,
        "H_mix_kJ_mol": -4.64,
        "Points": [
            (24, 1351),
            (800, 829),
            (1000, 721),
            (1200, 301),
        ],
    },
    {
        "Alloy": "HfMoNbTaZr",
        "VEC": 4.80,
        "G_mix_GPa": 59.2,
        "Tm_K": 2714,
        "H_mix_kJ_mol": -4.16,
        "Points": [
            (24, 1524),
            (800, 1005),
            (1000, 927),
            (1200, 694),
            (1400, 278),
        ],
    },
    {
        "Alloy": "HfMoNbTaTi",
        "VEC": 4.80,
        "G_mix_GPa": 61.4,
        "Tm_K": 2677,
        "H_mix_kJ_mol": -3.68,
        "Points": [
            (24, 1369),
            (800, 822),
            (1000, 778),
            (1200, 699),
            (1400, 367),
        ],
    },
    {
        "Alloy": "NbTiZr",
        "VEC": 4.33,
        "G_mix_GPa": 38.3,
        "Tm_K": 2273,
        "H_mix_kJ_mol": 2.67,
        "Points": [
            (24, 975),
            (1000, 200),
        ],
    },
    {
        "Alloy": "NbTaTiZr",
        "VEC": 4.50,
        "G_mix_GPa": 46.0,
        "Tm_K": 2527,
        "H_mix_kJ_mol": 2.50,
        "Points": [
            (24, 1000),
            (800, 400),
        ],
    },
    {
        "Alloy": "NbTaTiV",
        "VEC": 4.75,
        "G_mix_GPa": 49.5,
        "Tm_K": 2541,
        "H_mix_kJ_mol": -0.25,
        "Points": [
            (24, 1370),
            (800, 700),
            (1000, 437),
        ],
    },
]


def case3_validation_table() -> pd.DataFrame:
    """
    Build the documented Case-3 experimental/predicted comparison table.
    """

    rows = []

    for alloy in CASE3_DATA:

        for temperature, experimental in alloy["Points"]:

            predicted = case3_ys(
                temperature_c=temperature,
                vec=alloy["VEC"],
                h_mix_kj_mol=alloy["H_mix_kJ_mol"],
                g_mix_gpa=alloy["G_mix_GPa"],
                melting_temperature_k=alloy["Tm_K"],
            )

            rows.append(
                {
                    "Alloy": alloy["Alloy"],
                    "Temperature_C": temperature,
                    "Experimental_YS_MPa": experimental,
                    "Predicted_YS_MPa": predicted,
                    "Absolute_Error_MPa": abs(
                        experimental - predicted
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# METRICS
# =============================================================================

def mae(
    experimental: Sequence[float],
    predicted: Sequence[float],
) -> float:
    """Mean absolute error."""

    y = np.asarray(
        experimental,
        dtype=float,
    )

    yp = np.asarray(
        predicted,
        dtype=float,
    )

    return float(
        np.mean(
            np.abs(y - yp)
        )
    )


def rmse(
    experimental: Sequence[float],
    predicted: Sequence[float],
) -> float:
    """Root mean squared error."""

    y = np.asarray(
        experimental,
        dtype=float,
    )

    yp = np.asarray(
        predicted,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                (y - yp) ** 2
            )
        )
    )


# =============================================================================
# PLOTTING
# =============================================================================

def plot_case1_profile(
    composition_wt: Mapping[str, float],
    temperatures_c: Sequence[float],
    experimental: Optional[pd.DataFrame] = None,
):
    """
    Plot Case-1 predicted YS against temperature.
    """

    profile = case1_profile(
        composition_wt,
        temperatures_c,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        profile["Temperature_C"],
        profile["YS_MPa"],
        linewidth=2,
        label="URADES prediction",
    )

    if experimental is not None:

        ax.scatter(
            experimental["Temperature_C"],
            experimental["Experimental_Y S_MPa"],
            s=35,
            label="Experimental",
        )

    ax.set_xlabel(
        "Temperature (°C)"
    )

    ax.set_ylabel(
        "Yield Strength (MPa)"
    )

    ax.set_title(
        "Case 1 — Temperature-Dependent Yield Strength"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    return fig


def plot_case3_profile(
    vec: float,
    h_mix_kj_mol: float,
    g_mix_gpa: float,
    melting_temperature_k: float,
    temperatures_c: Sequence[float],
):
    """
    Plot Case-3 predicted YS against temperature.
    """

    profile = case3_profile(
        vec=vec,
        h_mix_kj_mol=h_mix_kj_mol,
        g_mix_gpa=g_mix_gpa,
        melting_temperature_k=melting_temperature_k,
        temperatures_c=temperatures_c,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        profile["Temperature_C"],
        profile["YS_MPa"],
        linewidth=2,
        label="URADES prediction",
    )

    ax.set_xlabel(
        "Temperature (°C)"
    )

    ax.set_ylabel(
        "Yield Strength (MPa)"
    )

    ax.set_title(
        "Case 3 — Temperature-Dependent Yield Strength"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    return fig


# =============================================================================
# VALIDATION
# =============================================================================

def validate_case1() -> dict:
    """
    Recalculate Case-1 validation statistics from the documented table.

    The D-43 measurements reported as ranges are excluded from numerical
    MAE/RMSE calculation, exactly because they are ranges rather than
    single experimental values.
    """

    rows = []

    for alloy in CASE1_DATA:

        for temperature, experimental in alloy["Points"]:

            if (
                alloy["Alloy"] == "D-43"
                and temperature in (20, 538)
            ):
                continue

            predicted = case1_ys(
                alloy["Composition"],
                temperature,
            )

            rows.append(
                (
                    float(experimental),
                    predicted,
                )
            )

    experimental = [
        pair[0]
        for pair in rows
    ]

    predicted = [
        pair[1]
        for pair in rows
    ]

    return {
        "n": len(rows),
        "MAE_MPa": mae(
            experimental,
            predicted,
        ),
        "RMSE_MPa": rmse(
            experimental,
            predicted,
        ),
    }


def validate_case3() -> dict:
    """
    Recalculate Case-3 MAE and RMSE from the documented 29-point dataset.
    """

    table = case3_validation_table()

    return {
        "n": len(table),
        "MAE_MPa": mae(
            table["Experimental_YS_MPa"],
            table["Predicted_YS_MPa"],
        ),
        "RMSE_MPa": rmse(
            table["Experimental_YS_MPa"],
            table["Predicted_YS_MPa"],
        ),
    }


# =============================================================================
# COMMAND-LINE DEMONSTRATION
# =============================================================================

def main():
    """
    Demonstration of the extended URADES temperature models.
    """

    print("=" * 72)
    print("URADES EXTENDED — TEMPERATURE-DEPENDENT YIELD STRENGTH")
    print("=" * 72)

    # -----------------------------------------------------------------
    # CASE 1 DEMO
    # -----------------------------------------------------------------

    cb752 = {
        "Nb": 87.5,
        "W": 10.0,
        "Zr": 2.5,
    }

    print("\nCASE 1 — Cb-752")
    print("-" * 72)

    sr = case1_sponge_ratio(
        cb752
    )

    print(
        f"Sponge Ratio: {sr:.3f}"
    )

    for temperature in [
        20,
        600,
        1000,
        1200,
    ]:

        ys = case1_ys(
            cb752,
            temperature,
        )

        phi = case1_coupling_valve(
            sr,
            temperature,
        )

        print(
            f"T = {temperature:4.0f} °C | "
            f"Phi = {phi:.4f} | "
            f"YS = {ys:.1f} MPa"
        )

    # -----------------------------------------------------------------
    # CASE 3 DEMO
    # -----------------------------------------------------------------

    rheA = CASE3_DATA[0]

    print(
        "\nCASE 3 — HfMoNbTaTiZr"
    )
    print("-" * 72)

    anchor = case3_athermal_strength(
        rheA["VEC"],
        rheA["H_mix_kJ_mol"],
    )

    print(
        f"Athermal strength: {anchor:.1f} MPa"
    )

    print(
        f"G_mix: {rheA['G_mix_GPa']:.1f} GPa"
    )

    print(
        f"T_m: {rheA['Tm_K']:.0f} K"
    )

    for temperature in [
        24,
        800,
        1000,
        1200,
    ]:

        ys = case3_ys(
            temperature_c=temperature,
            vec=rheA["VEC"],
            h_mix_kj_mol=rheA["H_mix_kJ_mol"],
            g_mix_gpa=rheA["G_mix_GPa"],
            melting_temperature_k=rheA["Tm_K"],
        )

        print(
            f"T = {temperature:4.0f} °C | "
            f"YS = {ys:.1f} MPa"
        )

    # -----------------------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------------------

    print(
        "\nDOCUMENTED DATASET CHECK"
    )
    print("-" * 72)

    case1_metrics = validate_case1()

    print(
        f"Case 1: n={case1_metrics['n']}, "
        f"MAE={case1_metrics['MAE_MPa']:.2f} MPa, "
        f"RMSE={case1_metrics['RMSE_MPa']:.2f} MPa"
    )

    case3_metrics = validate_case3()

    print(
        f"Case 3: n={case3_metrics['n']}, "
        f"MAE={case3_metrics['MAE_MPa']:.2f} MPa, "
        f"RMSE={case3_metrics['RMSE_MPa']:.2f} MPa"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
