"""
URADES — Interactive Screening Application
============================================

Streamlit interface for the URADES computational framework.

Run from the repository root:

    streamlit run app/app.py

The scientific calculations are performed by urades.core.
This file contains only the graphical interface and screening workflow.
"""

from __future__ import annotations

import os
import sys
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# =============================================================================
# REPOSITORY PATH
# =============================================================================

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# =============================================================================
# URADES CORE
# =============================================================================

from urades.core import (
    run_URADES,
    calc_GVI,
    identify_case,
    check_boundary_conditions,
    atomic_to_weight,
    weight_to_atomic,
)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="URADES",
    page_icon="⚙",
    layout="wide",
)


# =============================================================================
# CONSTANTS
# =============================================================================

ELEMENTS = ["W", "Mo", "Hf", "Zr", "Ti"]

CASE_NAMES = {
    1: "Case 1 — Nb Engineering Alloy (IAS)",
    2: "Case 2 — Nb-Matrix RCCA (SR)",
    3: "Case 3 — Nb-Based RHEA (EI)",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_composition(composition):
    """Remove zero-valued elements and return a clean composition."""

    return {
        element: float(value)
        for element, value in composition.items()
        if value > 0
    }


def composition_string(composition):
    """Create a compact composition string."""

    return " ".join(
        f"{element}{value:.2f}"
        for element, value in composition.items()
        if value > 0
    )


def run_forward(composition, unit):
    """Run the actual URADES core."""

    return run_URADES(
        composition,
        input_unit=unit,
        gvi_gate=True,
        verbose=False,
    )


def result_is_usable(result):
    """Check whether the composition passed the hard boundary conditions."""

    return result.get("status") != "REJECTED"


def get_gvi(result):
    """Safely extract GVI."""

    gvi = result.get("GVI")

    if not isinstance(gvi, dict):
        return None

    return gvi.get("GVI")


def get_case(result):
    """Safely extract case."""

    return result.get("case")


def get_properties(result):
    """Safely extract property dictionary."""

    return result.get("properties", {})


def show_composition(composition, unit):
    """Display composition in a compact table."""

    df = pd.DataFrame(
        {
            "Element": list(composition.keys()),
            unit: [
                round(value, 3)
                for value in composition.values()
            ],
        }
    )

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
    )


def show_boundary_result(result):
    """Display boundary-condition result."""

    if result.get("status") == "REJECTED":

        st.error("Composition rejected by model boundary conditions.")

        violations = result.get(
            "violations",
            [],
        )

        for violation in violations:
            st.write(f"• {violation}")

        return False

    st.success(
        "Composition satisfies the defined URADES boundary conditions."
    )

    return True


def show_gvi(result):
    """Display GVI and component survival scores."""

    gvi_data = result.get("GVI", {})

    if not isinstance(gvi_data, dict):
        return

    gvi = gvi_data.get("GVI")

    if gvi is None:
        return

    st.markdown("### Global Viability Index")

    if gvi >= 0.5:
        st.success(
            f"GVI = {gvi:.4f} — PASS"
        )
    else:
        st.warning(
            f"GVI = {gvi:.4f} — SECONDARY PHASE RISK"
        )

    columns = st.columns(4)

    with columns[0]:
        st.metric(
            "GVI",
            f"{gvi:.4f}",
        )

    with columns[1]:
        st.metric(
            "S_VEC",
            f"{gvi_data.get('S_VEC', np.nan):.4f}",
        )

    with columns[2]:
        st.metric(
            "S_δ",
            f"{gvi_data.get('S_delta', np.nan):.4f}",
        )

    with columns[3]:

        if "S_SR" in gvi_data:
            st.metric(
                "S_SR",
                f"{gvi_data['S_SR']:.4f}",
            )
        else:
            st.metric(
                "S_SR",
                "Not used",
            )

    if gvi_data.get("calphad_trigger", False):

        st.warning(
            "The documented CALPHAD verification trigger is active: "
            "Mo > 5 at%, Hf > 5 at%, and Zr > 5 at% simultaneously."
        )


def show_case_result(result):
    """Display case-specific URADES output."""

    case = get_case(result)
    props = get_properties(result)

    st.markdown("### URADES Prediction")

    if case == 1:

        st.info(CASE_NAMES[1])

        columns = st.columns(4)

        with columns[0]:
            st.metric(
                "DBTT",
                f"{props.get('DBTT', np.nan):+.1f} °C",
            )

        with columns[1]:
            st.metric(
                "YS",
                f"{props.get('YS_MPa', np.nan):.0f} MPa",
            )

        with columns[2]:
            st.metric(
                "Tm",
                f"{props.get('Tm_C', np.nan):.0f} °C",
            )

        with columns[3]:
            st.metric(
                "Density",
                f"{props.get('density', np.nan):.3f} g/cc",
            )

        if "DBTT_range" in props:

            st.caption(
                f"Approximate reported uncertainty range: "
                f"{props['DBTT_range']}"
            )

    elif case == 2:

        st.info(CASE_NAMES[2])

        columns = st.columns(5)

        with columns[0]:
            st.metric(
                "DBTT",
                f"{props.get('DBTT', np.nan):+.1f} °C",
            )

        with columns[1]:
            st.metric(
                "SR",
                f"{props.get('SR', np.nan):.4f}",
            )

        with columns[2]:
            st.metric(
                "YS",
                f"{props.get('YS_MPa', np.nan):.0f} MPa",
            )

        with columns[3]:
            st.metric(
                "Tm",
                f"{props.get('Tm_C', np.nan):.0f} °C",
            )

        with columns[4]:
            st.metric(
                "Density",
                f"{props.get('density', np.nan):.3f}",
            )

        st.caption(
            f"SR model α = {props.get('alpha', 0):.2f}"
        )

    elif case == 3:

        st.info(CASE_NAMES[3])

        columns = st.columns(4)

        with columns[0]:
            st.metric(
                "EI",
                f"{props.get('EI', np.nan):.4f}",
            )

        with columns[1]:
            st.metric(
                "Classification",
                props.get(
                    "zone",
                    "Unknown",
                ),
            )

        with columns[2]:
            st.metric(
                "Tm",
                f"{props.get('Tm_C', np.nan):.0f} °C",
            )

        with columns[3]:
            st.metric(
                "Density",
                f"{props.get('density', np.nan):.3f} g/cc",
            )

        st.warning(
            "Case 3 provides an embrittlement-risk classification "
            "rather than a numerical DBTT prediction."
        )


def result_to_row(result, composition):
    """Convert one URADES result into one screening-table row."""

    props = get_properties(result)
    gvi_data = result.get("GVI", {})

    row = {
        "Case": result.get("case"),
        "Nb (wt%)": result.get("Nb_wt_pct"),
        "Nb (at%)": result.get("Nb_at_pct"),
        "W": composition.get("W", 0.0),
        "Mo": composition.get("Mo", 0.0),
        "Hf": composition.get("Hf", 0.0),
        "Zr": composition.get("Zr", 0.0),
        "Ti": composition.get("Ti", 0.0),
        "GVI": gvi_data.get("GVI", np.nan),
        "VEC": gvi_data.get("VEC", np.nan),
        "delta (%)": gvi_data.get("delta", np.nan),
        "DBTT (°C)": props.get("DBTT", np.nan),
        "YS (MPa)": props.get("YS_MPa", np.nan),
        "Tm (°C)": props.get("Tm_C", np.nan),
        "Density (g/cc)": props.get("density", np.nan),
        "SR": props.get("SR", np.nan),
        "EI": props.get("EI", np.nan),
        "Classification": props.get("zone", ""),
    }

    return row


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("URADES")

mode = st.sidebar.radio(
    "Select analysis mode",
    [
        "Forward Calculator",
        "Inverse Design",
    ],
)


# =============================================================================
# HEADER
# =============================================================================

st.title("URADES")

st.markdown(
    """
### Unified Refractory Alloy Descriptor and Embrittlement Screener

A physics-informed analytical framework for rapid screening of
Nb-based refractory alloys.

**Forward mode** evaluates a single composition.

**Inverse-design mode** searches a defined composition space for
compositions satisfying user-selected constraints.
"""
)


# =============================================================================
# FORWARD CALCULATOR
# =============================================================================

if mode == "Forward Calculator":

    st.header("Forward Calculator")

    st.write(
        "Enter the alloy composition below. Nb is automatically "
        "calculated as the balance to 100%."
    )

    unit_label = st.radio(
        "Composition input unit",
        ["wt%", "at%"],
        horizontal=True,
    )

    unit = (
        "wt"
        if unit_label == "wt%"
        else "at"
    )

    st.markdown("### Alloying Elements")

    col1, col2, col3 = st.columns(3)

    with col1:

        W = st.number_input(
            "W",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1,
        )

        Mo = st.number_input(
            "Mo",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1,
        )

    with col2:

        Hf = st.number_input(
            "Hf",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1,
        )

        Zr = st.number_input(
            "Zr",
            min_value=0.0,
            max_value=100.0,
            value=2.0,
            step=0.1,
        )

    with col3:

        Ti = st.number_input(
            "Ti",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1,
        )

    alloying_total = (
        W + Mo + Hf + Zr + Ti
    )

    Nb = 100.0 - alloying_total

    if Nb < 0:

        st.error(
            "The entered alloying additions exceed 100%."
        )

        st.stop()

    composition = {
        "Nb": Nb,
        "W": W,
        "Mo": Mo,
        "Hf": Hf,
        "Zr": Zr,
        "Ti": Ti,
    }

    composition = clean_composition(
        composition
    )

    st.markdown("### Input Composition")

    show_composition(
        composition,
        unit_label,
    )

    st.info(
        f"Calculated Nb = **{Nb:.2f} {unit_label}**"
    )

    if st.button(
        "Run URADES",
        type="primary",
        use_container_width=True,
    ):

        result = run_forward(
            composition,
            unit,
        )

        st.markdown("## Screening Result")

        # -------------------------------------------------------------
        # ROUTING
        # -------------------------------------------------------------

        if result.get("status") == "REJECTED":

            st.error("URADES screening rejected this composition.")

            show_boundary_result(result)

            st.stop()

        # -------------------------------------------------------------
        # CASE
        # -------------------------------------------------------------

        columns = st.columns(3)

        with columns[0]:

            st.metric(
                "URADES Case",
                result.get(
                    "case",
                    "—",
                ),
            )

        with columns[1]:

            st.metric(
                "Nb (wt%)",
                f"{result.get('Nb_wt_pct', np.nan):.2f}",
            )

        with columns[2]:

            st.metric(
                "Nb (at%)",
                f"{result.get('Nb_at_pct', np.nan):.2f}",
            )

        # -------------------------------------------------------------
        # BOUNDARY CONDITIONS
        # -------------------------------------------------------------

        if not show_boundary_result(result):
            st.stop()

        # -------------------------------------------------------------
        # GVI
        # -------------------------------------------------------------

        show_gvi(result)

        # -------------------------------------------------------------
        # CASE MODEL
        # -------------------------------------------------------------

        show_case_result(result)

        # -------------------------------------------------------------
        # COMPLETE OUTPUT
        # -------------------------------------------------------------

        with st.expander(
            "Show complete URADES output"
        ):

            st.json(result)


# =============================================================================
# INVERSE DESIGN
# =============================================================================

else:

    st.header("Inverse Design")

    st.write(
        """
Search a user-defined composition space.

The screening sequence is:

**Composition → Case routing → Boundary conditions → GVI →
Property constraints → Candidate alloys**
"""
    )

    # -----------------------------------------------------------------
    # SEARCH SETTINGS
    # -----------------------------------------------------------------

    st.sidebar.markdown("---")
    st.sidebar.subheader("Composition Search")

    step = st.sidebar.number_input(
        "Step size (wt%)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
    )

    st.sidebar.subheader(
        "Maximum Alloying Content"
    )

    max_W = st.sidebar.number_input(
        "W max",
        min_value=0.0,
        max_value=30.0,
        value=15.0,
        step=step,
    )

    max_Mo = st.sidebar.number_input(
        "Mo max",
        min_value=0.0,
        max_value=30.0,
        value=15.0,
        step=step,
    )

    max_Hf = st.sidebar.number_input(
        "Hf max",
        min_value=0.0,
        max_value=30.0,
        value=15.0,
        step=step,
    )

    max_Zr = st.sidebar.number_input(
        "Zr max",
        min_value=0.0,
        max_value=30.0,
        value=10.0,
        step=step,
    )

    max_Ti = st.sidebar.number_input(
        "Ti max",
        min_value=0.0,
        max_value=30.0,
        value=10.0,
        step=step,
    )

    st.sidebar.subheader(
        "Screening Constraints"
    )

    max_DBTT = st.sidebar.number_input(
        "Maximum DBTT (°C)",
        value=-50.0,
        step=10.0,
    )

    min_YS = st.sidebar.number_input(
        "Minimum YS (MPa)",
        value=400.0,
        step=50.0,
    )

    max_density = st.sidebar.number_input(
        "Maximum density (g/cc)",
        value=10.0,
        step=0.5,
    )

    min_GVI = st.sidebar.slider(
        "Minimum GVI",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

    max_display = st.sidebar.number_input(
        "Maximum candidates displayed",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
    )

    # -----------------------------------------------------------------
    # SEARCH BUTTON
    # -----------------------------------------------------------------

    run_search = st.button(
        "Run Composition Search",
        type="primary",
        use_container_width=True,
    )

    if run_search:

        # -------------------------------------------------------------
        # GRID
        # -------------------------------------------------------------

        W_values = np.arange(
            0,
            max_W + step / 2,
            step,
        )

        Mo_values = np.arange(
            0,
            max_Mo + step / 2,
            step,
        )

        Hf_values = np.arange(
            0,
            max_Hf + step / 2,
            step,
        )

        Zr_values = np.arange(
            0,
            max_Zr + step / 2,
            step,
        )

        Ti_values = np.arange(
            0,
            max_Ti + step / 2,
            step,
        )

        total_grid = (
            len(W_values)
            * len(Mo_values)
            * len(Hf_values)
            * len(Zr_values)
            * len(Ti_values)
        )

        st.info(
            f"Total composition combinations in grid: "
            f"**{total_grid:,}**"
        )

        # -------------------------------------------------------------
        # WARNING FOR VERY LARGE SEARCHES
        # -------------------------------------------------------------

        if total_grid > 250_000:

            st.warning(
                "This search space is large. "
                "A smaller step size or narrower composition range "
                "will substantially increase computation time."
            )

        progress = st.progress(0)

        status_text = st.empty()

        candidates = []

        evaluated = 0
        boundary_rejected = 0
        gvi_rejected = 0
        property_rejected = 0

        # -------------------------------------------------------------
        # SEARCH
        # -------------------------------------------------------------

        for index, (
            W,
            Mo,
            Hf,
            Zr,
            Ti,
        ) in enumerate(
            product(
                W_values,
                Mo_values,
                Hf_values,
                Zr_values,
                Ti_values,
            )
        ):

            Nb = (
                100.0
                - W
                - Mo
                - Hf
                - Zr
                - Ti
            )

            if Nb < 0:
                continue

            composition = {
                "Nb": float(Nb),
                "W": float(W),
                "Mo": float(Mo),
                "Hf": float(Hf),
                "Zr": float(Zr),
                "Ti": float(Ti),
            }

            composition = clean_composition(
                composition
            )

            evaluated += 1

            try:

                result = run_forward(
                    composition,
                    "wt",
                )

            except Exception:

                continue

            # ---------------------------------------------------------
            # BOUNDARY CHECK
            # ---------------------------------------------------------

            if result.get("status") == "REJECTED":

                boundary_rejected += 1
                continue

            # ---------------------------------------------------------
            # GVI
            # ---------------------------------------------------------

            gvi = get_gvi(result)

            if (
                gvi is not None
                and gvi < min_GVI
            ):

                gvi_rejected += 1
                continue

            # ---------------------------------------------------------
            # CASE 3
            # ---------------------------------------------------------

            case = get_case(result)
            props = get_properties(result)

            if case == 3:

                # Case 3 is classified by EI rather than numerical DBTT.
                row = result_to_row(
                    result,
                    composition,
                )

                candidates.append(row)

                continue

            # ---------------------------------------------------------
            # PROPERTY FILTERS
            # ---------------------------------------------------------

            dbtt = props.get(
                "DBTT",
                np.nan,
            )

            ys = props.get(
                "YS_MPa",
                np.nan,
            )

            density = props.get(
                "density",
                np.nan,
            )

            if (
                not np.isfinite(dbtt)
                or dbtt > max_DBTT
            ):

                property_rejected += 1
                continue

            if (
                not np.isfinite(ys)
                or ys < min_YS
            ):

                property_rejected += 1
                continue

            if (
                not np.isfinite(density)
                or density > max_density
            ):

                property_rejected += 1
                continue

            candidates.append(
                result_to_row(
                    result,
                    composition,
                )
            )

            # ---------------------------------------------------------
            # PROGRESS
            # ---------------------------------------------------------

            if index % 100 == 0:

                progress_value = (
                    index / total_grid
                )

                progress.progress(
                    min(
                        progress_value,
                        1.0,
                    )
                )

                status_text.text(
                    f"Evaluated: {evaluated:,} | "
                    f"Candidates: {len(candidates):,}"
                )

        progress.progress(1.0)

        status_text.text(
            f"Completed — {evaluated:,} compositions evaluated."
        )

        # -------------------------------------------------------------
        # RESULT DATAFRAME
        # -------------------------------------------------------------

        results_df = pd.DataFrame(
            candidates
        )

        # -------------------------------------------------------------
        # SUMMARY
        # -------------------------------------------------------------

        st.markdown(
            "## Screening Summary"
        )

        columns = st.columns(5)

        with columns[0]:

            st.metric(
                "Evaluated",
                f"{evaluated:,}",
            )

        with columns[1]:

            st.metric(
                "Boundary rejected",
                f"{boundary_rejected:,}",
            )

        with columns[2]:

            st.metric(
                "GVI rejected",
                f"{gvi_rejected:,}",
            )

        with columns[3]:

            st.metric(
                "Property rejected",
                f"{property_rejected:,}",
            )

        with columns[4]:

            st.metric(
                "Candidates",
                f"{len(results_df):,}",
            )

        # -------------------------------------------------------------
        # NO RESULTS
        # -------------------------------------------------------------

        if results_df.empty:

            st.warning(
                "No compositions satisfied the selected criteria."
            )

        else:

            # ---------------------------------------------------------
            # SORT
            # ---------------------------------------------------------

            if "DBTT (°C)" in results_df.columns:

                results_df = results_df.sort_values(
                    by=[
                        "DBTT (°C)",
                        "GVI",
                    ],
                    na_position="last",
                )

            # ---------------------------------------------------------
            # TABLE
            # ---------------------------------------------------------

            st.markdown(
                "## Candidate Alloys"
            )

            display_df = results_df.head(
                int(max_display)
            )

            st.caption(
                f"Showing {len(display_df):,} of "
                f"{len(results_df):,} candidates."
            )

            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                height=500,
            )

            # ---------------------------------------------------------
            # CSV
            # ---------------------------------------------------------

            csv_data = results_df.to_csv(
                index=False
            )

            st.download_button(
                "Download complete candidate list (CSV)",
                data=csv_data,
                file_name="URADES_screening_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # =========================================================
            # VISUALIZATION
            # =========================================================

            st.markdown(
                "## Design-Space Visualization"
            )

            (
                tab_dbtt_ys,
                tab_density_tm,
                tab_gvi,
                tab_vec_delta,
                tab_sr_ei,
                tab_composition,
            ) = st.tabs(
                [
                    "DBTT vs YS",
                    "Density vs Tm",
                    "GVI",
                    "VEC vs δ",
                    "SR / EI",
                    "Composition",
                ]
            )

            # ---------------------------------------------------------
            # DBTT VS YS
            # ---------------------------------------------------------

            with tab_dbtt_ys:

                plot_df = results_df.dropna(
                    subset=[
                        "DBTT (°C)",
                        "YS (MPa)",
                    ]
                )

                if plot_df.empty:

                    st.info(
                        "No numerical DBTT candidates available "
                        "for this plot."
                    )

                else:

                    fig, ax = plt.subplots(
                        figsize=(8, 5)
                    )

                    for case in sorted(
                        plot_df["Case"].dropna().unique()
                    ):

                        sub = plot_df[
                            plot_df["Case"] == case
                        ]

                        ax.scatter(
                            sub["DBTT (°C)"],
                            sub["YS (MPa)"],
                            s=28,
                            alpha=0.65,
                            label=f"Case {int(case)}",
                        )

                    ax.axvline(
                        max_DBTT,
                        linestyle="--",
                        linewidth=1,
                        label="DBTT limit",
                    )

                    ax.axhline(
                        min_YS,
                        linestyle="--",
                        linewidth=1,
                        label="YS limit",
                    )

                    ax.set_xlabel(
                        "DBTT (°C)"
                    )

                    ax.set_ylabel(
                        "Yield Strength (MPa)"
                    )

                    ax.set_title(
                        "DBTT vs Yield Strength"
                    )

                    ax.grid(
                        True,
                        alpha=0.25,
                    )

                    ax.legend()

                    st.pyplot(
                        fig,
                        use_container_width=True,
                    )

                    plt.close(fig)

            # ---------------------------------------------------------
            # DENSITY VS TM
            # ---------------------------------------------------------

            with tab_density_tm:

                fig, ax = plt.subplots(
                    figsize=(8, 5)
                )

                for case in sorted(
                    results_df["Case"].dropna().unique()
                ):

                    sub = results_df[
                        results_df["Case"] == case
                    ]

                    ax.scatter(
                        sub["Density (g/cc)"],
                        sub["Tm (°C)"],
                        s=28,
                        alpha=0.65,
                        label=f"Case {int(case)}",
                    )

                ax.axvline(
                    max_density,
                    linestyle="--",
                    linewidth=1,
                    label="Density limit",
                )

                ax.set_xlabel(
                    "Density (g/cc)"
                )

                ax.set_ylabel(
                    "Melting Point (°C)"
                )

                ax.set_title(
                    "Density vs Melting Point"
                )

                ax.grid(
                    True,
                    alpha=0.25,
                )

                ax.legend()

                st.pyplot(
                    fig,
                    use_container_width=True,
                )

                plt.close(fig)

            # ---------------------------------------------------------
            # GVI
            # ---------------------------------------------------------

            with tab_gvi:

                fig, ax = plt.subplots(
                    figsize=(8, 5)
                )

                ax.hist(
                    results_df["GVI"].dropna(),
                    bins=25,
                    alpha=0.8,
                )

                ax.axvline(
                    min_GVI,
                    linestyle="--",
                    linewidth=1,
                    label=f"GVI threshold = {min_GVI:.2f}",
                )

                ax.set_xlabel(
                    "Global Viability Index (GVI)"
                )

                ax.set_ylabel(
                    "Number of candidates"
                )

                ax.set_title(
                    "GVI Distribution"
                )

                ax.grid(
                    True,
                    axis="y",
                    alpha=0.25,
                )

                ax.legend()

                st.pyplot(
                    fig,
                    use_container_width=True,
                )

                plt.close(fig)

            # ---------------------------------------------------------
            # VEC VS DELTA
            # ---------------------------------------------------------

            with tab_vec_delta:

                fig, ax = plt.subplots(
                    figsize=(8, 5)
                )

                for case in sorted(
                    results_df["Case"].dropna().unique()
                ):

                    sub = results_df[
                        results_df["Case"] == case
                    ]

                    ax.scatter(
                        sub["VEC"],
                        sub["delta (%)"],
                        s=28,
                        alpha=0.65,
                        label=f"Case {int(case)}",
                    )

                ax.axvline(
                    5.3,
                    linestyle="--",
                    linewidth=1,
                    label="VEC = 5.3",
                )

                ax.axhline(
                    6.5,
                    linestyle="--",
                    linewidth=1,
                    label="δ = 6.5%",
                )

                ax.set_xlabel(
                    "VEC"
                )

                ax.set_ylabel(
                    "Atomic size mismatch, δ (%)"
                )

                ax.set_title(
                    "VEC vs Atomic Size Mismatch"
                )

                ax.grid(
                    True,
                    alpha=0.25,
                )

                ax.legend()

                st.pyplot(
                    fig,
                    use_container_width=True,
                )

                plt.close(fig)

            # ---------------------------------------------------------
            # SR / EI
            # ---------------------------------------------------------

            with tab_sr_ei:

                st.markdown(
                    "### Case 2 — Sponge Ratio"
                )

                case2 = results_df[
                    results_df["Case"] == 2
                ].dropna(
                    subset=[
                        "SR",
                        "DBTT (°C)",
                    ]
                )

                if case2.empty:

                    st.info(
                        "No Case 2 candidates available."
                    )

                else:

                    fig, ax = plt.subplots(
                        figsize=(8, 5)
                    )

                    ax.scatter(
                        case2["SR"],
                        case2["DBTT (°C)"],
                        s=30,
                        alpha=0.7,
                    )

                    ax.set_xlabel(
                        "Sponge Ratio (SR)"
                    )

                    ax.set_ylabel(
                        "DBTT (°C)"
                    )

                    ax.set_title(
                        "Case 2 — SR vs DBTT"
                    )

                    ax.grid(
                        True,
                        alpha=0.25,
                    )

                    st.pyplot(
                        fig,
                        use_container_width=True,
                    )

                    plt.close(fig)

                st.markdown(
                    "### Case 3 — Embrittlement Index"
                )

                case3 = results_df[
                    results_df["Case"] == 3
                ].dropna(
                    subset=["EI"]
                )

                if case3.empty:

                    st.info(
                        "No Case 3 candidates available."
                    )

                else:

                    fig, ax = plt.subplots(
                        figsize=(8, 5)
                    )

                    ax.hist(
                        case3["EI"],
                        bins=25,
                        alpha=0.8,
                    )

                    ax.axvline(
                        0.10,
                        linestyle="--",
                        linewidth=1,
                        label="Ductile threshold",
                    )

                    ax.axvline(
                        0.50,
                        linestyle="--",
                        linewidth=1,
                        label="Transition threshold",
                    )

                    ax.set_xlabel(
                        "Embrittlement Index (EI)"
                    )

                    ax.set_ylabel(
                        "Number of candidates"
                    )

                    ax.set_title(
                        "Case 3 — EI Distribution"
                    )

                    ax.grid(
                        True,
                        axis="y",
                        alpha=0.25,
                    )

                    ax.legend()

                    st.pyplot(
                        fig,
                        use_container_width=True,
                    )

                    plt.close(fig)

            # ---------------------------------------------------------
            # COMPOSITION
            # ---------------------------------------------------------

            with tab_composition:

                composition_columns = [
                    "Nb (wt%)",
                    "W",
                    "Mo",
                    "Hf",
                    "Zr",
                    "Ti",
                ]

                available = [
                    column
                    for column in composition_columns
                    if column in results_df.columns
                ]

                if available:

                    st.dataframe(
                        results_df[
                            available
                        ].head(
                            int(max_display)
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )
