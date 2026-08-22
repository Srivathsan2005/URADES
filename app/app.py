"""
app/app.py
===========
URADES — Streamlit interactive application.

Two-tab layout:
  Tab 1 · Alloy Screener   — enter a composition, get full URADES output
  Tab 2 · Dataset Explorer — 8 charts explaining the framework and datasets

Run from the repository root:
    streamlit run app/app.py
"""

import os
import sys
import math
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from urades.core import (
    run_URADES, calc_GVI, predict_case2, predict_case3,
    predict_case1, identify_case, atomic_to_weight, weight_to_atomic,
    check_boundary_conditions,
    IAS_BASELINE, IAS_COEFFS,
    SR_kW, SR_kMo, SR_kBuffer, SR_BASELINE,
    EI_ALPHA, EI_DUCTILE, EI_TRANSITION, EI_BRITTLE,
    GVI_THRESHOLD, _calc_density, _calc_Tm_ROM, _calc_VEC, _calc_delta,
)
from urades.data import (
    CASE1_DATA, CASE2_DATA, CASE3_DATA,
    DUCTILE, TRANSITION, BRITTLE, CONFIRMED,
)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="URADES",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# DESIGN TOKENS
# =============================================================================

C1_COL   = "#2563EB"   # Case 1 — deep blue
C2_COL   = "#059669"   # Case 2 — emerald
C3_COL   = "#DC2626"   # Case 3 — red
PASS_COL = "#16A34A"
FAIL_COL = "#DC2626"
FLAG_COL = "#D97706"
BG       = "#F3F6FA"
CARD_BG  = "#FFFFFF"
BORDER   = "#D8E0EA"
TEXT     = "#172033"
MUTED    = "#667085"

ZONE_COL = {
    DUCTILE:    "#16A34A",
    TRANSITION: "#D97706",
    BRITTLE:    "#EA580C",
    CONFIRMED:  "#DC2626",
}

# =============================================================================
# GLOBAL CSS
# =============================================================================

st.markdown(f"""
<style>
  /* Base */
  html, body, [data-testid="stAppViewContainer"] {{
      background-color: {BG};
      color: {TEXT};
      font-family: 'Inter', 'Segoe UI', sans-serif;
  }}
  [data-testid="stHeader"] {{ background: transparent; }}
  [data-testid="stStatusWidget"],
  [data-testid="stStatusWidget"] button {{
      color: {TEXT} !important;
      opacity: 1 !important;
  }}
  [data-testid="stStatusWidget"] svg {{
      color: {C2_COL} !important;
      stroke: {C2_COL} !important;
  }}

  /* Cards */
  .urades-card {{
      background: {CARD_BG};
      border: 1px solid {BORDER};
      border-radius: 10px;
      padding: 1.2rem 1.4rem;
      margin-bottom: 1rem;
  }}

  /* Status badges */
  .badge {{
      display: inline-block;
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.04em;
  }}
  .badge-ok       {{ background:#14532d; color:#86efac; }}
  .badge-flagged  {{ background:#451a03; color:#fcd34d; }}
  .badge-rejected {{ background:#450a0a; color:#fca5a5; }}
  .badge-c1       {{ background:#1e3a8a; color:#93c5fd; }}
  .badge-c2       {{ background:#064e3b; color:#6ee7b7; }}
  .badge-c3       {{ background:#7f1d1d; color:#fca5a5; }}

  /* Metric tiles */
  .metric-row {{ display:flex; gap:0.8rem; flex-wrap:wrap; margin-top:0.6rem; }}
  .metric-tile {{
      flex:1; min-width:120px;
      background:{BG}; border:1px solid {BORDER};
      border-radius:8px; padding:0.7rem 1rem;
      text-align:center;
  }}
  .metric-label {{ font-size:0.72rem; color:{MUTED}; text-transform:uppercase;
                   letter-spacing:0.07em; margin-bottom:3px; }}
  .metric-value {{ font-size:1.45rem; font-weight:700; color:{TEXT}; }}
  .metric-sub   {{ font-size:0.72rem; color:{MUTED}; margin-top:2px; }}

  /* GVI bar */
  .gvi-bar-outer {{
      height:10px; background:{BORDER};
      border-radius:5px; overflow:hidden; margin-top:4px;
  }}

  /* Section headers */
  .section-eyebrow {{
      font-size:0.7rem; font-weight:700; letter-spacing:0.12em;
      text-transform:uppercase; color:{MUTED}; margin-bottom:4px;
  }}

  .section-title {{
      font-size:1.1rem; font-weight:700; color:{TEXT}; margin-bottom:0.8rem;
  }}

  .stTabs [data-baseweb="tab"],
  .stTabs [data-baseweb="tab"] p,
  .stTabs [role="tab"],
  .stTabs [role="tab"] p {{
      color: {MUTED} !important;
      font-weight:600 !important;
  }}
  .stTabs [data-baseweb="tab"][aria-selected="true"],
  .stTabs [data-baseweb="tab"][aria-selected="true"] p,
  .stTabs [role="tab"][aria-selected="true"],
  .stTabs [role="tab"][aria-selected="true"] p {{
      color: {TEXT} !important;
      border-bottom: 2px solid {C2_COL};
  }}

  .stSlider label {{ color:{MUTED} !important; font-size:0.8rem; }}
    [data-testid="stNumberInput"] label,
    [data-testid="stSelectbox"] label {{ color:{MUTED} !important; font-size:0.76rem; }}
    [data-testid="stVerticalBlock"] > [data-testid="stElement"] {{ margin-bottom:0.35rem; }}
    .bound-heading {{ color:{MUTED}; font-size:0.7rem; text-transform:uppercase;
                                        letter-spacing:0.08em; margin:0.2rem 0 0.1rem 0; }}
    .bound-name {{ color:{TEXT}; font-weight:700; padding-top:0.55rem; }}
  hr {{ border-color:{BORDER}; margin:1.2rem 0; }}
</style>
""", unsafe_allow_html=True)


def metric_html(label, value, sub=""):
    return (f"<div class='metric-tile'><div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{value}</div><div class='metric-sub'>{sub}</div></div>")


def gvi_bar_html(score: float, label: str = "") -> str:
    pct = round(score * 100, 1)
    color = PASS_COL if score >= GVI_THRESHOLD else FAIL_COL
    return (f"<div style='color:{MUTED};font-size:0.75rem'>{label}"
            f" <strong>{pct:.1f}%</strong></div>")


def case_badge(case: int) -> str:
    return f"<span class='badge badge-c{case}'>{CASE_LABELS[case]}</span>"


def status_badge(status: str) -> str:
    css = {"OK": "badge-ok", "FLAGGED": "badge-flagged",
           "REJECTED": "badge-rejected"}.get(status, "badge-flagged")
    return f"<span class='badge {css}'>{status}</span>"


def dark_fig(w=6, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.grid(True, color=BORDER, linewidth=0.6, linestyle=":")
    return fig, ax


@st.cache_data
def build_all_alloy_data():
    """Compile the common chart data used by the dataset explorer."""
    rows = []
    for name, W, Mo, V, Ti, Zr, Hf, exp_dbtt in CASE1_DATA:
        comp_wt = {"Nb": 100 - W - Mo - V - Ti - Zr - Hf,
                   "W": W, "Mo": Mo, "V": V, "Ti": Ti, "Zr": Zr, "Hf": Hf}
        comp = weight_to_atomic(comp_wt)
        props = predict_case1(comp)
        gvi = calc_GVI(comp, 1)
        rows.append({"case": 1, "name": name, "Nb_at": comp["Nb"],
                     "exp_dbtt": exp_dbtt, "pred_dbtt": props["DBTT"],
                     "density": _calc_density(comp), "Tm": _calc_Tm_ROM(comp),
                     "VEC": _calc_VEC(comp), "delta": _calc_delta(comp),
                     "GVI": gvi["GVI"], "S_VEC": gvi["S_VEC"],
                     "S_delta": gvi["S_delta"], "S_SR": None,
                     "YS": props["YS_MPa"], "SR": None, "EI": None, "zone": None})
    for name, W, Mo, Hf, Zr, Ti, exp_dbtt, _ in CASE2_DATA:
        comp = {"Nb": 100 - W - Mo - Hf - Zr - Ti, "W": W, "Mo": Mo,
                "Hf": Hf, "Zr": Zr, "Ti": Ti}
        props = predict_case2(comp)
        gvi = calc_GVI(comp, 2)
        rows.append({"case": 2, "name": name, "Nb_at": comp["Nb"],
                     "exp_dbtt": exp_dbtt, "pred_dbtt": props["DBTT"],
                     "density": props["density"], "Tm": props["Tm_C"],
                     "VEC": _calc_VEC(comp), "delta": _calc_delta(comp),
                     "GVI": gvi["GVI"], "S_VEC": gvi["S_VEC"],
                     "S_delta": gvi["S_delta"], "S_SR": gvi.get("S_SR"),
                     "YS": props["YS_MPa"], "SR": props["SR"], "EI": None, "zone": None})
    for name, comp, exp_zone in CASE3_DATA:
        props = predict_case3(comp)
        gvi = calc_GVI(comp, 3)
        rows.append({"case": 3, "name": name, "Nb_at": comp.get("Nb", 0),
                     "exp_dbtt": None, "pred_dbtt": None,
                     "density": _calc_density(comp), "Tm": _calc_Tm_ROM(comp),
                     "VEC": _calc_VEC(comp), "delta": _calc_delta(comp),
                     "GVI": gvi["GVI"], "S_VEC": gvi["S_VEC"],
                     "S_delta": gvi["S_delta"], "S_SR": None, "YS": None,
                     "SR": None, "EI": props["EI"], "zone": props["zone"]})
    return rows


CASE_COLORS = {1: C1_COL, 2: C2_COL, 3: C3_COL}
CASE_LABELS = {1: "Case 1 (IAS)", 2: "Case 2 (SR)", 3: "Case 3 (EI)"}


def bounds_grid(minimum, maximum, step):
    """Return inclusive grid points without exceeding the upper bound."""
    if maximum < minimum:
        return np.array([])
    count = int(math.floor((maximum - minimum) / step + 1e-12))
    return minimum + step * np.arange(count + 1)


def search_inverse_design(target_dbtt, dbtt_tolerance, min_ys, max_density,
                          min_tm, family, bounds, step):
    """Search the requested at% bounds using the existing URADES models."""
    candidates = []
    elements = ["W", "Mo", "Hf", "Zr", "Ti"]
    values = [
        bounds_grid(bounds[element][0], bounds[element][1], step)
        for element in elements
    ]
    family_case = {"Case 1 (IAS)": 1, "Case 2 (RCCA)": 2,
                   "Case 3 (RHEA)": 3}.get(family)

    for W, Mo, Hf, Zr, Ti in product(*values):
        if family_case == 2 and Hf + Zr + Ti > 22.4:
            continue
        nb = 100.0 - W - Mo - Hf - Zr - Ti
        if nb <= 0:
            continue

        composition = {"Nb": nb, "W": W, "Mo": Mo, "Hf": Hf,
                       "Zr": Zr, "Ti": Ti}
        case, _, _ = identify_case(composition)
        if family_case is not None and case != family_case:
            continue
        passed, _ = check_boundary_conditions(composition, case)
        if not passed:
            continue

        gvi = calc_GVI(composition, case)
        if gvi["GVI"] < GVI_THRESHOLD:
            continue
        if case == 1:
            props = predict_case1(composition)
        elif case == 2:
            props = predict_case2(composition)
        else:
            props = predict_case3(composition)

        dbtt = props.get("DBTT")
        filter_density = props["density"]
        filter_tm = props["Tm_C"]
        if case == 3:
            filter_density = _calc_density(composition)
            filter_tm = _calc_Tm_ROM(composition)
        if dbtt is not None and abs(dbtt - target_dbtt) > dbtt_tolerance:
            continue
        if "YS_MPa" in props and props["YS_MPa"] < min_ys:
            continue
        if filter_density > max_density or filter_tm < min_tm:
            continue

        candidates.append({
            "case": case, "composition": composition, "DBTT": dbtt,
            "YS": props.get("YS_MPa"), "density": props["density"],
            "Tm": props["Tm_C"], "GVI": gvi["GVI"],
            "EI": props.get("EI"), "zone": props.get("zone"),
        })

    candidates.sort(key=lambda row: (
        row["DBTT"] is None,
        row["DBTT"] if row["DBTT"] is not None else 0,
        -row["GVI"]))
    return candidates

# =============================================================================
# HEADER
# =============================================================================

st.markdown(f"""
<div style='padding:1.6rem 0 0.4rem 0;'>
  <div class='section-eyebrow'>Refractory Alloy Design Tool</div>
  <h1 style='font-size:2rem;font-weight:800;color:{TEXT};margin:0 0 0.3rem 0;
             letter-spacing:-0.02em;'>
    URADES
  </h1>
  <p style='color:{MUTED};font-size:0.95rem;max-width:640px;margin:0;'>
    Unified Refractory Alloy Descriptor and Embrittlement Screener —
    a physics-based hierarchical framework for predicting DBTT and
    embrittlement behaviour in Nb-based alloys across three compositional regimes.
  </p>
</div>
<hr/>
""", unsafe_allow_html=True)

# =============================================================================
# TABS
# =============================================================================

tab_inverse, tab_screen, tab_explore = st.tabs([
    "Inverse Design", "Alloy Screener", "Dataset Explorer"])

# =============================================================================
# TAB 1 — INVERSE DESIGN
# =============================================================================

with tab_inverse:
    st.markdown("<div class='section-eyebrow'>Inverse design constraints</div>",
                unsafe_allow_html=True)
    col_form, col_result = st.columns([1, 1.4], gap="large")

    with col_form:
        st.markdown("**Constraints**")
        constraint_cols = st.columns(2)
        with constraint_cols[0]:
            target_dbtt = st.number_input("Target DBTT (°C)", value=-50.0, step=5.0)
            min_ys = st.number_input("Min YS (MPa)", min_value=0.0,
                                     value=400.0, step=25.0)
            min_tm = st.number_input("Min Tm (°C)", min_value=0.0,
                                     value=1800.0, step=50.0)
        with constraint_cols[1]:
            dbtt_tolerance = st.number_input("± Tolerance", min_value=0.0,
                                             value=25.0, step=5.0)
            max_density = st.number_input("Max ρ (g/cc)", min_value=0.0,
                                          value=10.0, step=0.5)
            family = st.selectbox("Family", ["Any", "Case 1 (IAS)",
                                             "Case 2 (RCCA)", "Case 3 (RHEA)"])

        st.markdown("**Element Bounds (at%)**")
        bounds = {}
        heading_cols = st.columns([0.55, 1, 1])
        for column, heading in zip(heading_cols, ["Element", "Min", "Max"]):
            column.markdown(f"<div class='bound-heading'>{heading}</div>",
                            unsafe_allow_html=True)
        for index, element in enumerate(["W", "Mo", "Hf", "Zr", "Ti"]):
            bound_cols = st.columns([0.55, 1, 1])
            bound_cols[0].markdown(f"<div class='bound-name'>{element}</div>",
                                   unsafe_allow_html=True)
            bounds[element] = (
                bound_cols[1].number_input(f"{element} min", min_value=0.0,
                                           max_value=100.0, value=0.0, step=1.0,
                                           key=f"inverse_{element}_min",
                                           label_visibility="collapsed"),
                bound_cols[2].number_input(f"{element} max", min_value=0.0,
                                           max_value=100.0, value=40.0, step=1.0,
                                           key=f"inverse_{element}_max",
                                           label_visibility="collapsed"),
            )
        st.caption("Hf + Zr + Ti ≤ 22.4 applies mainly to Case 2 (RCCA).")
        st.caption("Nb = 100 − (W + Mo + Hf + Zr + Ti) is auto-computed.")
        search_step = st.number_input("Customizable Search Step (at%)",
                                      min_value=0.1, value=2.0, step=0.5)
        search_btn = st.button("Search candidates", use_container_width=True,
                               type="primary")

    with col_result:
        if search_btn:
            candidates = search_inverse_design(
                target_dbtt, dbtt_tolerance, min_ys, max_density, min_tm,
                family, bounds, search_step)
            st.markdown("**Qualified candidates**")
            if candidates:
                rows = []
                for candidate in candidates[:100]:
                    composition = candidate["composition"]
                    rows.append({
                        "Family": CASE_LABELS[candidate["case"]],
                        "W": round(composition["W"], 1),
                        "Mo": round(composition["Mo"], 1),
                        "Hf": round(composition["Hf"], 1),
                        "Zr": round(composition["Zr"], 1),
                        "Ti": round(composition["Ti"], 1),
                        "Nb": round(composition["Nb"], 1),
                        "DBTT": candidate["DBTT"],
                        "YS (MPa)": candidate["YS"],
                        "ρ (g/cc)": candidate["density"],
                        "Tm (°C)": candidate["Tm"],
                        "GVI": round(candidate["GVI"], 4),
                        "EI": candidate["EI"], "Zone": candidate["zone"],
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)
                st.caption(f"Showing {len(rows)} of {len(candidates)} matches.")
            else:
                st.info("No candidates match these constraints. Increase the search step or relax a target.")
            if family == "Case 3 (RHEA)":
                st.caption("Case 3 has no DBTT model; its candidates use the other constraints.")
        else:
            st.info("Set constraints and search the alloy design space.")

# =============================================================================
# TAB 1 — ALLOY SCREENER
# =============================================================================

with tab_screen:

    st.markdown("<div class='section-eyebrow'>Step 1 — Enter composition</div>",
                unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1.4], gap="large")

    with col_form:
        unit = st.radio("Composition unit", ["at%", "wt%"],
                        horizontal=True, label_visibility="collapsed")
        unit_key = "at" if unit == "at%" else "wt"

        st.markdown(f"<p style='color:{MUTED};font-size:0.8rem;margin:4px 0 12px 0;'>"
                    f"Nb is calculated as the balance (100 − sum of all others).</p>",
                    unsafe_allow_html=True)

        W  = st.slider("W  (%)",  0.0, 20.0, 4.0, 0.5)
        Mo = st.slider("Mo (%)", 0.0, 20.0, 0.0, 0.5)
        Hf = st.slider("Hf (%)", 0.0, 22.0, 15.0, 0.5)
        Zr = st.slider("Zr (%)", 0.0, 10.0, 0.0, 0.5)
        Ti = st.slider("Ti (%)", 0.0, 15.0, 5.5, 0.5)
        V  = st.slider("V  (%)",  0.0, 15.0, 0.0, 0.5)

        Nb = 100.0 - W - Mo - Hf - Zr - Ti - V
        nb_color = PASS_COL if Nb > 0 else FAIL_COL
        st.markdown(
            f"<div style='font-size:0.85rem;color:{nb_color};font-weight:600;"
            f"padding:6px 0;'>Nb (balance) = {Nb:.1f} {unit}</div>",
            unsafe_allow_html=True)

        run_btn = st.button("Run URADES", use_container_width=True,
                            type="primary", disabled=(Nb <= 0))

    with col_result:
        if run_btn and Nb > 0:
            comp = {"Nb": Nb, "W": W, "Mo": Mo, "Hf": Hf,
                    "Zr": Zr, "Ti": Ti, "V": V}
            # Remove zeros
            comp = {el: v for el, v in comp.items() if v > 0}

            result = run_URADES(comp, input_unit=unit_key, verbose=False)
            status = result["status"]
            case   = result["case"]

            # ── Status header ────────────────────────────────────────────────
            st.markdown(
                f"{status_badge(status)}&nbsp;&nbsp;{case_badge(case)}",
                unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            if status == "REJECTED":
                st.error("Boundary condition violated — alloy rejected before prediction.")
                for v in result["violations"]:
                    st.markdown(f"- {v}")

            else:
                props = result["properties"]
                gvi_d = result["GVI"]

                # ── GVI block ────────────────────────────────────────────────
                st.markdown("<div class='urades-card'>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='section-eyebrow'>Phase Stability — "
                    f"Global Viability Index</div>",
                    unsafe_allow_html=True)

                gvi_score = gvi_d["GVI"]
                gvi_label = (
                    f"<span style='color:{PASS_COL};font-weight:700;'>PASS</span>"
                    if gvi_score >= GVI_THRESHOLD else
                    f"<span style='color:{FAIL_COL};font-weight:700;'>SECONDARY PHASE RISK</span>"
                )
                st.markdown(
                    f"<div style='font-size:1.6rem;font-weight:800;"
                    f"color:{TEXT};margin-bottom:4px;'>"
                    f"GVI = {gvi_score:.4f}</div>"
                    f"<div style='font-size:0.85rem;margin-bottom:12px;'>"
                    f"{gvi_label}</div>",
                    unsafe_allow_html=True)

                # Component breakdown
                components = [
                    ("S_VEC  (valence electron score)",   gvi_d["S_VEC"]),
                    ("S_δ    (atomic size mismatch score)", gvi_d["S_delta"]),
                ]
                if gvi_d.get("S_SR") is not None:
                    components.append(
                        ("S_SR   (sponge ratio score — Case 2 only)", gvi_d["S_SR"]))

                html_bars = "".join(
                    gvi_bar_html(v, label=l) for l, v in components)
                st.markdown(html_bars, unsafe_allow_html=True)

                if gvi_d.get("calphad_trigger"):
                    st.warning(
                        "Mo > 5%, Hf > 5%, Zr > 5% simultaneously — "
                        "CALPHAD verification recommended regardless of GVI.")

                st.markdown(
                    f"<div style='font-size:0.75rem;color:{MUTED};margin-top:8px;'>"
                    f"VEC = {gvi_d['VEC']:.3f} &nbsp;|&nbsp; "
                    f"δ = {gvi_d['delta']:.3f}%</div>",
                    unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # ── Property predictions ──────────────────────────────────────
                st.markdown("<div class='urades-card'>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-eyebrow'>Property Predictions</div>",
                    unsafe_allow_html=True)

                tiles = []
                if "DBTT" in props:
                    dbtt_color = C2_COL if props["DBTT"] < 0 else C3_COL
                    tiles.append(metric_html(
                        "DBTT",
                        f"<span style='color:{dbtt_color}'>"
                        f"{props['DBTT']:+.1f} °C</span>",
                        props.get("DBTT_range", ""),
                    ))
                if "EI" in props:
                    zone_color = ZONE_COL.get(props["zone"], TEXT)
                    tiles.append(metric_html(
                        "Embrittlement Index",
                        f"<span style='color:{zone_color}'>{props['EI']:.4f}</span>",
                        props["zone"],
                    ))
                if "YS_MPa" in props:
                    tiles.append(metric_html("Yield Strength",
                                             f"{props['YS_MPa']:.0f} MPa", "estimate"))
                if "Tm_C" in props:
                    tiles.append(metric_html("Melting Point",
                                             f"{props['Tm_C']:.0f} °C", "ROM"))
                if "density" in props:
                    tiles.append(metric_html("Density",
                                             f"{props['density']:.3f}", "g/cc"))
                if "SR" in props and props["SR"] is not None:
                    tiles.append(metric_html("Sponge Ratio",
                                             f"{props['SR']:.4f}", "α = 0"))

                st.markdown(
                    f"<div class='metric-row'>{''.join(tiles)}</div>",
                    unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class='urades-card' style='text-align:center;padding:2.5rem;'>
              <div style='font-size:1.2rem;font-weight:700;margin-bottom:0.6rem;'>Ready to screen</div>
              <div style='color:{MUTED};font-size:0.9rem;'>
                Adjust the sliders and click <strong>Run URADES</strong>
                to screen your alloy.
              </div>
            </div>""", unsafe_allow_html=True)

# =============================================================================
# TAB 2 — DATASET EXPLORER
# =============================================================================

with tab_explore:

    alloy_data = build_all_alloy_data()

    st.markdown(f"""
    <p style='color:{MUTED};font-size:0.88rem;max-width:720px;margin-bottom:1.2rem;'>
    All three URADES training datasets visualised together.
    <span style='color:{C1_COL};font-weight:600;'>■ Case 1</span> (IAS, n=23) &nbsp;
    <span style='color:{C2_COL};font-weight:600;'>■ Case 2</span> (SR, n=11) &nbsp;
    <span style='color:{C3_COL};font-weight:600;'>■ Case 3</span> (EI, n=25)
    </p>""", unsafe_allow_html=True)

    # ── Row 1: DBTT vs YS  |  Density vs Tm ─────────────────────────────────
    r1a, r1b = st.columns(2, gap="large")

    with r1a:
        st.markdown(
            "<div class='section-eyebrow'>Performance envelope</div>"
            "<div class='section-title'>DBTT vs Yield Strength</div>",
            unsafe_allow_html=True)

        fig, ax = dark_fig(5.5, 4)
        for case in [1, 2]:
            rows = [d for d in alloy_data
                    if d["case"] == case and d["pred_dbtt"] is not None
                    and d["YS"] is not None]
            if rows:
                ax.scatter(
                    [r["pred_dbtt"] for r in rows],
                    [r["YS"]        for r in rows],
                    color=CASE_COLORS[case], s=55,
                    edgecolors="white", linewidths=0.4,
                    alpha=0.85, label=CASE_LABELS[case], zorder=3)
                for r in rows:
                    ax.annotate(r["name"], (r["pred_dbtt"], r["YS"]),
                                textcoords="offset points", xytext=(4, 3),
                                fontsize=5.5, color="#94A3B8")

        ax.axvline(0,   color="#475569", linewidth=0.9, linestyle="--", alpha=0.6)
        ax.axhline(400, color="#475569", linewidth=0.9, linestyle="--", alpha=0.6)
        ax.text(2,   405, "YS = 400 MPa", fontsize=6.5, color="#64748B")
        ax.text(2,   ax.get_ylim()[0]+10, "DBTT = 0°C", fontsize=6.5, color="#64748B",
                rotation=90, va="bottom")
        ax.set_xlabel("Predicted DBTT (°C)", fontsize=9)
        ax.set_ylabel("Predicted YS (MPa)",  fontsize=9)
        ax.legend(fontsize=7, facecolor="#1E293B", edgecolor="#334155",
                  labelcolor="#F1F5F9")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Lower-left quadrant = low DBTT + adequate strength — the target window for propulsion alloys.")

    with r1b:
        st.markdown(
            "<div class='section-eyebrow'>Thermal merit</div>"
            "<div class='section-title'>Density vs Melting Point</div>",
            unsafe_allow_html=True)

        fig, ax = dark_fig(5.5, 4)
        for case in [1, 2, 3]:
            rows = [d for d in alloy_data if d["case"] == case]
            ax.scatter(
                [r["Tm"]      for r in rows],
                [r["density"] for r in rows],
                color=CASE_COLORS[case], s=55,
                edgecolors="white", linewidths=0.4,
                alpha=0.85, label=CASE_LABELS[case], zorder=3)

        ax.axhline(10.0, color="#475569", linewidth=0.9, linestyle="--", alpha=0.6)
        ax.text(ax.get_xlim()[0]+20, 10.05, "ρ = 10 g/cc ceiling",
                fontsize=6.5, color="#64748B")
        ax.set_xlabel("Melting Point — ROM (°C)", fontsize=9)
        ax.set_ylabel("Density (g/cc)", fontsize=9)
        ax.legend(fontsize=7, facecolor="#1E293B", edgecolor="#334155",
                  labelcolor="#F1F5F9")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("High Tm + low density = upper-left. W additions push density up while raising Tm.")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Row 2: VEC vs δ  |  GVI distribution ─────────────────────────────────
    r2a, r2b = st.columns(2, gap="large")

    with r2a:
        st.markdown(
            "<div class='section-eyebrow'>Phase stability descriptors</div>"
            "<div class='section-title'>VEC vs δ Stability Map</div>",
            unsafe_allow_html=True)

        fig, ax = dark_fig(5.5, 4)

        # Background shading: GVI = 0.5 contour
        vec_grid  = np.linspace(4.0, 7.0, 200)
        delta_grid = np.linspace(0.0, 12.0, 200)
        VV, DD = np.meshgrid(vec_grid, delta_grid)
        s_vec_g   = 1 / (1 + np.exp(15 * (VV - 5.3)))
        s_delta_g = 1 / (1 + np.exp(10 * (DD - 6.5)))
        GVI_grid  = s_vec_g * s_delta_g
        ax.contourf(VV, DD, GVI_grid, levels=[0, GVI_THRESHOLD, 1.0],
                    colors=["#450a0a", "#064e3b"], alpha=0.25, zorder=1)
        ax.contour(VV, DD, GVI_grid, levels=[GVI_THRESHOLD],
                   colors=["#94A3B8"], linewidths=1.0, linestyles="--", zorder=2)
        ax.text(6.3, 1.0, "GVI < 0.5\n(risk zone)",
                fontsize=7, color="#fca5a5", ha="center")
        ax.text(4.5, 2.5, "GVI ≥ 0.5\n(viable)",
                fontsize=7, color="#86efac", ha="center")

        for case in [1, 2, 3]:
            rows = [d for d in alloy_data if d["case"] == case]
            ax.scatter(
                [r["VEC"]   for r in rows],
                [r["delta"] for r in rows],
                color=CASE_COLORS[case], s=55,
                edgecolors="white", linewidths=0.4,
                alpha=0.9, label=CASE_LABELS[case], zorder=4)

        ax.set_xlabel("VEC", fontsize=9)
        ax.set_ylabel("δ — atomic size mismatch (%)", fontsize=9)
        ax.legend(fontsize=7, facecolor="#1E293B", edgecolor="#334155",
                  labelcolor="#F1F5F9")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Shaded regions show GVI < 0.5 (secondary phase risk). Most engineering alloys cluster in the viable zone.")

    with r2b:
        st.markdown(
            "<div class='section-eyebrow'>Phase stability screening</div>"
            "<div class='section-title'>GVI Score Distribution</div>",
            unsafe_allow_html=True)

        fig, ax = dark_fig(5.5, 4)
        bins = np.linspace(0, 1, 21)
        for case in [1, 2, 3]:
            gvi_vals = [d["GVI"] for d in alloy_data if d["case"] == case]
            ax.hist(gvi_vals, bins=bins, alpha=0.65,
                    color=CASE_COLORS[case], label=CASE_LABELS[case],
                    edgecolor="#0F172A", linewidth=0.5)

        ax.axvline(GVI_THRESHOLD, color="white", linewidth=1.4,
                   linestyle="--", label=f"Threshold = {GVI_THRESHOLD}")
        ax.set_xlabel("GVI Score", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=7, facecolor="#1E293B", edgecolor="#334155",
                  labelcolor="#F1F5F9")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Most validated alloys score GVI > 0.9, confirming single-phase BCC structure consistent with CALPHAD.")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Row 3: Nb content  |  GVI component breakdown ────────────────────────
    r3a, r3b = st.columns(2, gap="large")

    with r3a:
        st.markdown(
            "<div class='section-eyebrow'>Alloy family boundaries</div>"
            "<div class='section-title'>Nb-Content Distribution</div>",
            unsafe_allow_html=True)

        fig, ax = dark_fig(5.5, 4)
        bins_nb = np.linspace(0, 100, 26)
        for case in [1, 2, 3]:
            nb_vals = [d["Nb_at"] for d in alloy_data if d["case"] == case]
            ax.hist(nb_vals, bins=bins_nb, alpha=0.65,
                    color=CASE_COLORS[case], label=CASE_LABELS[case],
                    edgecolor="#0F172A", linewidth=0.5)

        ax.axvline(50, color=C3_COL, linewidth=1.2, linestyle=":",
                   alpha=0.8, label="Case 2/3 boundary (50 at%)")
        ax.text(51, ax.get_ylim()[1]*0.85, "Case 3\nboundary",
                fontsize=6.5, color=C3_COL)
        ax.set_xlabel("Nb content (at%)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=7, facecolor="#1E293B", edgecolor="#334155",
                  labelcolor="#F1F5F9")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Case 1 and 2 are Nb-dominated (>50 at% Nb). Case 3 covers multi-principal alloys where Nb is one of several major elements.")

    with r3b:
        st.markdown(
            "<div class='section-eyebrow'>GVI component breakdown</div>"
            "<div class='section-title'>Why Alloys Pass or Fail GVI</div>",
            unsafe_allow_html=True)

        fig, ax = dark_fig(5.5, 4)

        s_vec_all   = [d["S_VEC"]   for d in alloy_data]
        s_delta_all = [d["S_delta"] for d in alloy_data]
        s_sr_all    = [d["S_SR"]    for d in alloy_data if d["S_SR"] is not None]

        bins_s = np.linspace(0, 1, 16)
        ax.hist(s_vec_all,   bins=bins_s, alpha=0.70, color=C1_COL,
                label="S_VEC  (valence electron score)", edgecolor="#0F172A", lw=0.5)
        ax.hist(s_delta_all, bins=bins_s, alpha=0.70, color=C2_COL,
                label="S_δ    (size mismatch score)",   edgecolor="#0F172A", lw=0.5)
        if s_sr_all:
            ax.hist(s_sr_all, bins=bins_s, alpha=0.70, color=C3_COL,
                    label="S_SR   (sponge ratio score, Case 2)", edgecolor="#0F172A", lw=0.5)

        ax.axvline(0.5, color="white", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.set_xlabel("Component Score (logistic)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=7, facecolor="#1E293B", edgecolor="#334155",
                  labelcolor="#F1F5F9")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("S_δ is rarely the bottleneck for Nb alloys. S_VEC drops for W/Mo-heavy alloys (VEC > 5.3). S_SR applies only in Case 2.")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Row 4: SR vs DBTT (Case 2)  |  EI classification (Case 3) ───────────
    r4a, r4b = st.columns(2, gap="large")

    with r4a:
        st.markdown(
            "<div class='section-eyebrow'>Case 2 model — α = 0</div>"
            "<div class='section-title'>Sponge Ratio vs DBTT</div>",
            unsafe_allow_html=True)

        fig, ax = dark_fig(5.5, 4)

        c2_rows = [d for d in alloy_data if d["case"] == 2]
        sr_vals  = [d["SR"]       for d in c2_rows]
        exp_vals = [d["exp_dbtt"] for d in c2_rows]
        names_c2 = [d["name"]     for d in c2_rows]

        # Polynomial trend
        if len(sr_vals) >= 3:
            z = np.polyfit(sr_vals, exp_vals, 2)
            p = np.poly1d(z)
            sr_fit = np.linspace(min(sr_vals)-0.05, max(sr_vals)+0.05, 200)
            ax.plot(sr_fit, p(sr_fit), color="#94A3B8", linestyle="--",
                    linewidth=1.4, label="2nd-order trend", zorder=2)

        ax.scatter(sr_vals, exp_vals, color=C2_COL, s=70,
                   edgecolors="white", linewidths=0.5, zorder=3)
        for r in c2_rows:
            ax.annotate(r["name"], (r["SR"], r["exp_dbtt"]),
                        textcoords="offset points", xytext=(4, 3),
                        fontsize=5.5, color="#94A3B8")

        ax.axhline(0, color="#475569", linewidth=0.8, linestyle="-", alpha=0.5)
        ax.text(max(sr_vals)*0.6, 5, "0°C reference", fontsize=6.5, color="#64748B")
        ax.set_xlabel("SR = W / (Hf+Zr+Ti+1)  [at%,  α=0]", fontsize=9)
        ax.set_ylabel("Experimental DBTT (°C)", fontsize=9)
        ax.legend(fontsize=7, facecolor="#1E293B", edgecolor="#334155",
                  labelcolor="#F1F5F9")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("SR captures the ratio of embrittlers (W) to ductilisers (Hf, Zr, Ti). α=0 means Mo does not enter the SR in the RCCA regime — the central finding.")

    with r4b:
        st.markdown(
            "<div class='section-eyebrow'>Case 3 model — α = 0.48</div>"
            "<div class='section-title'>EI Classification</div>",
            unsafe_allow_html=True)

        fig, ax = dark_fig(5.5, 4)

        c3_rows = [d for d in alloy_data if d["case"] == 3]
        ei_vals  = [d["EI"]   for d in c3_rows]
        names_c3 = [d["name"] for d in c3_rows]
        zones    = [d["zone"] for d in c3_rows]
        z_colors = [ZONE_COL[z] for z in zones]

        # Sort by EI
        order    = np.argsort(ei_vals)
        ei_s     = [ei_vals[i]  for i in order]
        names_s  = [names_c3[i] for i in order]
        zones_s  = [zones[i]    for i in order]
        colors_s = [ZONE_COL[z] for z in zones_s]
        ei_safe  = [max(e, 1e-4) for e in ei_s]

        ax.bar(range(len(ei_s)), ei_safe, color=colors_s,
               edgecolor="#0F172A", linewidth=0.4, width=0.7)
        ax.set_yscale("log")

        for threshold, label in [
            (EI_DUCTILE,    f"EI={EI_DUCTILE}"),
            (EI_TRANSITION, f"EI={EI_TRANSITION}"),
            (EI_BRITTLE,    f"EI={EI_BRITTLE}"),
        ]:
            ax.axhline(threshold, color="#475569", linewidth=0.8,
                       linestyle="--", alpha=0.7)
            ax.text(len(ei_s)-0.5, threshold*1.1, label,
                    fontsize=6, color="#64748B", ha="right")

        ax.set_xticks(range(len(names_s)))
        ax.set_xticklabels(names_s, rotation=70, ha="right",
                           fontsize=5.5, color="#94A3B8")
        ax.set_ylabel("EI  (log scale)", fontsize=9)

        patches = [mpatches.Patch(color=ZONE_COL[z], label=z)
                   for z in [DUCTILE, TRANSITION, BRITTLE, CONFIRMED]]
        ax.legend(handles=patches, fontsize=6.5, facecolor="#1E293B",
                  edgecolor="#334155", labelcolor="#F1F5F9",
                  loc="upper left")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("EI = (W + 0.48·Mo) / (Hf+Zr+Ti+1). Most RHEAs without W or Mo fall naturally in the Ductile Zone.")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Footer note ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='color:{MUTED};font-size:0.78rem;text-align:center;padding:0.8rem 0 1.2rem 0;'>
      URADES v2.0 · Physics-based hierarchical screening for Nb-based refractory alloys ·
      Case 1 IAS (n=23) · Case 2 SR α=0 (n=10 LOOCV) · Case 3 EI α=0.48 (n=25)
    </div>
    """, unsafe_allow_html=True)
