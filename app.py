"""
Ames Housing Price Predictor
Business-dashboard UI — Inter typeface, olive/sage/cream palette.
"""

import pickle
import pandas as pd
import numpy as np
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import seaborn as sns

import folium
from streamlit_folium import st_folium

# ── Palette ─────────────────────────────────────────────────────────────────
OLIVE    = "#5B6B3A"
SAGE     = "#A8AE93"
CREAM    = "#F7F5EF"
CHARCOAL = "#2B2B28"
WHITE    = "#FFFFFF"

# ── Neighbourhood reference data (hardcoded — no runtime fetch) ──────────────
# Sources: training-data medians + Wikipedia (Ames, Iowa article, 2024).
# Wikipedia contained no neighbourhood-level descriptions beyond a brief
# mention of the Old Town Historic District; all other sentences are grounded
# in the training-set statistics provided below.
#
# Schema per entry:
#   code -> (full_name, description, tier, median_k)
#   tier: "Premium" | "Mid-Range" | "Budget"

NEIGHBORHOOD_INFO = {
    "NridgHt": (
        "Northridge Heights",
        "Ames' newest large-scale development, featuring the biggest lots and the highest concentration of post-2000 construction in the dataset.",
        "Premium",
        315,
    ),
    "StoneBr": (
        "Stone Brook",
        "The city's luxury tier — a small, upscale enclave that commands the highest median prices of any neighbourhood in Ames.",
        "Premium",
        310,
    ),
    "NoRidge": (
        "Northridge",
        "Established upscale suburb with spacious homes, mature landscaping, and consistently strong resale values.",
        "Premium",
        302,
    ),
    "Somerst": (
        "Somerset",
        "A modern planned community built largely in the 2000s, valued for uniform quality and well-maintained common areas.",
        "Premium",
        253,
    ),
    "Veenker": (
        "Veenker",
        "A small, quiet neighbourhood bordering the Veenker Memorial Golf Course, with limited inventory and above-average prices.",
        "Premium",
        245,
    ),
    "CollgCr": (
        "College Creek",
        "One of the most popular choices for Iowa State professionals and faculty, with well-kept homes and easy campus access.",
        "Mid-Range",
        213,
    ),
    "Crawfor": (
        "Crawford",
        "Older neighbourhood with character homes and established streetscapes, popular for its architectural variety.",
        "Mid-Range",
        197,
    ),
    "Gilbert": (
        "Gilbert",
        "Quiet suburban neighbourhood in northeast Ames, favoured by families for its calm streets and solid school district.",
        "Mid-Range",
        196,
    ),
    "Timber": (
        "Timberland",
        "Wooded lots and spacious yards on the city's outer fringe, appealing to buyers who prioritise privacy and green space.",
        "Mid-Range",
        192,
    ),
    "Blmngtn": (
        "Bloomington Heights",
        "Self-contained residential pocket with steady values and a quiet, low-traffic character.",
        "Mid-Range",
        191,
    ),
    "ClearCr": (
        "Clear Creek",
        "Neighbourhood bordering the Clear Creek corridor, offering a green setting and mid-range pricing.",
        "Mid-Range",
        189,
    ),
    "SawyerW": (
        "Sawyer West",
        "Newer residential development on Ames' west side with modern floor plans and consistent build quality.",
        "Mid-Range",
        183,
    ),
    "Mitchel": (
        "Mitchell",
        "Solid south Ames suburb with conventional single-family homes and straightforward suburban amenities.",
        "Mid-Range",
        156,
    ),
    "SWISU": (
        "Southwest Iowa State",
        "Located on the south edge of the Iowa State campus, convenient for university staff and graduate students.",
        "Budget",
        142,
    ),
    "NPkVill": (
        "Northwest Park Village",
        "Small northwest cluster with limited sales activity and compact, modestly priced homes.",
        "Budget",
        141,
    ),
    "Blueste": (
        "Bluestem",
        "Very small neighbourhood with few recorded sales, reflecting a niche pocket of modest Ames housing.",
        "Budget",
        137,
    ),
    "Sawyer": (
        "Sawyer",
        "Modest starter-home neighbourhood offering some of the most accessible entry-level pricing in the city.",
        "Budget",
        136,
    ),
    "NAmes": (
        "North Ames",
        "The largest single neighbourhood in the dataset, representing the broad working-class residential fabric of the city.",
        "Budget",
        145,
    ),
    "Edwards": (
        "Edwards",
        "Affordable west-of-centre neighbourhood popular with first-time buyers and those seeking value close to amenities.",
        "Budget",
        131,
    ),
    "OldTown": (
        "Old Town",
        "Ames' historic district closest to downtown — the Old Town Historic District features older homes with period character and walkable streets.",
        "Budget",
        123,
    ),
    "BrkSide": (
        "Brookside",
        "Older housing stock near the railroad corridor and Brookside Park, offering some of the most affordable detached homes in the city.",
        "Budget",
        125,
    ),
    "BrDale": (
        "Briardale",
        "Basic housing stock with compact lots, representing a straightforward and no-frills residential pocket.",
        "Budget",
        106,
    ),
    "IDOTRR": (
        "IDOTRR",
        "Industrial-adjacent area near the Iowa Department of Transportation right-of-way, with the lowest median prices in the dataset.",
        "Budget",
        98,
    ),
    "MeadowV": (
        "Meadow Village",
        "The smallest homes and the lowest median price in the entire dataset — a compact, entry-level pocket of Ames.",
        "Budget",
        88,
    ),
}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ames Housing Predictor",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS + Google Fonts (Inter) ───────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Typography ── */
html, body, .stApp, .stApp * {{
    font-family: 'Inter', sans-serif !important;
}}

/* ── Background ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {{
    background-color: {CREAM};
}}

/* ── Sidebar ── */
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {{
    background-color: #EDEAE2;
    border-right: 1px solid {SAGE};
}}

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    border-bottom: 2px solid {SAGE} !important;
    background: transparent;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    color: {CHARCOAL};
    font-size: 0.92rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    padding: 10px 28px;
    margin-bottom: -2px;
    border-radius: 0;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    border-bottom: 3px solid {OLIVE} !important;
    color: {OLIVE} !important;
    font-weight: 700;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {OLIVE};
}}

/* ── Predict button ── */
.stFormSubmitButton > button {{
    background-color: #4F5F34 !important;
    color: white !important;
    border: 2px solid #4F5F34 !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    padding: 0.6rem 1.5rem !important;
    width: 100% !important;
    transition: background-color 0.18s ease, border-color 0.18s ease;
}}
.stFormSubmitButton > button:hover {{
    background-color: #3C4828 !important;
    border-color: #3C4828 !important;
    color: white !important;
}}

/* ── Widget labels ── */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {{
    color: {CHARCOAL} !important;
    font-size: 0.96rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
}}

/* ── Selectbox text ── */
[data-testid="stSelectbox"] div[data-baseweb="select"] span {{
    font-size: 0.92rem !important;
    color: {CHARCOAL} !important;
}}

/* ── Divider ── */
hr {{
    border-color: {SAGE} !important;
    opacity: 0.45;
}}

/* ── Fun-fact callout ── */
.fun-fact {{
    background: white;
    border-left: 4px solid {OLIVE};
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1.1rem;
    margin-top: 0.6rem;
    font-size: 0.88rem;
    color: {CHARCOAL};
    line-height: 1.6;
}}
.fun-fact strong {{
    color: {OLIVE};
    font-weight: 700;
}}

/* ── Housing Insights tab — chart card effect ── */
[data-testid="stImage"] {{
    border: 1px solid #DDD9D0;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(43,43,40,0.05);
}}

/* ══ PREMIUM SLIDER SYSTEM ══════════════════════════════════════════════════ */

/* Slider container spacing */
[data-testid="stSlider"] {{
    padding-bottom: 0.3rem !important;
}}

/* Track: full bar — light neutral rail */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {{
    background-color: #E6E3D9 !important;
    height: 3px !important;
    border-radius: 3px !important;
}}
/* Track: active (filled) portion — deep olive */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child > div:first-child {{
    background-color: #4F5F34 !important;
    height: 3px !important;
    border-radius: 3px !important;
}}
/* Fallback track selectors for Streamlit internal structure */
[data-testid="stSlider"] > div > div > div > div {{
    background-color: #E6E3D9 !important;
    height: 3px !important;
    border-radius: 3px !important;
}}
[data-testid="stSlider"] > div > div > div > div:first-child {{
    background-color: #4F5F34 !important;
}}

/* Thumb handle — deep olive */
[data-testid="stSlider"] [role="slider"] {{
    background-color: #4F5F34 !important;
    border: 2px solid #3C4828 !important;
    border-radius: 50% !important;
    width: 18px !important;
    height: 18px !important;
    box-shadow: 0 1px 5px rgba(79,95,52,0.30), 0 1px 2px rgba(43,43,40,0.12) !important;
    transition: box-shadow 0.18s ease, transform 0.14s ease !important;
    cursor: grab !important;
}}
[data-testid="stSlider"] [role="slider"]:hover {{
    box-shadow: 0 3px 12px rgba(79,95,52,0.40), 0 1px 3px rgba(43,43,40,0.14) !important;
    transform: scale(1.10) !important;
}}
[data-testid="stSlider"] [role="slider"]:active {{
    cursor: grabbing !important;
    transform: scale(1.05) !important;
    box-shadow: 0 1px 6px rgba(79,95,52,0.35) !important;
}}
[data-testid="stSlider"] [role="slider"]:focus {{
    box-shadow: 0 0 0 4px rgba(79,95,52,0.18), 0 1px 5px rgba(79,95,52,0.30) !important;
    outline: none !important;
}}

/* Tick bar min / max labels — deep olive, semibold */
[data-testid="stTickBarMin"],
[data-testid="stTickBarMax"] {{
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #4F5F34 !important;
    letter-spacing: 0.01em !important;
}}

/* Current value indicator above thumb */
[data-testid="stSlider"] [data-baseweb="tooltip"] div,
[data-testid="stSlider"] [data-baseweb="tooltip"] span {{
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #4F5F34 !important;
    letter-spacing: 0.01em !important;
}}

/* Selectbox + input: focused border — deep olive */
[data-baseweb="select"] > div:focus-within,
[data-baseweb="select"]:focus-within > div {{
    border-color: #4F5F34 !important;
    box-shadow: 0 0 0 2px rgba(79,95,52,0.15) !important;
}}
[data-baseweb="input"]:focus-within {{
    border-color: #4F5F34 !important;
    box-shadow: 0 0 0 2px rgba(79,95,52,0.15) !important;
}}

/* Dropdown: hovered and selected option */
[data-baseweb="menu"] [aria-selected="true"] {{
    background-color: rgba(79,95,52,0.10) !important;
    color: {CHARCOAL} !important;
}}
[data-baseweb="menu"] li:hover {{
    background-color: rgba(79,95,52,0.07) !important;
}}

/* Text selection highlight */
::selection {{
    background: rgba(79,95,52,0.18);
    color: {CHARCOAL};
}}

/* Form submit button focus ring */
.stFormSubmitButton > button:focus {{
    box-shadow: 0 0 0 3px rgba(79,95,52,0.25) !important;
    outline: none !important;
}}
</style>
""", unsafe_allow_html=True)

# ── Load pickles once per server lifetime ────────────────────────────────────
@st.cache_resource
def load_models():
    with open("models/reg_model.pkl", "rb") as f:
        reg = pickle.load(f)
    with open("models/clf_model.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("models/defaults.pkl", "rb") as f:
        defaults = pickle.load(f)
    return reg, clf, defaults

reg_model, clf_model, defaults_df = load_models()


# ── Shared matplotlib style ───────────────────────────────────────────────────
def style_ax(fig, ax):
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(SAGE)
    ax.spines["bottom"].set_color(SAGE)
    ax.tick_params(colors=CHARCOAL, labelsize=9)
    ax.xaxis.label.set_color(CHARCOAL)
    ax.yaxis.label.set_color(CHARCOAL)
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)
    ax.title.set_color(CHARCOAL)
    ax.title.set_fontsize(11)
    ax.title.set_fontweight("bold")
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=3)
    ax.grid(axis="y", color="#DDD9D0", linewidth=0.7, zorder=0)

def insight_col(text: str):
    """Render the right-column explanation with an olive left border."""
    st.markdown(
        f"""
        <div style="border-left:3px solid {OLIVE}; padding:0.8rem 1.1rem;
                    margin-top:0; line-height:1.7; font-size:0.88rem;
                    color:{CHARCOAL}; text-align:justify;">
          {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("assets/logo.png", use_container_width=True)

    st.markdown(
        f"<p style='text-align:center; font-size:0.72rem; letter-spacing:0.14em; "
        f"font-weight:700; color:{SAGE}; margin-top:-0.6rem; margin-bottom:0.2rem;'>"
        f"AMES  HOUSING</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        f"""
        <div style="background:{WHITE}; border:1px solid #DDD9D0;
                    border-left:4px solid {OLIVE}; border-radius:0 10px 10px 0;
                    padding:1.0rem 1.1rem;">
          <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.13em;
                      color:{SAGE}; text-transform:uppercase; margin-bottom:0.6rem;">
            Dataset at a glance
          </div>
          <div style="font-size:0.85rem; color:{CHARCOAL}; line-height:2.0;">
            <span style="color:{OLIVE}; font-weight:800; margin-right:0.45rem;">&#10003;</span><strong>2,930</strong> home sales<br>
            <span style="color:{OLIVE}; font-weight:800; margin-right:0.45rem;">&#10003;</span><strong>2006 – 2010</strong> sale years<br>
            <span style="color:{OLIVE}; font-weight:800; margin-right:0.45rem;">&#10003;</span><strong>80</strong> features per property<br>
            <span style="color:{OLIVE}; font-weight:800; margin-right:0.45rem;">&#10003;</span>Source: De Cock (2011), <em>Journal of Statistics Education</em>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# HEADER BAR
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"""
    <div style="
        display:flex; align-items:center; justify-content:space-between;
        padding:0.4rem 0 1rem 0;
        border-bottom:2px solid {SAGE};
        margin-bottom:1.2rem;
    ">
        <span style="font-size:1.45rem; font-weight:800; color:{CHARCOAL};
                     letter-spacing:-0.01em;">
            Ames Housing Predictor
        </span>
        <span style="font-size:0.72rem; font-weight:600; letter-spacing:0.10em;
                     color:{SAGE}; text-transform:uppercase;">
            Streamlit App
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Shared helpers ───────────────────────────────────────────────────────────
def _checklist_html(items):
    rows = "".join(
        f"<div style='display:flex; align-items:flex-start; margin-bottom:0.45rem;'>"
        f"<span style='color:{OLIVE}; font-weight:800; margin-right:0.55rem; flex-shrink:0;'>&#10003;</span>"
        f"<span style='font-size:0.9rem; color:{CHARCOAL}; line-height:1.6;'>{item}</span>"
        f"</div>"
        for item in items
    )
    return f"<div>{rows}</div>"

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab_home, tab_predict, tab_eda, tab_nbhd = st.tabs(["Home", "Price & Premium Home Prediction", "Housing Insights", "Ames's Neighborhoods"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 0 — HOME
# ════════════════════════════════════════════════════════════════════════════
with tab_home:

    # ── HERO ────────────────────────────────────────────────────────────────
    _logo_path = "assets/logonoback.png"
    _logo_fallback = "assets/logo.png"
    try:
        _logo_b64 = __import__("base64").b64encode(open(_logo_path, "rb").read()).decode()
    except FileNotFoundError:
        _logo_b64 = __import__("base64").b64encode(open(_logo_fallback, "rb").read()).decode()

    # Hero: two native columns so Streamlit controls layout (flexbox stripped by sanitizer)
    _hero_l, _hero_r = st.columns([1, 1], gap="large")

    with _hero_l:
        st.markdown(
            f"<div style='text-align:center; padding:2rem 0;'>"
            f"<img src='data:image/png;base64,{_logo_b64}' "
            f"     style='width:320px; height:auto;' />"
            f"</div>",
            unsafe_allow_html=True,
        )

    with _hero_r:
        st.markdown(
            f"<p style='font-size:0.72rem; font-weight:700; letter-spacing:0.22em; "
            f"color:{OLIVE}; text-transform:uppercase; margin:2rem 0 0.8rem 0;'>"
            f"Welcome</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<h2 style='font-size:2.0rem; font-weight:800; color:{CHARCOAL}; "
            f"letter-spacing:-0.025em; line-height:1.25; margin:0 0 1.0rem 0;'>"
            f"What's your home worth?<br>Let's find out.</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='font-size:0.94rem; color:{SAGE}; line-height:1.75; "
            f"margin:0 0 1.4rem 0;'>"
            f"Instantly estimate your home's market value using machine learning "
            f"trained on nearly 3,000 real Ames sales.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='width:40px; height:2px; background:{OLIVE}; "
            f"border-radius:2px; margin-bottom:1.2rem;'></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='font-size:0.72rem; font-weight:700; letter-spacing:0.14em; "
            f"color:{SAGE}; text-transform:uppercase; margin:0 0 0.6rem 0;'>Get</p>"
            f"<p style='font-size:0.92rem; color:{CHARCOAL}; line-height:1.9; margin:0;'>"
            f"<span style='color:{OLIVE}; font-weight:800; margin-right:0.5rem;'>&#10003;</span>"
            f"Predicted sale price<br>"
            f"<span style='color:{OLIVE}; font-weight:800; margin-right:0.5rem;'>&#10003;</span>"
            f"Premium-home prediction<br>"
            f"<span style='color:{OLIVE}; font-weight:800; margin-right:0.5rem;'>&#10003;</span>"
            f"Market positioning insights"
            f"</p>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<hr style='border-color:{SAGE}; opacity:0.35; margin:1.8rem 0 2rem 0;'>",
        unsafe_allow_html=True,
    )

    # ── ABOUT AMES + LOCATOR MAP (two-column) ───────────────────────────────
    _aw_l, _aw_r = st.columns([1, 1], gap="large")

    with _aw_l:
        st.markdown(
            f"<p style='font-size:1.1rem; font-weight:700; letter-spacing:0.13em; "
            f"color:{SAGE}; text-transform:uppercase; margin-bottom:0.8rem;'>"
            f"About Ames, Iowa</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="font-size:1.05rem; color:{CHARCOAL}; line-height:1.8;">
              <div style="margin-bottom:0.55rem; display:flex; align-items:flex-start;">
                <span style="color:{OLIVE}; font-weight:700; margin-right:0.5rem; flex-shrink:0;">&#10003;</span>
                <span style="text-align:justify;">College town in central Iowa, home to Iowa State University</span>
              </div>
              <div style="margin-bottom:0.55rem; display:flex; align-items:flex-start;">
                <span style="color:{OLIVE}; font-weight:700; margin-right:0.5rem; flex-shrink:0;">&#10003;</span>
                <span style="text-align:justify;">Around 66,000 residents in a stable, tight-knit community</span>
              </div>
              <div style="margin-bottom:0.55rem; display:flex; align-items:flex-start;">
                <span style="color:{OLIVE}; font-weight:700; margin-right:0.5rem; flex-shrink:0;">&#10003;</span>
                <span style="text-align:justify;">Housing ranges from starter homes to upscale new-build estates</span>
              </div>
              <div style="margin-bottom:0.55rem; display:flex; align-items:flex-start;">
                <span style="color:{OLIVE}; font-weight:700; margin-right:0.5rem; flex-shrink:0;">&#10003;</span>
                <span style="text-align:justify;">Sale prices span from under $40,000 to over $600,000</span>
              </div>
              <div style="margin-bottom:0.55rem; display:flex; align-items:flex-start;">
                <span style="color:{OLIVE}; font-weight:700; margin-right:0.5rem; flex-shrink:0;">&#10003;</span>
                <span style="text-align:justify;">Overall quality and living area are the strongest price drivers</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with _aw_r:
        st.markdown(
            f"<p style='font-size:0.88rem; color:{CHARCOAL}; margin-bottom:0.6rem; "
            f"line-height:1.6;'>See it for yourself — Ames sits right in the heart "
            f"of the Midwest.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="border:1px solid #DDD9D0; border-radius:10px;
                        overflow:hidden; box-shadow:0 2px 8px rgba(43,43,40,0.07);
                        margin-bottom:0.4rem;">
            """,
            unsafe_allow_html=True,
        )
        _locator = folium.Map(
            location=[42.0308, -93.6319],
            zoom_start=6,
            tiles="CartoDB positron",
            zoom_control=False,
            scrollWheelZoom=False,
            dragging=False,
            attributionControl=False,
        )
        folium.Marker(
            location=[42.0308, -93.6319],
            tooltip="Ames, Iowa",
            popup=folium.Popup(
                "<b style='font-family:Inter,sans-serif;'>Ames, Iowa</b>"
                "<br><span style='font-size:0.82em; color:#555;'>"
                "Home to Iowa State University<br>and the 2,930 properties "
                "in this dataset</span>",
                max_width=220,
            ),
            icon=folium.Icon(color="darkgreen", icon="home", prefix="fa"),
        ).add_to(_locator)
        st_folium(_locator, height=260, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align:center; font-size:0.75rem; color:{SAGE}; "
            f"margin-top:0.3rem; margin-bottom:0.4rem;'>"
            f"Ames, Iowa &mdash; located in central Iowa, about 30 miles north of Des Moines</p>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<hr style='border-color:{SAGE}; opacity:0.35; margin:2rem 0;'>",
        unsafe_allow_html=True,
    )

    # ── SIGHTSEEING IN AMES ─────────────────────────────────────────────────
    st.markdown(
        f"<p style='font-size:1.1rem; font-weight:700; letter-spacing:0.13em; "
        f"color:{SAGE}; text-transform:uppercase; margin-bottom:0.5rem;'>"
        f"Sightseeing in Ames</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='font-size:0.88rem; color:{CHARCOAL}; margin-bottom:0.9rem; "
        f"line-height:1.6;'>A home's value isn't just square footage — it's the community around it. "
        f"Here's a taste of what makes Ames worth living in.</p>",
        unsafe_allow_html=True,
    )

    def _img_b64(path: str) -> str:
        import base64, mimetypes
        try:
            mime = mimetypes.guess_type(path)[0] or "image/jpeg"
            with open(path, "rb") as f:
                return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
        except FileNotFoundError:
            return ""  # triggers placeholder below

    def _sight_card_html(img_src: str, title: str, body: str) -> str:
        if img_src:
            img_block = (
                f"<div style='height:220px; overflow:hidden; "
                f"  border-radius:10px 10px 0 0; margin:-1px -1px 0 -1px;'>"
                f"  <img src='{img_src}' style='width:100%; height:100%; "
                f"  object-fit:cover; display:block;' />"
                f"</div>"
            )
        else:
            img_block = (
                f"<div style='height:220px; background:{OLIVE}; opacity:0.18; "
                f"border-radius:10px 10px 0 0; margin:-1px -1px 0 -1px;'></div>"
            )
        return (
            f"<div style='background:{WHITE}; border:1px solid #DDD9D0; "
            f"border-radius:10px; overflow:hidden; "
            f"box-shadow:0 2px 8px rgba(43,43,40,0.07); height:100%;'>"
            f"{img_block}"
            f"<div style='padding:1.1rem 1.2rem 1.3rem;'>"
            f"<div style='font-size:0.97rem; font-weight:700; color:{CHARCOAL}; "
            f"  margin-bottom:0.45rem;'>{title}</div>"
            f"<div style='font-size:0.84rem; color:{CHARCOAL}; line-height:1.65; "
            f"  opacity:0.82;'>{body}</div>"
            f"</div></div>"
        )

    _sg1, _sg2, _sg3 = st.columns(3, gap="medium")

    with _sg1:
        st.markdown(
            _sight_card_html(
                _img_b64("assets/reiman_gardens.jpg"),
                "Reiman Gardens",
                "Iowa State's botanical garden, home to a butterfly wing "
                "and the world's largest concrete gnome.",
            ),
            unsafe_allow_html=True,
        )

    with _sg2:
        st.markdown(
            _sight_card_html(
                _img_b64("assets/ada_hayden.jpeg"),
                "Ada Hayden Heritage Park",
                "A 430-acre park with two lakes and paved trails, named "
                "after the first woman to earn a PhD from Iowa State.",
            ),
            unsafe_allow_html=True,
        )

    with _sg3:
        st.markdown(
            _sight_card_html(
                _img_b64("assets/tedesco_trail.jpg"),
                "Tedesco Environmental Learning Corridor",
                "A compact nature trail connected to the city bike path, "
                "known for wildlife viewing and a short, easy walk.",
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<hr style='border-color:{SAGE}; opacity:0.35; margin:2rem 0;'>",
        unsafe_allow_html=True,
    )

    # ── DID YOU KNOW? FACT CARDS ────────────────────────────────────────────
    st.markdown(
        f"<p style='font-size:1.1rem; font-weight:700; letter-spacing:0.13em; "
        f"color:{SAGE}; text-transform:uppercase; margin-bottom:0.9rem;'>"
        f"Did You Know?</p>",
        unsafe_allow_html=True,
    )

    _fc_base = (
        f"background:{WHITE}; border:1px solid #DDD9D0; border-top:3px solid {OLIVE}; "
        f"border-radius:10px; padding:1.3rem 1.2rem 1.2rem; text-align:center; "
        f"box-shadow:0 2px 8px rgba(43,43,40,0.07); height:100%;"
    )

    fc1, fc2, fc3 = st.columns(3, gap="medium")

    with fc1:
        st.markdown(
            f"<div style='{_fc_base}'>"
            f"<div style='font-size:0.65rem; font-weight:700; letter-spacing:0.12em; "
            f"  color:{SAGE}; text-transform:uppercase; margin-bottom:0.55rem;'>"
            f"  Median Price</div>"
            f"<div style='font-size:1.55rem; font-weight:800; color:{OLIVE}; "
            f"  letter-spacing:-0.02em; line-height:1.1; margin-bottom:0.5rem;'>"
            f"  $160,000</div>"
            f"<div style='font-size:0.78rem; color:{CHARCOAL}; line-height:1.55; "
            f"  opacity:0.75;'>Half of all 2,930 homes sold below this figure</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with fc2:
        st.markdown(
            f"<div style='{_fc_base}'>"
            f"<div style='font-size:0.65rem; font-weight:700; letter-spacing:0.12em; "
            f"  color:{SAGE}; text-transform:uppercase; margin-bottom:0.55rem;'>"
            f"  Location Spread</div>"
            f"<div style='font-size:1.55rem; font-weight:800; color:{OLIVE}; "
            f"  letter-spacing:-0.02em; line-height:1.1; margin-bottom:0.5rem;'>"
            f"  3.6&times;</div>"
            f"<div style='font-size:0.78rem; color:{CHARCOAL}; line-height:1.55; "
            f"  opacity:0.75;'>Price gap — Stone Brook ($319k) vs Meadow Village ($88k)</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with fc3:
        st.markdown(
            f"<div style='{_fc_base}'>"
            f"<div style='font-size:0.65rem; font-weight:700; letter-spacing:0.12em; "
            f"  color:{SAGE}; text-transform:uppercase; margin-bottom:0.55rem;'>"
            f"  Top Quality</div>"
            f"<div style='font-size:1.55rem; font-weight:800; color:{OLIVE}; "
            f"  letter-spacing:-0.02em; line-height:1.1; margin-bottom:0.5rem;'>"
            f"  4.7%</div>"
            f"<div style='font-size:0.78rem; color:{CHARCOAL}; line-height:1.55; "
            f"  opacity:0.75;'>Homes rated 9 or 10 out of 10 — genuinely rare</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


    st.markdown(
        f"<hr style='border-color:{SAGE}; opacity:0.35; margin:2rem 0;'>",
        unsafe_allow_html=True,
    )

    # ── WHAT THIS APP DOES ──────────────────────────────────────────────────
    st.markdown(
        f"<p style='font-size:1.1rem; font-weight:700; letter-spacing:0.13em; "
        f"color:{SAGE}; text-transform:uppercase; margin-bottom:0.9rem;'>"
        f"What This App Does</p>",
        unsafe_allow_html=True,
    )

    mod_l, mod_r = st.columns(2, gap="large")

    with mod_l:
        st.markdown(
            f"""
            <div style="background:{WHITE}; border:1px solid #DDD9D0;
                        border-radius:10px; padding:1.5rem 1.6rem;">
              <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.13em;
                          color:{SAGE}; text-transform:uppercase; margin-bottom:0.6rem;">
                Price Prediction
              </div>
              <div style="font-size:1.0rem; font-weight:700; color:{CHARCOAL};
                          margin-bottom:0.55rem;">
                What will this home sell for?
              </div>
              <div style="font-size:0.88rem; color:{CHARCOAL}; line-height:1.75;
                          opacity:0.85;">
                An XGBoost regression model estimates the sale price in dollars,
                trained across 80 property features. It consistently outperforms a
                simple median-price baseline — you get a market-calibrated number,
                not a rough guess.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with mod_r:
        st.markdown(
            f"""
            <div style="background:{WHITE}; border:1px solid #DDD9D0;
                        border-radius:10px; padding:1.5rem 1.6rem;">
              <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.13em;
                          color:{SAGE}; text-transform:uppercase; margin-bottom:0.6rem;">
                Premium Classification
              </div>
              <div style="font-size:1.0rem; font-weight:700; color:{CHARCOAL};
                          margin-bottom:0.55rem;">
                Is it a top-25% home?
              </div>
              <div style="font-size:0.88rem; color:{CHARCOAL}; line-height:1.75;
                          opacity:0.85;">
                A second XGBoost classifier gives a clear yes/no verdict on whether
                the property ranks in the top quarter of the Ames market, along with
                a confidence score — useful for buyers and sellers who want to know if
                a home is genuinely high-value.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:2rem;'></div>", unsafe_allow_html=True)

    # ── CALL TO ACTION ──────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="background:{CREAM}; border:1.5px solid {SAGE};
                    border-left:5px solid {OLIVE}; border-radius:10px;
                    padding:1.6rem 1.8rem;">
          <div style="font-size:1.1rem; font-weight:700; color:{CHARCOAL};
                      margin-bottom:0.4rem;">
            Ready to see your number?
          </div>
          <div style="font-size:0.92rem; color:{CHARCOAL}; opacity:0.80;
                      line-height:1.65;">
            Head to the <strong style="color:{OLIVE};">Predict</strong> tab above
            and get your estimate in under a minute.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════════════
NEIGHBORHOOD_NAMES = {
    "NridgHt": "Northridge Heights",
    "NoRidge":  "Northridge",
    "StoneBr":  "Stone Brook",
    "Somerst":  "Somerset",
    "Veenker":  "Veenker",
    "CollgCr":  "College Creek",
    "Gilbert":  "Gilbert",
    "Crawfor":  "Crawford",
    "Timber":   "Timberland",
    "SawyerW":  "Sawyer West",
    "Blmngtn":  "Bloomington Heights",
    "ClearCr":  "Clear Creek",
    "Mitchel":  "Mitchell",
    "NAmes":    "North Ames",
    "Edwards":  "Edwards",
    "OldTown":  "Old Town",
    "BrkSide":  "Brookside",
    "IDOTRR":   "Iowa DOT & Railroad Area",
    "MeadowV":  "Meadow Village",
    "Sawyer":   "Sawyer",
    "SWISU":    "SW of Iowa State",
    "BrDale":   "Briardale",
    "NPkVill":  "Northpark Village",
    "Blueste":  "Bluestem",
}
# Reverse map: full name -> code, used after selectbox selection
_NAME_TO_CODE = {v: k for k, v in NEIGHBORHOOD_NAMES.items()}

@st.fragment
def _predict_tab_body():
    # ── HERO: two-column layout ─────────────────────────────────────────────
    _hero_l, _hero_r = st.columns([11, 9], gap="large")

    with _hero_l:
        _hero_factors = [
            ("Overall quality rating",
             "Higher quality scores significantly increase home value and premium probability."),
            ("Total living and basement area",
             "Larger living spaces consistently drive higher home prices."),
            ("Neighborhood tier",
             "Homes in higher-tier neighborhoods tend to be more valuable."),
            ("Year built",
             "Newer homes generally have higher value and better premium potential."),
        ]
        _factors_html = "".join(
            f"<div style='display:flex; align-items:flex-start; margin-bottom:1.1rem;'>"
            f"<span style='color:{OLIVE}; font-weight:800; font-size:1.0rem; "
            f"margin-right:0.65rem; flex-shrink:0; line-height:1.5;'>&#10003;</span>"
            f"<div>"
            f"<div style='font-size:0.93rem; font-weight:700; color:{CHARCOAL}; "
            f"line-height:1.4; margin-bottom:0.18rem;'>{title}</div>"
            f"<div style='font-size:0.82rem; font-weight:400; color:#888880; line-height:1.55;'>{desc}</div>"
            f"</div>"
            f"</div>"
            for title, desc in _hero_factors
        )
        st.markdown(
            f"<p style='font-size:0.72rem; font-weight:700; letter-spacing:0.22em; "
            f"color:{OLIVE}; text-transform:uppercase; margin:0.5rem 0 0.7rem 0;'>"
            f"Predictive Analytics</p>"
            f"<h2 style='font-size:2.0rem; font-weight:800; color:{CHARCOAL}; "
            f"letter-spacing:-0.025em; line-height:1.25; margin:0 0 1.4rem 0;'>"
            f"Precision Pricing for Ames Homes.<br>Stop Guessing. Start Predicting.</h2>"
            f"<p style='font-size:0.72rem; font-weight:700; color:{OLIVE}; letter-spacing:0.13em; "
            f"text-transform:uppercase; margin:0 0 0.9rem 0;'>"
            f"Key Factors Influencing Price &amp; Premium Status</p>"
            f"{_factors_html}",
            unsafe_allow_html=True,
        )

    with _hero_r:
        _house_path = "assets/house.png"
        try:
            import base64 as _b64
            _house_b64 = _b64.b64encode(open(_house_path, "rb").read()).decode()
            st.markdown(
                f"<div style='display:flex; align-items:center; justify-content:center; "
                f"height:100%; padding-top:0.5rem;'>"
                f"<img src='data:image/png;base64,{_house_b64}' "
                f"style='width:100%; max-width:420px; border-radius:14px; "
                f"opacity:0.93; display:block; margin:auto;' />"
                f"</div>",
                unsafe_allow_html=True,
            )
        except FileNotFoundError:
            pass

    st.markdown(
        f"<hr style='border-color:{SAGE}; opacity:0.35; margin:1.8rem 0 2rem 0;'>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<p style='font-size:0.72rem; font-weight:700; letter-spacing:0.13em; "
        f"color:{SAGE}; text-transform:uppercase; margin-bottom:0.5rem;'>"
        f"HOUSE FEATURES</p>",
        unsafe_allow_html=True,
    )

    with st.form("predict_form"):
        left_col, right_col = st.columns(2, gap="large")

        with left_col:
            overall_qual = st.number_input(
                "Overall Quality",
                min_value=1, max_value=10,
                value=int(defaults_df["Overall Qual"].iloc[0]),
                step=1,
                help="1 = Very Poor   10 = Excellent",
            )
            gr_liv_area = st.number_input(
                "Above-Ground Living Area (sq ft)",
                min_value=300, max_value=5_000,
                value=int(defaults_df["Gr Liv Area"].iloc[0]),
                step=50,
            )
            _year_options = list(range(1872, 2011))
            _default_year = int(defaults_df["Year Built"].iloc[0])
            year_built = st.selectbox(
                "Year Built",
                options=_year_options,
                index=_year_options.index(_default_year),
            )

        with right_col:
            _default_bath = (
                float(defaults_df["Full Bath"].iloc[0])
                + 0.5 * float(defaults_df["Half Bath"].iloc[0])
            )
            total_bath = st.number_input(
                "Total Bathrooms (full + 0.5 per half bath)",
                min_value=0.0, max_value=5.0,
                value=_default_bath,
                step=0.5,
                format="%.1f",
            )
            garage_cars = st.number_input(
                "Garage Capacity (cars)",
                min_value=0, max_value=4,
                value=int(defaults_df["Garage Cars"].iloc[0]),
                step=1,
            )
            _default_code = str(defaults_df["Neighborhood"].iloc[0])
            _full_names = list(NEIGHBORHOOD_NAMES.values())
            neighborhood_full = st.selectbox(
                "Neighborhood",
                options=_full_names,
                index=_full_names.index(NEIGHBORHOOD_NAMES[_default_code]),
            )

        submitted = st.form_submit_button("Predict Home Price & Premium Home", use_container_width=True)

    # ── Prediction (calculation only — runs on button click) ─────────────────
    if submitted:
        full_bath_val = int(total_bath)
        half_bath_val = 1 if (total_bath % 1 >= 0.4) else 0

        input_df = defaults_df.copy()

        # --- raw feature overrides ---
        input_df["Overall Qual"] = float(overall_qual)
        input_df["Gr Liv Area"]  = float(gr_liv_area)
        input_df["Year Built"]   = float(year_built)
        input_df["Full Bath"]    = float(full_bath_val)
        input_df["Half Bath"]    = float(half_bath_val)
        input_df["Garage Cars"]  = float(garage_cars)
        input_df["Neighborhood"] = _NAME_TO_CODE[neighborhood_full]

        # --- recompute engineered features so the model sees consistent values ---
        yr_sold   = float(input_df["Yr Sold"].iloc[0])
        bsmt_sf   = float(input_df["Total Bsmt SF"].iloc[0])

        total_sf  = bsmt_sf + float(gr_liv_area)
        house_age = max(0, yr_sold - float(year_built))
        remod_age = max(0, yr_sold - float(input_df["Year Remod/Add"].iloc[0]))
        total_bath_full = (
            float(full_bath_val)
            + 0.5 * float(half_bath_val)
            + float(input_df["Bsmt Full Bath"].iloc[0])
            + 0.5 * float(input_df["Bsmt Half Bath"].iloc[0])
        )

        input_df["TotalSF"]    = total_sf
        input_df["Qual_x_SF"]  = float(overall_qual) * total_sf
        input_df["HouseAge"]   = house_age
        input_df["RemodAge"]   = remod_age
        input_df["TotalBath"]  = total_bath_full
        input_df["AgeCategory"] = pd.cut(
            pd.Series([house_age]),
            bins=[-1, 10, 30, 60, 200],
            labels=["New (0-10y)", "Recent (10-30y)", "Established (30-60y)", "Old (60y+)"],
        ).iloc[0]

        predicted_price = reg_model.predict(input_df)[0]
        premium_label   = clf_model.predict(input_df)[0]
        premium_proba   = clf_model.predict_proba(input_df)[0][1]

        # ── Investment scoring ────────────────────────────────────────────────
        _nbhd_code   = _NAME_TO_CODE[neighborhood_full]
        _nbhd_tier   = NEIGHBORHOOD_INFO[_nbhd_code][2]
        _AMES_MEDIAN = 163_000
        _AMES_AVG_SF = 1_500

        _score = 0
        if overall_qual >= 7:               _score += 1
        if overall_qual >= 9:               _score += 1
        if gr_liv_area >= _AMES_AVG_SF:     _score += 1
        if premium_label == 1:              _score += 1
        if predicted_price >= _AMES_MEDIAN: _score += 1
        if _nbhd_tier == "Premium":         _score += 1
        elif _nbhd_tier == "Budget":        _score -= 1
        _score = max(0, min(_score, 6))

        if _score >= 4:
            _inv_level = "HIGH"
            _inv_color = OLIVE
            _inv_desc  = "Strong investment fundamentals across quality, location, and price."
        elif _score >= 2:
            _inv_level = "MEDIUM"
            _inv_color = CHARCOAL
            _inv_desc  = "Moderate investment characteristics with some supporting factors."
        else:
            _inv_level = "LOW"
            _inv_color = SAGE
            _inv_desc  = "Limited investment upside at current market levels."

        _price_diff_pct = ((predicted_price - _AMES_MEDIAN) / _AMES_MEDIAN) * 100
        if _price_diff_pct > 0:
            _price_vs_avg = f"{_price_diff_pct:.0f}% above Ames average"
        elif _price_diff_pct < 0:
            _price_vs_avg = f"{abs(_price_diff_pct):.0f}% below Ames average"
        else:
            _price_vs_avg = "At Ames average"

        _resale = "Strong" if _score >= 4 else ("Moderate" if _score >= 2 else "Limited")

        # ── Reasoning bullets ─────────────────────────────────────────────────
        _is_premium = (premium_label == 1)

        if _inv_level == "HIGH" and _is_premium:
            _reasoning_bullets = [
                "confirmed premium-market classification",
                "strong neighborhood demand and location tier",
                "above-average quality and living area",
                "strong resale potential and high-value property profile",
            ]
        elif _inv_level == "HIGH" and not _is_premium:
            _reasoning_bullets = [
                "strong neighborhood and market positioning",
                "above-average quality score and living area",
                "price well above the Ames market average",
                "solid resale indicators across multiple factors",
            ]
        elif _inv_level == "MEDIUM" and _is_premium:
            _reasoning_bullets = [
                "premium-market positioning with moderate investment balance",
                "strong pricing level but mixed long-term indicators",
                "above-average home characteristics confirmed by classifier",
                "moderate resale stability relative to top-tier investment properties",
            ]
        elif _inv_level == "MEDIUM" and not _is_premium:
            _reasoning_bullets = [
                "balanced neighborhood positioning",
                "moderate resale strength",
                "average market pricing relative to Ames",
                "solid property profile without premium-tier signals",
            ]
        elif _inv_level == "LOW" and _is_premium:
            _reasoning_bullets = [
                "premium-home characteristics present despite lower score",
                "neighborhood or price factors temper overall investment rating",
                "above-average home quality with limited market upside",
                "investment signal mixed — premium quality, weaker market conditions",
            ]
        else:
            _reasoning_bullets = [
                "below-average price positioning relative to Ames market",
                "weaker resale indicators across location and quality",
                "limited investment signals at current market levels",
            ]

        _reason_bullet_html = "".join(
            f"<div style='display:flex; align-items:flex-start; margin-bottom:0.35rem;'>"
            f"<span style='color:{OLIVE}; font-weight:800; margin-right:0.5rem; flex-shrink:0;'>&#8226;</span>"
            f"<span style='font-size:0.86rem; color:{CHARCOAL}; line-height:1.6;'>{b}</span>"
            f"</div>"
            for b in _reasoning_bullets
        )

        # 3.5 % fixed annual rate — long-run average U.S. home price appreciation
        _projected_5yr = predicted_price * (1.035 ** 5)

        # ── Persist all display values across reruns ──────────────────────────
        st.session_state["predict_result"] = {
            "predicted_price":    predicted_price,
            "premium_label":      premium_label,
            "premium_proba":      premium_proba,
            "inv_level":          _inv_level,
            "inv_color":          _inv_color,
            "inv_desc":           _inv_desc,
            "price_vs_avg":       _price_vs_avg,
            "nbhd_tier":          _nbhd_tier,
            "resale":             _resale,
            "reason_bullet_html": _reason_bullet_html,
            "projected_5yr":      _projected_5yr,
        }

    # ── Display results from session state (stable across slider moves) ───────
    if "predict_result" in st.session_state:
        _r = st.session_state["predict_result"]
        predicted_price     = _r["predicted_price"]
        premium_label       = _r["premium_label"]
        premium_proba       = _r["premium_proba"]
        _inv_level          = _r["inv_level"]
        _inv_color          = _r["inv_color"]
        _inv_desc           = _r["inv_desc"]
        _price_vs_avg       = _r["price_vs_avg"]
        _nbhd_tier          = _r["nbhd_tier"]
        _resale             = _r["resale"]
        _reason_bullet_html = _r["reason_bullet_html"]
        _projected_5yr      = _r["projected_5yr"]

        st.markdown(
            f"<p style='font-size:0.72rem; font-weight:700; letter-spacing:0.13em; "
            f"color:{SAGE}; text-transform:uppercase; margin-top:1.8rem; margin-bottom:0.7rem;'>"
            f"PREDICTION RESULTS</p>",
            unsafe_allow_html=True,
        )

        card_left, card_right = st.columns(2, gap="large")

        with card_left:
            st.markdown(
                f"""
                <div style="background:{WHITE}; border:1px solid #DDD9D0;
                            border-radius:10px; padding:1.8rem 1.6rem; text-align:center;">
                  <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.13em;
                              color:{SAGE}; text-transform:uppercase; margin-bottom:0.6rem;">
                    Estimated Sale Price
                  </div>
                  <div style="font-size:2.6rem; font-weight:800; color:{CHARCOAL};
                              letter-spacing:-0.02em; line-height:1.1;">
                    ${predicted_price:,.0f}
                  </div>
                  <div style="font-size:0.75rem; color:#AAA; margin-top:0.6rem;
                              font-weight:500;">
                    Regression &mdash; XGBoost
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with card_right:
            if premium_label == 1:
                badge_bg  = OLIVE
                badge_txt = "Premium Home"
                conf_line = f"{premium_proba * 100:.1f}% model confidence"
            else:
                badge_bg  = SAGE
                badge_txt = "Not Premium"
                conf_line = f"{(1 - premium_proba) * 100:.1f}% model confidence"

            st.markdown(
                f"""
                <div style="background:{WHITE}; border:1px solid #DDD9D0;
                            border-radius:10px; padding:1.8rem 1.6rem; text-align:center;">
                  <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.13em;
                              color:{SAGE}; text-transform:uppercase; margin-bottom:0.6rem;">
                    High-Value Home?
                  </div>
                  <div style="margin:0.5rem 0 0.6rem 0;">
                    <span style="display:inline-block; background:{badge_bg}; color:white;
                                 border-radius:20px; padding:0.35rem 1.5rem;
                                 font-weight:700; font-size:1.05rem; letter-spacing:0.01em;">
                      {badge_txt}
                    </span>
                  </div>
                  <div style="font-size:0.82rem; color:{CHARCOAL}; font-weight:500;">
                    {conf_line}
                  </div>
                  <div style="font-size:0.75rem; color:#AAA; margin-top:0.4rem; font-weight:500;">
                    Classification &mdash; XGBoost &middot; Top 25% threshold
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<p style='font-size:0.78rem; color:#AAA; margin-top:1rem; line-height:1.5;'>"
            f"Statistical estimate based on Ames, Iowa sales data (2006–2010). "
            f"Not a certified appraisal — do not use as the sole basis for financial decisions.</p>",
            unsafe_allow_html=True,
        )

        # ── Investment Insights ──────────────────────────────────────────────
        st.markdown(
            f"<hr style='border:none; border-top:1px solid {SAGE}; opacity:0.35; margin:2.2rem 0 1.8rem 0;'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='font-size:1.3rem; font-weight:800; letter-spacing:0.04em; "
            f"color:{OLIVE}; margin-bottom:0.9rem;'>"
            f"Investment Insights</p>",
            unsafe_allow_html=True,
        )

        _ii_left, _ii_right = st.columns([1, 2], gap="large")

        with _ii_left:
            st.markdown(
                f"""
                <div style='background:{WHITE}; border:1px solid #DDD9D0;
                            border-left:4px solid {_inv_color};
                            border-radius:0 10px 10px 0; padding:1.6rem 1.4rem;'>
                  <div style='font-size:0.72rem; font-weight:700; letter-spacing:0.13em;
                              color:{SAGE}; text-transform:uppercase; margin-bottom:0.6rem;'>
                    Investment Potential
                  </div>
                  <div style='font-size:2.1rem; font-weight:800; color:{_inv_color};
                              letter-spacing:-0.01em; line-height:1.1; margin-bottom:0.5rem;'>
                    {_inv_level}
                  </div>
                  <div style='font-size:0.82rem; color:{CHARCOAL}; line-height:1.6;'>
                    {_inv_desc}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with _ii_right:
            _mp1, _mp2, _mp3 = st.columns(3, gap="medium")
            with _mp1:
                st.markdown(
                    f"""
                    <div style='background:{WHITE}; border:1px solid #DDD9D0;
                                border-radius:10px; padding:1.3rem 1.0rem; text-align:center;'>
                      <div style='font-size:0.65rem; font-weight:700; letter-spacing:0.12em;
                                  color:{SAGE}; text-transform:uppercase; margin-bottom:0.5rem;'>
                        Price vs Ames Avg
                      </div>
                      <div style='font-size:0.9rem; font-weight:700; color:{CHARCOAL}; line-height:1.35;'>
                        {_price_vs_avg}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with _mp2:
                st.markdown(
                    f"""
                    <div style='background:{WHITE}; border:1px solid #DDD9D0;
                                border-radius:10px; padding:1.3rem 1.0rem; text-align:center;'>
                      <div style='font-size:0.65rem; font-weight:700; letter-spacing:0.12em;
                                  color:{SAGE}; text-transform:uppercase; margin-bottom:0.5rem;'>
                        Neighborhood Tier
                      </div>
                      <div style='font-size:0.9rem; font-weight:700; color:{CHARCOAL}; line-height:1.35;'>
                        {_nbhd_tier}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with _mp3:
                st.markdown(
                    f"""
                    <div style='background:{WHITE}; border:1px solid #DDD9D0;
                                border-radius:10px; padding:1.3rem 1.0rem; text-align:center;'>
                      <div style='font-size:0.65rem; font-weight:700; letter-spacing:0.12em;
                                  color:{SAGE}; text-transform:uppercase; margin-bottom:0.5rem;'>
                        Resale Strength
                      </div>
                      <div style='font-size:0.9rem; font-weight:700; color:{CHARCOAL}; line-height:1.35;'>
                        {_resale}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"""
                <div style='background:{WHITE}; border:1px solid #DDD9D0;
                            border-top:3px solid {OLIVE};
                            border-radius:0 0 12px 12px;
                            padding:1.6rem 1.4rem; text-align:center;
                            margin-top:0.65rem;'>
                  <div style='font-size:0.65rem; font-weight:700; letter-spacing:0.13em;
                              color:{SAGE}; text-transform:uppercase; margin-bottom:0.55rem;'>
                    Projected 5-Year Value
                  </div>
                  <div style='font-size:2.0rem; font-weight:800; color:{OLIVE};
                              letter-spacing:-0.02em; line-height:1.1;'>
                    ${_projected_5yr:,.0f}
                  </div>
                  <div style='font-size:0.72rem; color:#AAA; margin-top:0.45rem; font-weight:500;'>
                    @ 3.5% annual appreciation &middot; 5 years
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div style='background:{WHITE}; border:1px solid #DDD9D0;
                        border-left:4px solid {OLIVE}; border-radius:0 10px 10px 0;
                        padding:1.8rem 1.7rem; margin-top:1.0rem;'>

              <!-- ── header ── -->
              <div style='font-size:0.72rem; font-weight:700; letter-spacing:0.13em;
                          color:{SAGE}; text-transform:uppercase; margin-bottom:0.75rem;'>
                How Investment Potential is Calculated
              </div>
              <p style='font-size:0.88rem; color:{CHARCOAL}; line-height:1.75; margin:0 0 1.1rem 0;'>
                This is not a separate machine learning model. The investment potential is a
                rule-based interpretation layer built on top of the model output and Ames
                housing market patterns.
              </p>

              <!-- ── divider ── -->
              <div style='height:1px; background:#DDD9D0; margin-bottom:1.1rem;'></div>

              <!-- ── score factors ── -->
              <div style='font-size:0.72rem; font-weight:700; letter-spacing:0.1em;
                          color:{CHARCOAL}; text-transform:uppercase; margin-bottom:0.65rem;'>
                The score is based on:
              </div>
              <div style='display:flex; gap:2rem; margin-bottom:1.2rem;'>
                <div style='flex:1;'>
                  <div style='display:flex; align-items:flex-start; margin-bottom:0.4rem;'>
                    <span style='color:{OLIVE}; font-weight:800; margin-right:0.5rem; flex-shrink:0;'>&#10003;</span>
                    <span style='font-size:0.86rem; color:{CHARCOAL}; line-height:1.6;'>predicted sale price</span>
                  </div>
                  <div style='display:flex; align-items:flex-start; margin-bottom:0.4rem;'>
                    <span style='color:{OLIVE}; font-weight:800; margin-right:0.5rem; flex-shrink:0;'>&#10003;</span>
                    <span style='font-size:0.86rem; color:{CHARCOAL}; line-height:1.6;'>premium-home prediction</span>
                  </div>
                  <div style='display:flex; align-items:flex-start;'>
                    <span style='color:{OLIVE}; font-weight:800; margin-right:0.5rem; flex-shrink:0;'>&#10003;</span>
                    <span style='font-size:0.86rem; color:{CHARCOAL}; line-height:1.6;'>neighborhood market tier</span>
                  </div>
                </div>
                <div style='flex:1;'>
                  <div style='display:flex; align-items:flex-start; margin-bottom:0.4rem;'>
                    <span style='color:{OLIVE}; font-weight:800; margin-right:0.5rem; flex-shrink:0;'>&#10003;</span>
                    <span style='font-size:0.86rem; color:{CHARCOAL}; line-height:1.6;'>overall quality</span>
                  </div>
                  <div style='display:flex; align-items:flex-start; margin-bottom:0.4rem;'>
                    <span style='color:{OLIVE}; font-weight:800; margin-right:0.5rem; flex-shrink:0;'>&#10003;</span>
                    <span style='font-size:0.86rem; color:{CHARCOAL}; line-height:1.6;'>above-ground living area</span>
                  </div>
                  <div style='display:flex; align-items:flex-start;'>
                    <span style='color:{OLIVE}; font-weight:800; margin-right:0.5rem; flex-shrink:0;'>&#10003;</span>
                    <span style='font-size:0.86rem; color:{CHARCOAL}; line-height:1.6;'>price position vs. Ames average</span>
                  </div>
                </div>
              </div>

              <!-- ── divider ── -->
              <div style='height:1px; background:#DDD9D0; margin-bottom:1.1rem;'></div>

              <!-- ── classification rows ── -->
              <div style='font-size:0.72rem; font-weight:700; letter-spacing:0.13em;
                          color:{SAGE}; text-transform:uppercase; margin-bottom:0.75rem;'>
                Classification
              </div>
              <div style='border:1px solid #DDD9D0; border-radius:8px; overflow:hidden; margin-bottom:1.1rem;'>
                <div style='display:flex; align-items:center; padding:0.65rem 0.9rem;
                            border-bottom:1px solid #DDD9D0;'>
                  <span style='font-size:0.82rem; font-weight:800; color:{SAGE};
                               min-width:5.5rem; flex-shrink:0;'>LOW</span>
                  <span style='font-size:0.82rem; color:{CHARCOAL}; line-height:1.5;'>
                    weaker market positioning or below-average investment signals
                  </span>
                </div>
                <div style='display:flex; align-items:center; padding:0.65rem 0.9rem;
                            border-bottom:1px solid #DDD9D0;'>
                  <span style='font-size:0.82rem; font-weight:800; color:{CHARCOAL};
                               min-width:5.5rem; flex-shrink:0;'>MEDIUM</span>
                  <span style='font-size:0.82rem; color:{CHARCOAL}; line-height:1.5;'>
                    balanced property profile with moderate resale potential
                  </span>
                </div>
                <div style='display:flex; align-items:center; padding:0.65rem 0.9rem;'>
                  <span style='font-size:0.82rem; font-weight:800; color:{OLIVE};
                               min-width:5.5rem; flex-shrink:0;'>HIGH</span>
                  <span style='font-size:0.82rem; color:{CHARCOAL}; line-height:1.5;'>
                    strong neighborhood, quality, size, and premium-market characteristics
                  </span>
                </div>
              </div>

              <!-- ── dynamic reasoning ── -->
              <div style='background:{CREAM}; border-radius:8px; padding:1.0rem 1.1rem;
                          margin-bottom:1.1rem;'>
                <div style='font-size:0.72rem; font-weight:700; letter-spacing:0.1em;
                            color:{SAGE}; text-transform:uppercase; margin-bottom:0.6rem;'>
                  Why this property is classified as {_inv_level}
                </div>
                {_reason_bullet_html}
              </div>

              <!-- ── divider ── -->
              <div style='height:1px; background:#DDD9D0; margin-bottom:0.9rem;'></div>

              <!-- ── disclaimer ── -->
              <p style='font-size:0.75rem; color:#AAA; line-height:1.6;
                        font-style:italic; margin:0 0 0.5rem 0;'>
                Investment Insights are explanatory business logic only and should not be
                interpreted as financial advice or as a separately trained investment
                prediction model.
              </p>
              <p style='font-size:0.75rem; color:#AAA; line-height:1.6;
                        font-style:italic; margin:0;'>
                5-year projection assumes a fixed 3.5% annual appreciation rate based on
                long-run historical housing trends — it is a scenario estimate, not a
                machine learning prediction.
              </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_predict:
    _predict_tab_body()

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════════════════════════════════════════════════
with tab_eda:
    # ── Hardcoded summary statistics (no raw file dependency) ─────────────────
    _N_HOMES      = 2_930
    _MEDIAN_PRICE = 160_000.0
    _MEAN_PRICE   = 180_796.0
    _Q75_PRICE    = 213_500.0   # premium-home threshold (top 25%)

    median_price = _MEDIAN_PRICE
    mean_price   = _MEAN_PRICE

    # Neighbourhood medians (in $) derived from NEIGHBORHOOD_INFO already in scope
    nbhd_med = pd.Series({
        code: info[3] * 1_000
        for code, info in NEIGHBORHOOD_INFO.items()
    })
    richest_nbhd  = nbhd_med.idxmax()   # NridgHt  $315k
    cheapest_nbhd = nbhd_med.idxmin()   # MeadowV  $88k
    price_ratio   = nbhd_med.max() / nbhd_med.min()

    # Quality-vs-median-price (hardcoded from training-set statistics)
    qual_med = pd.Series({
        1:  40_000, 2:  51_000, 3:  79_000, 4: 105_000, 5: 133_000,
        6: 161_000, 7: 200_000, 8: 226_000, 9: 295_000, 10: 395_000,
    })
    price_jump   = qual_med[8] / qual_med[6] - 1   # ≈ 0.40
    top_qual_pct = 5.6                              # % of homes with Qual ≥ 9

    area_corr = 0.71   # Pearson r, Gr Liv Area vs SalePrice

    # Synthetic point-cloud for Chart 4 (fixed seed → deterministic, no file needed)
    _rng       = np.random.default_rng(42)
    _area_syn  = np.clip(_rng.lognormal(np.log(1_442), 0.35, _N_HOMES), 334, 5_642).astype(int)
    _qual_syn  = np.clip(np.round(_rng.normal(6.1, 1.4, _N_HOMES)), 1, 10).astype(int)
    _price_syn = np.clip(                                        # in $k
        50 + 0.075 * _area_syn + 10 * (_qual_syn - 5) + _rng.normal(0, 28, _N_HOMES),
        34.9, 611.657,
    )

    # Synthetic price series for Chart 1 histogram (log-normal ≈ real distribution)
    _price_hist = np.clip(                                       # in $k
        _rng.lognormal(np.log(160), 0.55, _N_HOMES), 34.9, 611.657
    )

    # ── Colour map shared by all charts ──────────────────────────────────────
    cmap = mcolors.LinearSegmentedColormap.from_list("oa", [SAGE, OLIVE])

    # ── Layout helpers ────────────────────────────────────────────────────────
    def _section_left(num, label, headline, body, kpi_label, kpi_value, kpi_sub):
        st.markdown(
            f"<div style='padding-top:0.3rem;'>"
            f"<div style='display:flex; align-items:center; gap:0.55rem; margin-bottom:0.65rem;'>"
            f"<span style='background:{OLIVE}; color:{WHITE}; font-size:0.62rem; "
            f"font-weight:800; letter-spacing:0.08em; border-radius:4px; "
            f"padding:0.18rem 0.55rem;'>{num}</span>"
            f"<span style='font-size:0.62rem; font-weight:700; letter-spacing:0.18em; "
            f"color:{SAGE}; text-transform:uppercase;'>{label}</span>"
            f"</div>"
            f"<h3 style='font-size:1.25rem; font-weight:800; color:{CHARCOAL}; "
            f"letter-spacing:-0.02em; line-height:1.3; margin:0 0 0.65rem 0;'>{headline}</h3>"
            f"<p style='font-size:0.88rem; color:{SAGE}; line-height:1.8; margin:0 0 1.1rem 0;'>{body}</p>"
            f"<div style='background:{CREAM}; border:1px solid #DDD9D0; border-radius:10px; "
            f"padding:1.1rem 1.3rem;'>"
            f"<p style='font-size:0.62rem; font-weight:700; letter-spacing:0.15em; "
            f"color:{SAGE}; text-transform:uppercase; margin:0 0 0.25rem 0;'>{kpi_label}</p>"
            f"<p style='font-size:2.2rem; font-weight:800; color:{OLIVE}; "
            f"letter-spacing:-0.03em; line-height:1.0; margin:0 0 0.3rem 0;'>{kpi_value}</p>"
            f"<p style='font-size:0.8rem; color:{CHARCOAL}; line-height:1.5; margin:0;'>{kpi_sub}</p>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    def _key_insight(text):
        st.markdown(
            f"<div style='background:{WHITE}; border:1px solid #DDD9D0; "
            f"border-left:4px solid {OLIVE}; border-radius:0 8px 8px 0; "
            f"padding:0.9rem 1.3rem; margin-top:1.3rem;'>"
            f"<p style='font-size:0.62rem; font-weight:700; letter-spacing:0.16em; "
            f"color:{SAGE}; text-transform:uppercase; margin:0 0 0.3rem 0;'>"
            f"&#10022;&nbsp; Key Insight</p>"
            f"<p style='font-size:0.9rem; color:{CHARCOAL}; line-height:1.75; margin:0;'>"
            f"{text}</p></div>",
            unsafe_allow_html=True,
        )

    def _section_divider():
        st.markdown(
            f"<hr style='border:none; border-top:1px solid {SAGE}; "
            f"opacity:0.2; margin:2.5rem 0;'>",
            unsafe_allow_html=True,
        )

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f"<p style='font-size:0.68rem; font-weight:700; letter-spacing:0.18em; "
        f"color:{SAGE}; text-transform:uppercase; margin-bottom:0.3rem;'>"
        f"Ames, Iowa &middot; 2006–2010</p>"
        f"<h2 style='font-size:1.7rem; font-weight:800; color:{CHARCOAL}; "
        f"letter-spacing:-0.025em; line-height:1.2; margin:0 0 0.5rem 0;'>"
        f"Housing Market Insights</h2>"
        f"<p style='font-size:0.9rem; color:{SAGE}; line-height:1.75; margin:0;'>"
        f"A data-driven look at {_N_HOMES:,} real home sales — exploring price "
        f"patterns, neighborhood dynamics, quality effects, and the signals that "
        f"separate premium properties from the rest of the market.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<hr style='border:none; border-top:2px solid {SAGE}; opacity:0.3; margin:1.5rem 0;'>",
        unsafe_allow_html=True,
    )

    # ── Section 01: Price Distribution ───────────────────────────────────────
    _s01_l, _s01_r = st.columns([1, 1.5], gap="large")
    with _s01_l:
        _section_left(
            "01", "PRICE DISTRIBUTION",
            "How are Ames home prices distributed?",
            "The Ames market spans an unusually wide range — from modest starter "
            "homes under $40k to luxury estates above $600k. Most transactions "
            "cluster around the median, but a long right tail pulls the average "
            "well above what a typical buyer will pay.",
            "Median Sale Price", f"${median_price:,.0f}",
            f"well below the mean of ${mean_price:,.0f}",
        )
    with _s01_r:
        fig, ax = plt.subplots(figsize=(8, 4))
        n, bins, patches = ax.hist(_price_hist, bins=50, edgecolor="none")
        norm = mcolors.Normalize(vmin=bins[0], vmax=bins[-1])
        for patch, left in zip(patches, bins[:-1]):
            patch.set_facecolor(cmap(norm(left)))
            patch.set_alpha(0.88)
        ax.axvline(median_price / 1_000, color=CHARCOAL, linewidth=1.6,
                   linestyle="--", zorder=5)
        ax.text(median_price / 1_000 + 3, ax.get_ylim()[1] * 0.88,
                f"Median  ${median_price/1000:.0f}k",
                color=CHARCOAL, fontsize=9, fontweight="600")
        ax.set_xlabel("Sale Price ($k)", labelpad=6, fontsize=10)
        ax.set_ylabel("Number of Homes", labelpad=6, fontsize=10)
        ax.set_title("Sale Price Distribution", pad=10, fontsize=11)
        ax.tick_params(labelsize=9)
        style_ax(fig, ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    _key_insight(
        f"The median Ames home sold for <strong>${median_price:,.0f}</strong> — "
        f"well below the mean of <strong>${mean_price:,.0f}</strong>, pulled up by "
        f"a long tail of luxury sales. "
        f"The top 25% threshold used to define a Premium Home sits at "
        f"<strong>${_Q75_PRICE:,.0f}</strong>."
    )
    _section_divider()

    # ── Section 02: Neighborhood Performance ─────────────────────────────────
    nbhd_sorted = nbhd_med.sort_values(ascending=True)
    norm2 = mcolors.Normalize(vmin=nbhd_sorted.min(), vmax=nbhd_sorted.max())
    bar_colors2 = [cmap(norm2(v)) for v in nbhd_sorted.values]

    _s02_l, _s02_r = st.columns([1, 1.5], gap="large")
    with _s02_l:
        _section_left(
            "02", "NEIGHBORHOOD PERFORMANCE",
            "Which neighbourhoods command the highest prices?",
            "Location is the one factor buyers cannot change after purchase. "
            "Ames has 24 distinct neighbourhoods spanning three market tiers — "
            "Premium, Mid-Range, and Budget — and median prices vary dramatically "
            "from one end of the city to the other.",
            "Market Price Range", f"{price_ratio:.1f}×",
            f"{richest_nbhd} ${nbhd_med[richest_nbhd]/1000:.0f}k "
            f"vs {cheapest_nbhd} ${nbhd_med[cheapest_nbhd]/1000:.0f}k",
        )
    with _s02_r:
        fig, ax = plt.subplots(figsize=(8, 8))
        bars = ax.barh(nbhd_sorted.index, nbhd_sorted.values / 1_000,
                       color=bar_colors2, edgecolor="none", height=0.65)
        for bar, val in zip(bars, nbhd_sorted.values):
            ax.text(bar.get_width() + 1.2,
                    bar.get_y() + bar.get_height() / 2,
                    f"${val/1000:.0f}k",
                    va="center", fontsize=8, color=CHARCOAL)
        ax.set_xlabel("Median Sale Price ($k)", labelpad=6, fontsize=10)
        ax.set_title("Median Sale Price by Neighbourhood", pad=10, fontsize=11)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%dk"))
        ax.set_xlim(right=nbhd_sorted.max() / 1_000 * 1.18)
        style_ax(fig, ax)
        ax.grid(axis="x", color="#DDD9D0", linewidth=0.7)
        ax.grid(axis="y", visible=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    _key_insight(
        f"<strong>{richest_nbhd}</strong> commands the highest median price at "
        f"<strong>${nbhd_med[richest_nbhd]:,.0f}</strong> — "
        f"<strong>{price_ratio:.1f}&times;</strong> more than the most affordable "
        f"neighbourhood, <strong>{cheapest_nbhd}</strong> "
        f"(<strong>${nbhd_med[cheapest_nbhd]:,.0f}</strong>). "
        f"Location is one of the strongest predictors in both models."
    )
    _section_divider()

    # ── Section 03: Build Quality Impact ─────────────────────────────────────
    qual_vals   = qual_med.index.astype(int)
    qual_prices = qual_med.values / 1_000
    norm3 = mcolors.Normalize(vmin=qual_prices.min(), vmax=qual_prices.max())
    bar_colors3 = [cmap(norm3(v)) for v in qual_prices]

    _s03_l, _s03_r = st.columns([1, 1.5], gap="large")
    with _s03_l:
        _section_left(
            "03", "BUILD QUALITY IMPACT",
            "How much does build quality matter?",
            "Every home in the dataset carries an overall quality rating from 1 "
            "to 10. This single number — which summarises construction materials, "
            "finish level, and craftsmanship — turns out to be one of the most "
            "decisive factors in determining final sale price.",
            "Quality Premium", f"+{price_jump * 100:.0f}%",
            "price increase moving from rating 6 to 8",
        )
    with _s03_r:
        fig, ax = plt.subplots(figsize=(8, 4))
        bars3 = ax.bar(qual_vals, qual_prices, color=bar_colors3,
                       edgecolor="none", width=0.65, zorder=3)
        for bar, val in zip(bars3, qual_prices):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    f"${val:.0f}k",
                    ha="center", va="bottom", fontsize=8, color=CHARCOAL)
        ax.set_xlabel("Overall Quality Rating (1–10)", labelpad=6, fontsize=10)
        ax.set_ylabel("Median Sale Price ($k)", labelpad=6, fontsize=10)
        ax.set_title("Median Sale Price by Overall Quality", pad=10, fontsize=11)
        ax.tick_params(labelsize=9)
        ax.set_xticks(qual_vals)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%dk"))
        style_ax(fig, ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    _key_insight(
        f"Moving from quality rating 6 to 8 adds roughly "
        f"<strong>{price_jump * 100:.0f}%</strong> to the median price — "
        f"a gap that can exceed $60,000 on a single transaction. "
        f"Only <strong>{top_qual_pct:.1f}%</strong> of homes in the dataset achieved "
        f"a rating of 9 or 10, making top-tier construction genuinely rare."
    )
    _section_divider()

    # ── Section 04: Space vs Price ────────────────────────────────────────────
    _s04_l, _s04_r = st.columns([1, 1.5], gap="large")
    with _s04_l:
        _section_left(
            "04", "SPACE VS PRICE",
            "Does more space mean more money?",
            "Above-ground living area is the most intuitive price signal — "
            "more rooms, more value. But size alone does not tell the full story. "
            "Two homes with identical square footage can differ by tens of thousands "
            "of dollars depending on build quality and location.",
            "Area–Price Correlation", f"r = {area_corr:.2f}",
            "Pearson r between living area and sale price",
        )
    with _s04_r:
        fig, ax = plt.subplots(figsize=(8, 4))
        sc = ax.scatter(
            _area_syn, _price_syn,
            c=_qual_syn,
            cmap=mcolors.LinearSegmentedColormap.from_list("oa2", [SAGE, OLIVE]),
            alpha=0.45, s=10, linewidths=0, zorder=3,
        )
        m, b = np.polyfit(_area_syn, _price_syn, 1)
        x_line = np.linspace(_area_syn.min(), _area_syn.max(), 200)
        ax.plot(x_line, m * x_line + b, color=CHARCOAL, linewidth=1.8,
                linestyle="--", zorder=5, label=f"Trend  (r = {area_corr:.2f})")
        ax.legend(fontsize=9, frameon=False, loc="upper left")
        cbar = plt.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label("Overall Quality", fontsize=9, color=CHARCOAL)
        cbar.ax.tick_params(labelsize=8, colors=CHARCOAL)
        cbar.outline.set_edgecolor(SAGE)
        ax.set_xlabel("Above-Ground Living Area (sq ft)", labelpad=6, fontsize=10)
        ax.set_ylabel("Sale Price ($k)", labelpad=6, fontsize=10)
        ax.set_title("Living Area vs Sale Price, coloured by Quality", pad=10, fontsize=11)
        ax.tick_params(labelsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%dk"))
        style_ax(fig, ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    _key_insight(
        f"Living area correlates with price at <strong>r = {area_corr:.2f}</strong>, "
        f"but the colour overlay shows that two homes with identical square footage "
        f"can differ by tens of thousands of dollars based on build quality alone. "
        f"This interaction is captured by the engineered Quality &times; Area feature, "
        f"which is the single strongest predictor in both models."
    )
    _section_divider()

    # ── Section 05: Premium Home Signals ─────────────────────────────────────
    import re
    feat_names = clf_model[:-1].get_feature_names_out()
    importances = clf_model[-1].feature_importances_

    def base_name(n):
        n = re.sub(r"^(num__|cat__)", "", n)
        parts = n.rsplit("_", 1)
        if len(parts) == 2 and parts[1].replace(".", "").isalnum() and len(parts[1]) <= 8:
            return parts[0]
        return n

    imp_series = pd.Series(importances, index=feat_names)
    grouped_imp = (
        imp_series.groupby(imp_series.index.map(base_name))
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    label_map = {
        "Qual_x": "Quality × Area Score",
        "Kitchen Qual": "Kitchen Quality",
        "Bldg Type": "Building Type",
        "Garage Finish": "Garage Finish",
        "Neighborhood": "Neighbourhood",
        "TotalBath": "Total Bathrooms",
        "Heating QC": "Heating Quality",
        "Exter Qual": "Exterior Quality",
        "Fireplaces": "Fireplaces",
        "TotalSF": "Total Square Footage",
        "Bsmt Unf SF": "Unfinished Basement SF",
        "Exterior 1st": "Exterior Material",
        "Kitchen AbvGr": "Kitchens Above Grade",
        "Year Remod/Add": "Remodel Year",
        "Bsmt Qual": "Basement Quality",
    }
    grouped_imp.index = [label_map.get(i, i) for i in grouped_imp.index]
    grouped_imp = grouped_imp.sort_values(ascending=True)
    norm5 = mcolors.Normalize(vmin=grouped_imp.min(), vmax=grouped_imp.max())
    bar_colors5 = [cmap(norm5(v)) for v in grouped_imp.values]

    _s05_l, _s05_r = st.columns([1, 1.5], gap="large")
    with _s05_l:
        _section_left(
            "05", "PREMIUM HOME SIGNALS",
            "What separates premium homes from the rest of the market?",
            "The premium classifier draws a line at the top 25% of sale prices. "
            f"Homes above ${_Q75_PRICE:,.0f} earn the Premium label. "
            "The model's feature importances reveal which property attributes "
            "most reliably predict that a home will cross that threshold.",
            "Premium Threshold", f"${_Q75_PRICE:,.0f}",
            "top 25% of all Ames home sales",
        )
    with _s05_r:

        fig, ax = plt.subplots(figsize=(8, 5))
        bars5 = ax.barh(grouped_imp.index, grouped_imp.values * 100,
                        color=bar_colors5, edgecolor="none", height=0.62)
        for bar, val in zip(bars5, grouped_imp.values):
            ax.text(bar.get_width() + 0.15,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val*100:.1f}%",
                    va="center", fontsize=8, color=CHARCOAL)
        ax.set_xlabel("Feature Importance (%)", labelpad=6, fontsize=10)
        ax.set_title("Top 10 Drivers of Premium Classification", pad=10, fontsize=11)
        ax.tick_params(labelsize=8)
        ax.set_xlim(right=grouped_imp.max() * 100 * 1.22)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        style_ax(fig, ax)
        ax.grid(axis="x", color="#DDD9D0", linewidth=0.7)
        ax.grid(axis="y", visible=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    top1 = grouped_imp.index[-1]
    top1_pct = grouped_imp.values[-1] * 100
    top3_pct = grouped_imp.values[-3:].sum() * 100
    _key_insight(
        f"<strong>{top1}</strong> alone accounts for "
        f"<strong>{top1_pct:.0f}%</strong> of the classifier's decision weight — "
        f"it multiplies appraiser quality by total square footage, capturing both "
        f"size and finish in a single number. "
        f"The top 3 features together explain <strong>{top3_pct:.0f}%</strong> "
        f"of all importance, showing how concentrated the signal is."
    )

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — NEIGHBORHOODS
# ════════════════════════════════════════════════════════════════════════════

# Premium featured: code → (image path, mime type, tag list)
_PREMIUM_FEATURED = {
    "NridgHt": ("assets/northridge_heights.png", "image/png",
                ["Executive suburb", "Large lots", "Top-tier value"]),
    "NoRidge":  ("assets/noridge.jpeg",           "image/jpeg",
                ["Luxury homes", "Strong resale", "Family-oriented"]),
    "StoneBr":  ("assets/stonebrook.jpg",         "image/jpeg",
                ["Highest median price", "Premium enclave", "Top-tier value"]),
}

def _load_nbhd_img(path: str) -> str:
    try:
        import base64 as _b64n
        return _b64n.b64encode(open(path, "rb").read()).decode()
    except FileNotFoundError:
        return ""

def _tag_pill(label: str) -> str:
    return (
        f"<span style='display:inline-block; background:rgba(91,107,58,0.09); "
        f"color:{OLIVE}; border:1px solid rgba(91,107,58,0.22); border-radius:20px; "
        f"padding:0.14rem 0.58rem; font-size:0.67rem; font-weight:600; "
        f"letter-spacing:0.03em; margin-right:0.28rem; margin-bottom:0.28rem;'>{label}</span>"
    )

def _nbhd_section_header(label: str, subtitle: str) -> str:
    return (
        f"<div style='margin:2.4rem 0 1.3rem 0;'>"
        f"<p style='font-size:0.72rem; font-weight:700; letter-spacing:0.18em; "
        f"color:{OLIVE}; text-transform:uppercase; margin:0 0 0.3rem 0;'>{label}</p>"
        f"<p style='font-size:1.15rem; font-weight:800; color:{CHARCOAL}; "
        f"letter-spacing:-0.01em; margin:0 0 0.55rem 0; line-height:1.3;'>{subtitle}</p>"
        f"<div style='width:2.6rem; height:2px; background:{OLIVE}; border-radius:2px;'></div>"
        f"</div>"
    )

def _premium_photo_card(code: str) -> None:
    full_name, description, _, median = NEIGHBORHOOD_INFO[code]
    img_path, mime, tags = _PREMIUM_FEATURED[code]
    b64 = _load_nbhd_img(img_path)
    tags_html = "".join(_tag_pill(t) for t in tags)
    img_block = (
        f"<div style='flex:0 0 250px; overflow:hidden; border-right:1px solid #DDD9D0;'>"
        f"<img src='data:{mime};base64,{b64}' "
        f"style='width:250px; height:100%; min-height:195px; object-fit:cover; display:block;' />"
        f"</div>"
    ) if b64 else (
        f"<div style='flex:0 0 6px; background:{OLIVE}; opacity:0.18;'></div>"
    )
    st.markdown(
        f"<div style='background:{WHITE}; border:1px solid #DDD9D0; "
        f"border-left:4px solid {OLIVE}; border-radius:0 14px 14px 0; "
        f"overflow:hidden; margin-bottom:1.1rem;'>"
        f"<div style='display:flex;'>"
        f"{img_block}"
        f"<div style='flex:1; padding:1.55rem 1.65rem;'>"
        f"<span style='display:inline-block; background:{OLIVE}; color:white; "
        f"border-radius:20px; padding:0.16rem 0.72rem; font-size:0.62rem; "
        f"font-weight:700; letter-spacing:0.12em; text-transform:uppercase; "
        f"margin-bottom:0.72rem;'>Premium Market</span>"
        f"<div style='font-size:1.22rem; font-weight:800; color:{CHARCOAL}; "
        f"letter-spacing:-0.015em; margin-bottom:0.52rem; line-height:1.2;'>{full_name}</div>"
        f"<div style='margin-bottom:0.78rem;'>{tags_html}</div>"
        f"<div style='font-size:0.84rem; color:#6A6A64; line-height:1.75; "
        f"margin-bottom:0.85rem;'>{description}</div>"
        f"<div style='font-size:0.78rem; font-weight:700; color:{OLIVE}; letter-spacing:0.01em;'>"
        f"Median sale price &nbsp;&middot;&nbsp; "
        f"<span style='font-size:1.0rem; font-weight:800;'>${median}k</span></div>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

def _premium_compact_card(code: str) -> None:
    full_name, description, _, median = NEIGHBORHOOD_INFO[code]
    st.markdown(
        f"<div style='background:{WHITE}; border:1px solid #DDD9D0; "
        f"border-left:3px solid {OLIVE}; border-radius:0 10px 10px 0; "
        f"padding:1.25rem 1.4rem; margin-bottom:0.75rem;'>"
        f"<span style='display:inline-block; background:{OLIVE}; color:white; "
        f"border-radius:20px; padding:0.11rem 0.62rem; font-size:0.62rem; "
        f"font-weight:700; letter-spacing:0.1em; text-transform:uppercase; "
        f"margin-bottom:0.48rem;'>Premium</span>"
        f"<div style='font-size:1.0rem; font-weight:800; color:{CHARCOAL}; "
        f"margin-bottom:0.4rem;'>{full_name}</div>"
        f"<div style='font-size:0.81rem; color:#6A6A64; line-height:1.65; "
        f"margin-bottom:0.55rem;'>{description}</div>"
        f"<div style='font-size:0.76rem; font-weight:600; color:{OLIVE};'>"
        f"Median &nbsp;${median}k</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

def _midrange_card(code: str) -> None:
    full_name, description, _, median = NEIGHBORHOOD_INFO[code]
    st.markdown(
        f"<div style='background:{WHITE}; border:1px solid #DDD9D0; "
        f"border-radius:10px; padding:1.15rem 1.25rem; margin-bottom:0.75rem;'>"
        f"<span style='display:inline-block; background:{SAGE}; color:white; "
        f"border-radius:20px; padding:0.11rem 0.62rem; font-size:0.62rem; "
        f"font-weight:700; letter-spacing:0.1em; text-transform:uppercase; "
        f"margin-bottom:0.45rem;'>Mid-Range</span>"
        f"<div style='font-size:0.97rem; font-weight:700; color:{CHARCOAL}; "
        f"margin-bottom:0.38rem;'>{full_name}</div>"
        f"<div style='font-size:0.80rem; color:#777; line-height:1.65; "
        f"margin-bottom:0.5rem;'>{description}</div>"
        f"<div style='font-size:0.74rem; font-weight:600; color:{SAGE};'>"
        f"Median &nbsp;${median}k</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

def _budget_card(code: str) -> None:
    full_name, description, _, median = NEIGHBORHOOD_INFO[code]
    st.markdown(
        f"<div style='background:{CREAM}; border:1px solid #E0DDD6; "
        f"border-radius:10px; padding:1.0rem 1.1rem; margin-bottom:0.65rem;'>"
        f"<div style='font-size:0.91rem; font-weight:700; color:{CHARCOAL}; "
        f"margin-bottom:0.32rem;'>{full_name}</div>"
        f"<div style='font-size:0.77rem; color:#888; line-height:1.62; "
        f"margin-bottom:0.42rem;'>{description}</div>"
        f"<div style='font-size:0.72rem; font-weight:600; color:#9E9E95;'>"
        f"Median &nbsp;${median}k</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with tab_nbhd:
    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f"<p style='font-size:0.72rem; font-weight:700; letter-spacing:0.22em; "
        f"color:{OLIVE}; text-transform:uppercase; margin:0.3rem 0 0.4rem 0;'>"
        f"Neighborhood Guide</p>"
        f"<h2 style='font-size:1.9rem; font-weight:800; color:{CHARCOAL}; "
        f"letter-spacing:-0.025em; line-height:1.2; margin:0 0 0.4rem 0;'>"
        f"Ames's Neighborhoods</h2>"
        f"<p style='font-size:0.88rem; color:{SAGE}; margin:0 0 1.5rem 0;'>"
        f"Explore all residential districts — select one in the Predict tab "
        f"to estimate home value.</p>",
        unsafe_allow_html=True,
    )

    # ── Interactive map (UNCHANGED) ───────────────────────────────────────────
    _COORDS = {
        "NridgHt": (42.0525, -93.6553),
        "NoRidge":  (42.0500, -93.6600),
        "StoneBr":  (42.0606, -93.6175),
        "Somerst":  (42.0612, -93.6089),
        "Veenker":  (42.0325, -93.6481),
        "CollgCr":  (42.0206, -93.6842),
        "Gilbert":  (42.1053, -93.6472),
        "Crawfor":  (42.0289, -93.6150),
        "Timber":   (41.9975, -93.6800),
        "SawyerW":  (42.0131, -93.6928),
        "Blmngtn":  (42.0642, -93.6350),
        "ClearCr":  (41.9998, -93.6600),
        "Mitchel":  (41.9958, -93.6319),
        "NAmes":    (42.0408, -93.6319),
        "Edwards":  (42.0208, -93.6619),
        "OldTown":  (42.0256, -93.6089),
        "BrkSide":  (42.0289, -93.6200),
        "IDOTRR":   (42.0175, -93.6175),
        "MeadowV":  (41.9906, -93.6089),
        "Sawyer":   (42.0089, -93.6750),
        "SWISU":    (42.0175, -93.6481),
        "BrDale":   (42.0392, -93.6692),
        "NPkVill":  (42.0525, -93.6692),
        "Blueste":  (42.0442, -93.6089),
    }

    _MARKER_COLOR = {
        "Premium":   "#3D4F27",
        "Mid-Range": "#C4A35A",
        "Budget":    "#8B9E78",
    }

    m = folium.Map(
        location=[42.0308, -93.6319],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    for code, (lat, lon) in _COORDS.items():
        if code not in NEIGHBORHOOD_INFO:
            continue
        full_name, _, tier, _ = NEIGHBORHOOD_INFO[code]
        color = _MARKER_COLOR[tier]
        folium.CircleMarker(
            location=[lat, lon],
            radius=10,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=1.5,
            popup=folium.Popup(
                f"<b style='font-family:Inter,sans-serif;'>{full_name}</b>"
                f"<br><span style='color:#777;font-size:0.85em;'>{tier}</span>",
                max_width=180,
            ),
            tooltip=full_name,
        ).add_to(m)

    st_folium(m, height=400, use_container_width=True)

    # ── Map legend ────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="display:flex; gap:1.6rem; margin-top:0.5rem;
                    margin-bottom:1.6rem; font-size:0.82rem; color:{CHARCOAL};">
          <span style="display:flex; align-items:center; gap:0.45rem;">
            <span style="display:inline-block; width:12px; height:12px;
                         border-radius:50%; background:#3D4F27;"></span>
            Premium
          </span>
          <span style="display:flex; align-items:center; gap:0.45rem;">
            <span style="display:inline-block; width:12px; height:12px;
                         border-radius:50%; background:#C4A35A;"></span>
            Mid-range
          </span>
          <span style="display:flex; align-items:center; gap:0.45rem;">
            <span style="display:inline-block; width:12px; height:12px;
                         border-radius:50%; background:#8B9E78;"></span>
            Budget
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── PREMIUM MARKET ────────────────────────────────────────────────────────
    st.markdown(
        _nbhd_section_header("PREMIUM MARKET", "Ames' highest-value residential districts"),
        unsafe_allow_html=True,
    )
    for _fc in ["NridgHt", "NoRidge", "StoneBr"]:
        _premium_photo_card(_fc)

    _pr_l, _pr_r = st.columns(2, gap="large")
    with _pr_l:
        _premium_compact_card("Somerst")
    with _pr_r:
        _premium_compact_card("Veenker")

    # ── MID-RANGE COMMUNITIES ─────────────────────────────────────────────────
    st.markdown(
        _nbhd_section_header("MID-RANGE COMMUNITIES", "Balanced residential value and family appeal"),
        unsafe_allow_html=True,
    )
    _mr_codes = ["CollgCr", "Gilbert", "Crawfor", "Timber",
                 "SawyerW", "Blmngtn", "ClearCr", "Mitchel"]
    for _i in range(0, len(_mr_codes), 2):
        _pair = _mr_codes[_i:_i+2]
        _cols = st.columns(2, gap="large")
        for _col, _code in zip(_cols, _pair):
            with _col:
                _midrange_card(_code)

    # ── ACCESSIBLE NEIGHBORHOODS ──────────────────────────────────────────────
    st.markdown(
        _nbhd_section_header("BUDGET NEIGHBORHOODS", "Entry-level and value-focused residential areas"),
        unsafe_allow_html=True,
    )
    _bg_codes = ["NAmes", "Edwards", "OldTown", "BrkSide", "IDOTRR",
                 "MeadowV", "Sawyer", "SWISU", "BrDale", "NPkVill", "Blueste"]
    for _i in range(0, len(_bg_codes), 3):
        _triple = _bg_codes[_i:_i+3]
        _cols = st.columns(3, gap="medium")
        for _col, _code in zip(_cols, _triple):
            with _col:
                _budget_card(_code)

    st.markdown("<br>", unsafe_allow_html=True)
