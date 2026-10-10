"""Theme and styling engine for the Dark Intelligence design system."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# Core Palette Constants
COLOR_BG_PRIMARY = "#061321"      # Midnight navy
COLOR_BG_SECONDARY = "#091B2C"    # Deep blue-black
COLOR_CARD = "#0B2033"            # Dark navy card
COLOR_CARD_HOVER = "#0E283F"      # Card hover
COLOR_ACCENT_TEAL = "#19D9C5"     # Primary vibrant teal
COLOR_ACCENT_CYAN = "#12BDF5"     # Secondary bright cyan
COLOR_ACCENT_PURPLE = "#8B5CF6"   # Purple
COLOR_ACCENT_BLUE = "#3B82F6"     # Royal blue
COLOR_ACCENT_MAGENTA = "#EC4899"  # Magenta/pink
COLOR_ACCENT_GREEN = "#10B981"    # Emerald green
COLOR_ACCENT_ORANGE = "#F59E0B"   # Amber/orange
COLOR_BORDER = "#16324D"          # Border navy
COLOR_BORDER_GLOW = "rgba(25, 217, 197, 0.25)"
COLOR_TEXT_PRIMARY = "#F8FAFC"    # Crisp off-white
COLOR_TEXT_MUTED = "#94A3B8"      # Muted slate

CHART_COLORWAY = [
    COLOR_ACCENT_CYAN,
    COLOR_ACCENT_TEAL,
    COLOR_ACCENT_PURPLE,
    COLOR_ACCENT_MAGENTA,
    COLOR_ACCENT_ORANGE,
    COLOR_ACCENT_GREEN,
    COLOR_ACCENT_BLUE,
]


def inject_theme_css() -> None:
    """Inject the complete Dark Intelligence CSS design system into Streamlit."""
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Root Resets */
    :root {{
        --bg-primary: {COLOR_BG_PRIMARY};
        --bg-secondary: {COLOR_BG_SECONDARY};
        --bg-card: {COLOR_CARD};
        --accent-teal: {COLOR_ACCENT_TEAL};
        --accent-cyan: {COLOR_ACCENT_CYAN};
        --border-color: {COLOR_BORDER};
        --text-primary: {COLOR_TEXT_PRIMARY};
        --text-muted: {COLOR_TEXT_MUTED};
    }}

    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {COLOR_BG_PRIMARY} !important;
        color: {COLOR_TEXT_PRIMARY} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}

    /* Main Container & Layout */
    .main .block-container {{
        max-width: 1440px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }}

    /* Headings */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Outfit', sans-serif !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {COLOR_BG_SECONDARY} !important;
        border-right: 1px solid {COLOR_BORDER} !important;
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        font-family: 'Inter', sans-serif !important;
        color: {COLOR_TEXT_MUTED};
    }}

    /* Sidebar Radio Buttons (Navigation) */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {{
        display: none !important;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {{
        gap: 0.35rem !important;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] {{
        background: transparent !important;
        padding: 0.55rem 0.85rem !important;
        border-radius: 10px !important;
        margin: 0 !important;
        border: 1px solid transparent !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover {{
        background: rgba(18, 189, 245, 0.08) !important;
        border-color: rgba(25, 217, 197, 0.2) !important;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] input:checked + div {{
        background: transparent !important;
    }}

    /* Active Radio Item */
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {{
        background: linear-gradient(90deg, rgba(25, 217, 197, 0.18) 0%, rgba(18, 189, 245, 0.08) 100%) !important;
        border: 1px solid rgba(25, 217, 197, 0.4) !important;
        box-shadow: 0 0 15px rgba(25, 217, 197, 0.15) !important;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] span {{
        color: #E2E8F0 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) span {{
        color: {COLOR_ACCENT_TEAL} !important;
        font-weight: 600 !important;
    }}

    /* Custom Top Navigation / Brand */
    .top-header-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: {COLOR_BG_SECONDARY};
        padding: 0.75rem 1.25rem;
        border-radius: 12px;
        border: 1px solid {COLOR_BORDER};
        margin-bottom: 1.5rem;
    }}

    .top-search-pill {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: {COLOR_BG_PRIMARY};
        padding: 0.45rem 1rem;
        border-radius: 8px;
        border: 1px solid {COLOR_BORDER};
        color: {COLOR_TEXT_MUTED};
        font-size: 0.85rem;
        min-width: 320px;
    }}

    .top-badge-pill {{
        background: rgba(25, 217, 197, 0.1);
        border: 1px solid rgba(25, 217, 197, 0.3);
        color: {COLOR_ACCENT_TEAL};
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }}

    /* Premium Dashboard Cards */
    .dark-card {{
        background-color: {COLOR_CARD};
        border: 1px solid {COLOR_BORDER};
        border-radius: 14px;
        padding: 1.25rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}

    .dark-card:hover {{
        border-color: rgba(25, 217, 197, 0.35);
    }}

    /* KPI Summary Card */
    .kpi-card {{
        background: {COLOR_CARD};
        border: 1px solid {COLOR_BORDER};
        border-radius: 14px;
        padding: 1.15rem;
        position: relative;
        overflow: hidden;
        transition: all 0.2s ease;
    }}

    .kpi-card:hover {{
        border-color: rgba(25, 217, 197, 0.4);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35), 0 0 12px rgba(25, 217, 197, 0.1);
    }}

    .kpi-icon-box {{
        width: 42px;
        height: 42px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        margin-bottom: 0.75rem;
    }}

    .kpi-title {{
        color: {COLOR_TEXT_MUTED};
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }}

    .kpi-value {{
        color: #FFFFFF;
        font-size: 1.85rem;
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
        line-height: 1.1;
        margin-bottom: 0.4rem;
    }}

    .kpi-delta {{
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.15rem 0.45rem;
    }}

    .delta-pos {{
        color: {COLOR_ACCENT_TEAL};
        background: rgba(25, 217, 197, 0.12);
    }}

    .delta-neg {{
        color: {COLOR_ACCENT_MAGENTA};
        background: rgba(236, 72, 153, 0.12);
    }}

    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {COLOR_ACCENT_TEAL} 0%, {COLOR_ACCENT_CYAN} 100%) !important;
        color: #061321 !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.25rem !important;
        font-family: 'Outfit', sans-serif !important;
        box-shadow: 0 4px 14px rgba(25, 217, 197, 0.25) !important;
        transition: all 0.2s ease !important;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(25, 217, 197, 0.4) !important;
    }}

    .stDownloadButton > button {{
        background: {COLOR_CARD} !important;
        color: {COLOR_ACCENT_TEAL} !important;
        border: 1px solid rgba(25, 217, 197, 0.4) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }}

    .stDownloadButton > button:hover {{
        background: rgba(25, 217, 197, 0.12) !important;
        border-color: {COLOR_ACCENT_TEAL} !important;
    }}

    /* Inputs, Selectboxes, Multiselect */
    .stTextInput input, .stSelectbox div[data-baseweb="select"], .stMultiSelect div[data-baseweb="select"] {{
        background-color: {COLOR_BG_SECONDARY} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
    }}

    .stSelectbox div[data-baseweb="select"]:hover, .stTextInput input:hover {{
        border-color: rgba(25, 217, 197, 0.4) !important;
    }}

    /* DataFrames / Tables */
    [data-testid="stDataFrame"] {{
        background: {COLOR_CARD} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {COLOR_BG_SECONDARY} !important;
        border-radius: 10px !important;
        padding: 0.25rem !important;
        border: 1px solid {COLOR_BORDER} !important;
        gap: 0.25rem !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        color: {COLOR_TEXT_MUTED} !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        padding: 0.45rem 1rem !important;
        border: none !important;
    }}

    .stTabs [aria-selected="true"] {{
        background: {COLOR_CARD} !important;
        color: {COLOR_ACCENT_TEAL} !important;
        font-weight: 600 !important;
        border: 1px solid rgba(25, 217, 197, 0.3) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
    }}

    /* Metric Default Component Overrides */
    [data-testid="stMetricValue"] {{
        font-family: 'Outfit', sans-serif !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }}

    [data-testid="stMetricLabel"] {{
        color: {COLOR_TEXT_MUTED} !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }}

    /* File Uploader */
    [data-testid="stFileUploader"] section {{
        background-color: {COLOR_CARD} !important;
        border: 2px dashed rgba(25, 217, 197, 0.3) !important;
        border-radius: 14px !important;
        padding: 2rem !important;
    }}

    [data-testid="stFileUploader"] section:hover {{
        border-color: {COLOR_ACCENT_TEAL} !important;
        background-color: rgba(25, 217, 197, 0.03) !important;
    }}

    /* Code Blocks */
    .stCodeBlock {{
        background-color: {COLOR_BG_PRIMARY} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 10px !important;
    }}

    /* Dividers */
    hr {{
        border-color: {COLOR_BORDER} !important;
        opacity: 0.6 !important;
    }}

    /* Info / Warning / Success Alerts */
    .stAlert {{
        background-color: {COLOR_CARD} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 10px !important;
        color: #E2E8F0 !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def apply_dark_theme(fig: go.Figure, title: str = "", height: int = 340) -> go.Figure:
    """Apply consistent Dark Intelligence aesthetics to any Plotly chart."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLOR_TEXT_MUTED, size=11),
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family="Outfit, sans-serif", size=15, color="#FFFFFF"),
            x=0.01,
            y=0.96,
        ),
        margin=dict(l=20, r=20, t=45 if title else 20, b=20),
        height=height,
        colorway=CHART_COLORWAY,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color=COLOR_TEXT_MUTED),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(22, 50, 77, 0.6)",
            gridwidth=1,
            linecolor=COLOR_BORDER,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(22, 50, 77, 0.6)",
            gridwidth=1,
            linecolor=COLOR_BORDER,
            zeroline=False,
        ),
        hoverlabel=dict(
            bgcolor=COLOR_CARD,
            bordercolor=COLOR_ACCENT_TEAL,
            font=dict(family="Inter, sans-serif", color="#FFFFFF", size=12),
        ),
    )
    return fig


def render_top_header() -> None:
    """Render the top search and status header bar without personal identity or login."""
    st.markdown(
        f"""
        <div class="top-header-container">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="background: linear-gradient(135deg, {COLOR_ACCENT_TEAL}, {COLOR_ACCENT_CYAN}); width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; color: #061321; font-weight: 700;">
                    📊
                </div>
                <div>
                    <div style="font-family: 'Outfit', sans-serif; font-weight: 700; color: #FFFFFF; font-size: 1.05rem; line-height: 1.1;">Retail Intelligence</div>
                    <div style="font-size: 0.75rem; color: {COLOR_TEXT_MUTED};">Sales Forecasting & Analytics</div>
                </div>
            </div>
            <div class="top-search-pill">
                <span>🔍</span>
                <span style="flex-grow: 1;">Search insights, metrics, or datasets...</span>
                <span style="background: {COLOR_CARD}; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.7rem; border: 1px solid {COLOR_BORDER};">Ctrl + K</span>
            </div>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div class="top-badge-pill">● System Live</div>
                <div style="display: flex; align-items: center; gap: 0.5rem; background: {COLOR_CARD}; padding: 0.35rem 0.75rem; border-radius: 8px; border: 1px solid {COLOR_BORDER};">
                    <div style="width: 24px; height: 24px; border-radius: 6px; background: linear-gradient(135deg, {COLOR_ACCENT_PURPLE}, {COLOR_ACCENT_BLUE}); display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; color: #FFFFFF;">
                        DA
                    </div>
                    <div>
                        <div style="font-size: 0.8rem; font-weight: 600; color: #FFFFFF; line-height: 1.1;">Data Analyst</div>
                        <div style="font-size: 0.68rem; color: {COLOR_TEXT_MUTED};">Analytics Workspace</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
