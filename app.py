"""Main application entry point for Retail Intelligence & Sales Forecasting."""

from __future__ import annotations

import streamlit as st

from src.analytics.ui import render_eda_page
from src.forecasting.ui import render_forecasting_ui
from src.insights.ui import render_business_insights_page
from src.reporting.powerbi.ui import render_powerbi_page
from src.ui.overview import render_overview_page
from src.ui.pages import (
    render_cleaning_page,
    render_excel_report_page,
    render_pdf_report_page,
    render_understanding_page,
    render_upload_page,
)
from src.ui.theme import (
    COLOR_ACCENT_CYAN,
    COLOR_ACCENT_TEAL,
    COLOR_BORDER,
    COLOR_CARD,
    COLOR_TEXT_MUTED,
    inject_theme_css,
    render_top_header,
)


st.set_page_config(
    page_title="Retail Intelligence & Sales Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Run the Retail Intelligence Dark Intelligence application."""

    # 1. Inject Premium Dark Intelligence Design System
    inject_theme_css()

    # 2. Sidebar Navigation
    with st.sidebar:
        # Sidebar Header Branding
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.5rem 0.25rem 1rem 0.25rem; border-bottom: 1px solid {COLOR_BORDER}; margin-bottom: 1rem;">
                <div style="background: linear-gradient(135deg, {COLOR_ACCENT_TEAL}, {COLOR_ACCENT_CYAN}); width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.25rem; color: #061321; font-weight: 700; box-shadow: 0 0 16px rgba(25, 217, 197, 0.35);">
                    📊
                </div>
                <div>
                    <div style="font-family: 'Outfit', sans-serif; font-weight: 700; color: #FFFFFF; font-size: 1.15rem; line-height: 1.1;">Retail Intelligence</div>
                    <div style="font-size: 0.75rem; color: {COLOR_TEXT_MUTED};">& Sales Forecasting</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='font-size: 0.72rem; font-weight: 700; color: {COLOR_TEXT_MUTED}; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.4rem; padding-left: 0.5rem;'>MAIN MENU</div>",
            unsafe_allow_html=True,
        )

        menu_options = [
            "🏠 Overview",
            "📤 Data Upload",
            "📊 Data Understanding",
            "🧹 Data Cleaning",
            "🔍 Exploratory Analysis",
            "💡 Business Insights",
            "📈 Forecasting",
            "📑 Excel Report",
            "📊 Power BI Dashboard",
            "📄 PDF Report",
        ]

        selected_page = st.radio(
            "Navigation",
            options=menu_options,
            index=0,
            key="main_navigation_radio",
        )

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        # Bottom Growth Promo Card
        st.markdown(
            f"""
            <div style="background: {COLOR_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 14px; padding: 1rem; position: relative; overflow: hidden; box-shadow: 0 4px 16px rgba(0,0,0,0.3);">
                <div style="width: 32px; height: 32px; border-radius: 8px; background: rgba(25, 217, 197, 0.15); display: flex; align-items: center; justify-content: center; color: {COLOR_ACCENT_TEAL}; font-size: 1.1rem; margin-bottom: 0.6rem;">
                    📈
                </div>
                <div style="font-family: 'Outfit', sans-serif; font-weight: 700; color: #FFFFFF; font-size: 0.95rem; line-height: 1.2; margin-bottom: 0.25rem;">
                    Turn Data Into Business Growth
                </div>
                <div style="font-size: 0.73rem; color: {COLOR_TEXT_MUTED}; line-height: 1.3;">
                    Upload. Analyze. Forecast. Visualize. Decide.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        st.caption("v0.1.0 • Dark Intelligence SaaS")

    # 3. Render Top Status Bar Header
    render_top_header()

    # 4. Route Page Views
    clean_page_name = selected_page.split(" ", 1)[1] if " " in selected_page else selected_page

    if clean_page_name == "Overview":
        render_overview_page()

    elif clean_page_name == "Data Upload":
        render_upload_page()

    elif clean_page_name == "Data Understanding":
        render_understanding_page()

    elif clean_page_name == "Data Cleaning":
        render_cleaning_page()

    elif clean_page_name == "Exploratory Analysis":
        render_eda_page()

    elif clean_page_name == "Business Insights":
        render_business_insights_page()

    elif clean_page_name == "Forecasting":
        render_forecasting_ui()

    elif clean_page_name == "Excel Report":
        render_excel_report_page()

    elif clean_page_name == "Power BI Dashboard":
        render_powerbi_page()

    elif clean_page_name == "PDF Report":
        render_pdf_report_page()


if __name__ == "__main__":
    main()
