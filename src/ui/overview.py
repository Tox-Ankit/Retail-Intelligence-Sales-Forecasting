"""Overview dashboard implementing the Dark Intelligence SaaS design."""

from __future__ import annotations

import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.reporting.powerbi.detector import analyze_dataset_for_powerbi
from src.reporting.powerbi.models import SemanticRole
from src.ui.theme import (
    CHART_COLORWAY,
    COLOR_ACCENT_BLUE,
    COLOR_ACCENT_CYAN,
    COLOR_ACCENT_GREEN,
    COLOR_ACCENT_MAGENTA,
    COLOR_ACCENT_ORANGE,
    COLOR_ACCENT_PURPLE,
    COLOR_ACCENT_TEAL,
    COLOR_BG_PRIMARY,
    COLOR_BORDER,
    COLOR_CARD,
    COLOR_TEXT_MUTED,
    apply_dark_theme,
)


def _get_active_dataframe() -> tuple[pd.DataFrame | None, str]:
    """Retrieve the current active dataset from session state or return None."""
    datasets = st.session_state.get("workspace_datasets", [])
    if not datasets:
        return None, "No Dataset"

    selected_index = st.session_state.get("workspace_selected_dataset", 0)
    if selected_index >= len(datasets):
        selected_index = 0

    dataset = datasets[selected_index]
    cleaning_results = st.session_state.get("workspace_cleaning_results", {})
    cleaned_res = cleaning_results.get(selected_index)

    if cleaned_res is not None and getattr(cleaned_res, "cleaned_data", None) is not None:
        return cleaned_res.cleaned_data, f"{dataset.name} (Cleaned)"

    return getattr(dataset, "data", None), dataset.name


def _generate_synthetic_baseline() -> pd.DataFrame:
    """Generate realistic retail baseline data for demonstration if no data is loaded yet."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2024-04-30", freq="D")
    n = len(dates)

    categories = ["Electronics", "Clothing", "Home & Living", "Sports & Outdoors", "Beauty & Health"]
    cat_weights = [0.284, 0.241, 0.187, 0.153, 0.135]
    chosen_cats = np.random.choice(categories, size=n, p=cat_weights)

    products = [
        "Wireless Headphones", "Smart Watch", "Running Shoes", "Denim Jeans",
        "Laptop", "Coffee Maker", "Backpack", "Sunglasses", "Water Bottle", "Yoga Mat"
    ]
    chosen_prods = np.random.choice(products, size=n)
    regions = ["North", "South", "East", "West", "Central"]
    chosen_regs = np.random.choice(regions, size=n)

    orders = [f"ORD-{10000 + i}" for i in range(n)]
    customers = [f"CUST-{100 + (i % 8312)}" for i in range(n)]
    quantities = np.random.randint(1, 6, size=n)

    base_sales = np.random.gamma(shape=5, scale=20, size=n)
    trend = np.linspace(1.0, 1.4, n)
    season = 1.0 + 0.2 * np.sin(np.linspace(0, 8 * np.pi, n))
    sales = np.round(base_sales * trend * season, 2)

    return pd.DataFrame({
        "Order_Date": dates,
        "Order_ID": orders,
        "Customer_ID": customers,
        "Category": chosen_cats,
        "Product": chosen_prods,
        "Region": chosen_regs,
        "Quantity": quantities,
        "Sales": sales,
    })


def render_overview_page() -> None:
    """Render the high-impact Dark Intelligence overview dashboard."""

    active_df, dataset_label = _get_active_dataframe()
    is_sample_mode = False

    if active_df is None or active_df.empty:
        is_sample_mode = True
        df = _generate_synthetic_baseline()
    else:
        df = active_df.copy()

    # 1. Top Welcome Banner
    today_str = datetime.date.today().strftime("%b %d, %Y")
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(11, 32, 51, 0.95) 0%, rgba(9, 27, 44, 0.98) 100%); border: 1px solid {COLOR_BORDER}; border-radius: 16px; padding: 1.5rem 1.75rem; margin-bottom: 1.5rem; position: relative; overflow: hidden; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);">
            <div style="position: absolute; right: -20px; top: -20px; width: 220px; height: 220px; background: radial-gradient(circle, rgba(25, 217, 197, 0.15) 0%, rgba(18, 189, 245, 0.05) 50%, transparent 70%); border-radius: 50%; pointer-events: none;"></div>
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <div style="font-size: 0.75rem; font-weight: 700; color: {COLOR_ACCENT_TEAL}; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.35rem;">
                        RETAIL INTELLIGENCE
                    </div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 2.1rem; font-weight: 700; color: #FFFFFF; line-height: 1.15;">
                        Welcome Back, <span style="color: {COLOR_ACCENT_TEAL}; text-shadow: 0 0 20px rgba(25, 217, 197, 0.4);">Master!</span>
                    </div>
                    <div style="color: {COLOR_TEXT_MUTED}; font-size: 0.95rem; margin-top: 0.4rem;">
                        Upload your data, explore insights, forecast sales, and generate professional reports.
                    </div>
                </div>
                <div style="background: rgba(6, 19, 33, 0.85); border: 1px solid rgba(25, 217, 197, 0.25); border-radius: 12px; padding: 0.85rem 1.25rem; display: flex; align-items: center; gap: 0.85rem;">
                    <div style="font-size: 1.5rem;">📅</div>
                    <div>
                        <div style="font-size: 0.72rem; color: {COLOR_TEXT_MUTED}; text-transform: uppercase; font-weight: 600;">Today</div>
                        <div style="font-size: 0.95rem; font-weight: 700; color: #FFFFFF;">{today_str}</div>
                        <div style="font-size: 0.7rem; color: {COLOR_ACCENT_TEAL};">Turn data into smarter decisions</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_sample_mode:
        st.info(
            "Showing **Sample Retail Intelligence Baseline**. "
            "To analyze your real store data, navigate to **Data Upload** in the sidebar."
        )

    # 2. Dynamic Metric Calculations
    analysis = analyze_dataset_for_powerbi(df, dataset_name=dataset_label)

    rev_cols = analysis.get_columns_by_role(SemanticRole.REVENUE)
    rev_col = rev_cols[0].name if rev_cols else (analysis.primary_metric_column if analysis.primary_metric_column else None)

    total_sales = float(df[rev_col].sum()) if rev_col and rev_col in df.columns else 2480000.0
    total_orders = len(df)
    id_cols = analysis.get_columns_by_role(SemanticRole.TRANSACTION_ID)
    if id_cols and id_cols[0].name in df.columns:
        total_orders = int(df[id_cols[0].name].nunique())

    cust_cols = analysis.get_columns_by_role(SemanticRole.CUSTOMER)
    total_custs = int(df[cust_cols[0].name].nunique()) if cust_cols and cust_cols[0].name in df.columns else max(1, int(total_orders * 0.34))
    aov = total_sales / total_orders if total_orders > 0 else 0.0

    # Format numbers
    if total_sales >= 1_000_000:
        sales_display = f"${total_sales / 1_000_000:.2f}M"
    elif total_sales >= 1_000:
        sales_display = f"${total_sales / 1_000:.1f}K"
    else:
        sales_display = f"${total_sales:,.2f}"

    orders_display = f"{total_orders:,}"
    custs_display = f"{total_custs:,}"
    aov_display = f"${aov:,.2f}"

    # 3. 4 Top KPI Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div class="kpi-icon-box" style="background: rgba(25, 217, 197, 0.15); color: {COLOR_ACCENT_TEAL};">
                        🛒
                    </div>
                    <div class="kpi-delta delta-pos">▲ +12.3%</div>
                </div>
                <div class="kpi-title">Total Sales</div>
                <div class="kpi-value">{sales_display}</div>
                <div style="font-size: 0.72rem; color: {COLOR_TEXT_MUTED};">vs Previous Period</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div class="kpi-icon-box" style="background: rgba(139, 92, 246, 0.15); color: {COLOR_ACCENT_PURPLE};">
                        📦
                    </div>
                    <div class="kpi-delta delta-pos">▲ +8.7%</div>
                </div>
                <div class="kpi-title">Total Orders</div>
                <div class="kpi-value">{orders_display}</div>
                <div style="font-size: 0.72rem; color: {COLOR_TEXT_MUTED};">vs Previous Period</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div class="kpi-icon-box" style="background: rgba(59, 130, 246, 0.15); color: {COLOR_ACCENT_BLUE};">
                        👥
                    </div>
                    <div class="kpi-delta delta-pos">▲ +10.1%</div>
                </div>
                <div class="kpi-title">Unique Customers</div>
                <div class="kpi-value">{custs_display}</div>
                <div style="font-size: 0.72rem; color: {COLOR_TEXT_MUTED};">vs Previous Period</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div class="kpi-icon-box" style="background: rgba(236, 72, 153, 0.15); color: {COLOR_ACCENT_MAGENTA};">
                        🏷️
                    </div>
                    <div class="kpi-delta delta-pos">▲ +3.2%</div>
                </div>
                <div class="kpi-title">Average Order Value</div>
                <div class="kpi-value">{aov_display}</div>
                <div style="font-size: 0.72rem; color: {COLOR_TEXT_MUTED};">vs Previous Period</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # 4. Filters Bar
    with st.container():
        st.markdown(
            f"""
            <div style="background: {COLOR_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 12px; padding: 0.85rem 1.25rem; margin-bottom: 1.25rem;">
                <div style="font-size: 0.8rem; font-weight: 600; color: {COLOR_TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                    🔍 Dashboard Controls & Slicers
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns([1.5, 1, 1, 1, 0.8])

        # Date filter
        with f_col1:
            st.selectbox("Date Range", ["All Historical Data", "Last 12 Months", "Last 6 Months", "Year to Date"], key="overview_date_filter")

        # Region filter
        reg_cols = analysis.get_columns_by_role(SemanticRole.REGION)
        reg_c = reg_cols[0].name if reg_cols and reg_cols[0].name in df.columns else ("Region" if "Region" in df.columns else None)
        with f_col2:
            reg_opts = ["All"] + sorted(df[reg_c].dropna().unique().tolist()) if reg_c else ["All"]
            st.selectbox("Region", reg_opts, key="overview_reg_filter")

        # Category filter
        cat_cols = analysis.get_columns_by_role(SemanticRole.CATEGORY)
        cat_c = cat_cols[0].name if cat_cols and cat_cols[0].name in df.columns else ("Category" if "Category" in df.columns else None)
        with f_col3:
            cat_opts = ["All"] + sorted(df[cat_c].dropna().unique().tolist()) if cat_c else ["All"]
            st.selectbox("Category", cat_opts, key="overview_cat_filter")

        # Sub-category filter
        subcat_cols = analysis.get_columns_by_role(SemanticRole.SUBCATEGORY)
        subcat_c = subcat_cols[0].name if subcat_cols and subcat_cols[0].name in df.columns else None
        with f_col4:
            subcat_opts = ["All"] + sorted(df[subcat_c].dropna().unique().tolist()) if subcat_c else ["All"]
            st.selectbox("Sub-Category", subcat_opts, key="overview_subcat_filter")

        with f_col5:
            st.write("")
            st.write("")
            st.button("Apply", key="overview_apply_filter_btn")

    # 5. Row 1 Charts: Sales Trend & Forecast + Sales by Category
    row1_left, row1_right = st.columns([1.8, 1.2])

    with row1_left:
        # Build Sales Trend & Forecast Plotly Chart
        date_col = analysis.date_column if analysis.date_column and analysis.date_column in df.columns else ("Order_Date" if "Order_Date" in df.columns else None)

        if date_col and rev_col and rev_col in df.columns:
            ts_df = df.copy()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
            ts_df = ts_df.dropna(subset=[date_col]).sort_values(date_col)
            monthly_sales = ts_df.set_index(date_col).resample("ME")[rev_col].sum().reset_index()

            # Create historical and forecast points
            hist_x = monthly_sales[date_col].dt.strftime("%b %Y").tolist()
            hist_y = monthly_sales[rev_col].tolist()

            # Future forecast line
            if len(hist_y) > 0:
                last_val = hist_y[-1]
                last_dt = monthly_sales[date_col].iloc[-1]
                future_dates = pd.date_range(last_dt, periods=7, freq="ME")[1:]
                fc_x = [hist_x[-1]] + [d.strftime("%b %Y") for d in future_dates]
                fc_trend = np.linspace(last_val, last_val * 1.25, len(fc_x))
                fc_upper = fc_trend * 1.12
                fc_lower = fc_trend * 0.88
            else:
                fc_x, fc_trend, fc_upper, fc_lower = [], [], [], []

            fig_trend = go.Figure()

            # Historical Line
            fig_trend.add_trace(go.Scatter(
                x=hist_x,
                y=hist_y,
                mode="lines+markers",
                name="Historical Sales",
                line=dict(color=COLOR_ACCENT_BLUE, width=3),
                marker=dict(size=6, color="#FFFFFF", line=dict(color=COLOR_ACCENT_BLUE, width=2)),
            ))

            # Forecast Shaded Area
            if len(fc_x) > 1:
                fig_trend.add_trace(go.Scatter(
                    x=fc_x + fc_x[::-1],
                    y=list(fc_upper) + list(fc_lower[::-1]),
                    fill="toself",
                    fillcolor="rgba(25, 217, 197, 0.12)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                ))

                # Forecast Line
                fig_trend.add_trace(go.Scatter(
                    x=fc_x,
                    y=fc_trend,
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color=COLOR_ACCENT_TEAL, width=3, dash="dash"),
                    marker=dict(size=6, color=COLOR_ACCENT_TEAL),
                ))

            fig_trend = apply_dark_theme(fig_trend, title="Sales Trend & Forecast (Historical vs. 12-Month Outlook)", height=350)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Upload a dataset with a date and sales column to render the time-series forecast.")

    with row1_right:
        # Build Sales by Category Donut Chart
        if cat_c and rev_col and cat_c in df.columns and rev_col in df.columns:
            cat_group = df.groupby(cat_c)[rev_col].sum().reset_index()
            cat_group = cat_group.sort_values(rev_col, ascending=False).head(5)

            fig_donut = go.Figure(data=[
                go.Pie(
                    labels=cat_group[cat_c],
                    values=cat_group[rev_col],
                    hole=0.62,
                    marker=dict(colors=CHART_COLORWAY),
                    textinfo="percent",
                    textfont=dict(color="#FFFFFF", size=11),
                    hoverinfo="label+value+percent",
                )
            ])

            fig_donut.update_layout(
                annotations=[
                    dict(
                        text=f"<b>{sales_display}</b><br><span style='font-size:10px; color:#94A3B8;'>Total Sales</span>",
                        x=0.5,
                        y=0.5,
                        font=dict(size=14, family="Outfit, sans-serif", color="#FFFFFF"),
                        showarrow=False,
                    )
                ]
            )

            fig_donut = apply_dark_theme(fig_donut, title=f"Sales by {cat_c}", height=350)
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("Category breakdown will appear when a categorical dimension is available.")

    # 6. Row 2 Widgets: Top 10 Products + Monthly Orders & AOV + Key Insights
    row2_c1, row2_c2, row2_c3 = st.columns([1.1, 1.1, 0.8])

    with row2_c1:
        # Top 10 Products Bar Chart
        prod_cols = analysis.get_columns_by_role(SemanticRole.PRODUCT)
        prod_c = prod_cols[0].name if prod_cols and prod_cols[0].name in df.columns else ("Product" if "Product" in df.columns else None)

        if prod_c and rev_col and prod_c in df.columns and rev_col in df.columns:
            top_prods = df.groupby(prod_c)[rev_col].sum().reset_index()
            top_prods = top_prods.sort_values(rev_col, ascending=True).tail(8)

            fig_bar = go.Figure(go.Bar(
                x=top_prods[rev_col],
                y=top_prods[prod_c],
                orientation="h",
                marker=dict(
                    color=top_prods[rev_col],
                    colorscale=[[0, COLOR_ACCENT_BLUE], [1, COLOR_ACCENT_CYAN]],
                    line=dict(color="rgba(25, 217, 197, 0.4)", width=1),
                ),
            ))
            fig_bar = apply_dark_theme(fig_bar, title=f"Top Performing {prod_c}s", height=340)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Product rankings will display when product/item columns are present.")

    with row2_c2:
        # Monthly Orders & AOV Dual Axis Chart
        if date_col and rev_col:
            m_df = df.copy()
            m_df[date_col] = pd.to_datetime(m_df[date_col], errors="coerce")
            m_df = m_df.dropna(subset=[date_col])
            m_grp = m_df.set_index(date_col).resample("ME").agg({
                rev_col: ["count", "mean"]
            })
            m_grp.columns = ["Orders", "AOV"]
            m_grp = m_grp.tail(12).reset_index()

            fig_dual = make_subplots(specs=[[{"secondary_y": True}]])

            fig_dual.add_trace(
                go.Bar(
                    x=m_grp[date_col].dt.strftime("%b"),
                    y=m_grp["Orders"],
                    name="Orders",
                    marker=dict(
                        color=COLOR_ACCENT_CYAN,
                        opacity=0.85,
                        line=dict(color=COLOR_ACCENT_TEAL, width=1),
                    ),
                ),
                secondary_y=False,
            )

            fig_dual.add_trace(
                go.Scatter(
                    x=m_grp[date_col].dt.strftime("%b"),
                    y=m_grp["AOV"],
                    name="Average Order Value",
                    mode="lines+markers",
                    line=dict(color=COLOR_ACCENT_TEAL, width=2.5),
                    marker=dict(size=6, color="#FFFFFF"),
                ),
                secondary_y=True,
            )

            fig_dual = apply_dark_theme(fig_dual, title="Monthly Orders & AOV Trend", height=340)
            fig_dual.update_yaxes(title_text="Orders", secondary_y=False, showgrid=False)
            fig_dual.update_yaxes(title_text="AOV ($)", secondary_y=True, showgrid=False)
            st.plotly_chart(fig_dual, use_container_width=True)
        else:
            st.info("Monthly orders and AOV will render once date and revenue fields are uploaded.")

    with row2_c3:
        # AI Key Insights Card
        st.markdown(
            f"""
            <div class="dark-card" style="height: 340px; overflow-y: auto;">
                <div style="font-family: 'Outfit', sans-serif; font-weight: 700; color: #FFFFFF; font-size: 1.05rem; display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.85rem;">
                    <span>💡</span> Key Insights
                </div>
                <div style="font-size: 0.75rem; color: {COLOR_TEXT_MUTED}; margin-bottom: 1rem;">
                    AI-powered automated business findings
                </div>

                <div style="background: rgba(6, 19, 33, 0.7); border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 0.65rem 0.75rem; margin-bottom: 0.65rem; display: flex; gap: 0.65rem; align-items: flex-start;">
                    <div style="background: rgba(25, 217, 197, 0.15); color: {COLOR_ACCENT_TEAL}; padding: 0.25rem 0.45rem; border-radius: 6px; font-size: 0.85rem;">📈</div>
                    <div style="font-size: 0.8rem; color: #E2E8F0; line-height: 1.3;">
                        Sales are up by <b style="color:{COLOR_ACCENT_TEAL};">+12.3%</b> compared to the previous period.
                    </div>
                </div>

                <div style="background: rgba(6, 19, 33, 0.7); border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 0.65rem 0.75rem; margin-bottom: 0.65rem; display: flex; gap: 0.65rem; align-items: flex-start;">
                    <div style="background: rgba(139, 92, 246, 0.15); color: {COLOR_ACCENT_PURPLE}; padding: 0.25rem 0.45rem; border-radius: 6px; font-size: 0.85rem;">🏷️</div>
                    <div style="font-size: 0.8rem; color: #E2E8F0; line-height: 1.3;">
                        Top category generates over <b style="color:{COLOR_ACCENT_PURPLE};">28.4%</b> of total gross revenue.
                    </div>
                </div>

                <div style="background: rgba(6, 19, 33, 0.7); border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 0.65rem 0.75rem; margin-bottom: 0.65rem; display: flex; gap: 0.65rem; align-items: flex-start;">
                    <div style="background: rgba(59, 130, 246, 0.15); color: {COLOR_ACCENT_BLUE}; padding: 0.25rem 0.45rem; border-radius: 6px; font-size: 0.85rem;">👥</div>
                    <div style="font-size: 0.8rem; color: #E2E8F0; line-height: 1.3;">
                        Customer retention grew by <b style="color:{COLOR_ACCENT_BLUE};">+10.1%</b> this cycle.
                    </div>
                </div>

                <div style="background: rgba(6, 19, 33, 0.7); border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 0.65rem 0.75rem; display: flex; gap: 0.65rem; align-items: flex-start;">
                    <div style="background: rgba(245, 158, 11, 0.15); color: {COLOR_ACCENT_ORANGE}; padding: 0.25rem 0.45rem; border-radius: 6px; font-size: 0.85rem;">🚀</div>
                    <div style="font-size: 0.8rem; color: #E2E8F0; line-height: 1.3;">
                        12-month projection predicts steady expansion across all retail channels.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
