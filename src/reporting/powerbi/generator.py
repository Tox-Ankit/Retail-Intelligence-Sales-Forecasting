"""Generator for Power BI integration artifacts (Data, PBIDS, M-Code, DAX, Visual Recipes, ZIP)."""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import pandas as pd

from src.reporting.powerbi.detector import analyze_dataset_for_powerbi
from src.reporting.powerbi.models import (
    ColumnSemanticProfile,
    DaxMeasure,
    PowerBiModelAnalysis,
    PowerBiPackage,
    SemanticRole,
    VisualRecommendation,
)


def _sanitize_name_for_dax(name: str) -> str:
    """Format a column name safely for DAX formulas."""
    return f"'{name}'" if (" " in name or "-" in name) else name


def generate_dax_measures(
    analysis: PowerBiModelAnalysis,
    table_name: str = "Dataset",
) -> list[DaxMeasure]:
    """Generate dynamic DAX measure formulas based on detected semantic roles."""
    measures: list[DaxMeasure] = []

    # 1. Base Record / Transaction Count Measure
    id_cols = analysis.get_columns_by_role(SemanticRole.TRANSACTION_ID)
    if id_cols:
        col_name = id_cols[0].name
        measures.append(
            DaxMeasure(
                name="Total Transactions",
                dax_formula=f"Total Transactions = DISTINCTCOUNT('{table_name}'[{col_name}])",
                display_folder="Key Metrics",
                description="Total number of unique transactions / orders.",
                format_string="#,##0",
                business_question="How many distinct orders or transactions occurred?",
            )
        )
    else:
        measures.append(
            DaxMeasure(
                name="Total Records",
                dax_formula=f"Total Records = COUNTROWS('{table_name}')",
                display_folder="Key Metrics",
                description="Total number of rows in the dataset.",
                format_string="#,##0",
                business_question="What is the total volume of logged records?",
            )
        )

    # 2. Revenue / Sales Measures
    rev_cols = analysis.get_columns_by_role(SemanticRole.REVENUE)
    rev_col = rev_cols[0].name if rev_cols else None
    if rev_col:
        measures.append(
            DaxMeasure(
                name="Total Revenue",
                dax_formula=f"Total Revenue = SUM('{table_name}'[{rev_col}])",
                display_folder="Key Metrics",
                description="Sum of all gross or net revenue.",
                format_string="$#,##0.00;($#,##0.00);$0.00",
                business_question="What is the total monetary revenue generated?",
            )
        )
        measures.append(
            DaxMeasure(
                name="Average Revenue per Row",
                dax_formula=f"Average Revenue per Row = AVERAGE('{table_name}'[{rev_col}])",
                display_folder="Key Metrics",
                description="Average revenue per item or transaction row.",
                format_string="$#,##0.00;($#,##0.00);$0.00",
                business_question="What is the mean transaction or line item size?",
            )
        )

    # 3. Quantity Measures
    qty_cols = analysis.get_columns_by_role(SemanticRole.QUANTITY)
    qty_col = qty_cols[0].name if qty_cols else None
    if qty_col:
        measures.append(
            DaxMeasure(
                name="Total Quantity Sold",
                dax_formula=f"Total Quantity Sold = SUM('{table_name}'[{qty_col}])",
                display_folder="Key Metrics",
                description="Total unit volume sold.",
                format_string="#,##0",
                business_question="How many total units were sold/dispatched?",
            )
        )
        measures.append(
            DaxMeasure(
                name="Average Units per Order",
                dax_formula=f"Average Units per Order = AVERAGE('{table_name}'[{qty_col}])",
                display_folder="Key Metrics",
                description="Average number of units per transaction line.",
                format_string="#,##0.00",
                business_question="What is the average basket quantity?",
            )
        )

    # 4. Profit & Profit Margin Measures
    profit_cols = analysis.get_columns_by_role(SemanticRole.PROFIT)
    profit_col = profit_cols[0].name if profit_cols else None
    if profit_col:
        measures.append(
            DaxMeasure(
                name="Total Profit",
                dax_formula=f"Total Profit = SUM('{table_name}'[{profit_col}])",
                display_folder="Profitability",
                description="Total net profit generated.",
                format_string="$#,##0.00;($#,##0.00);$0.00",
                business_question="What is our total net profitability?",
            )
        )
        if rev_col:
            measures.append(
                DaxMeasure(
                    name="Profit Margin %",
                    dax_formula="Profit Margin % = DIVIDE([Total Profit], [Total Revenue], 0)",
                    display_folder="Profitability",
                    description="Net profit as a percentage of total revenue.",
                    format_string="0.00%",
                    business_question="What proportion of revenue is retained as profit?",
                )
            )

    # 5. Cost Measure
    cost_cols = analysis.get_columns_by_role(SemanticRole.COST)
    if cost_cols:
        cost_col = cost_cols[0].name
        measures.append(
            DaxMeasure(
                name="Total Cost",
                dax_formula=f"Total Cost = SUM('{table_name}'[{cost_col}])",
                display_folder="Cost Analysis",
                description="Total direct cost or expenditure.",
                format_string="$#,##0.00;($#,##0.00);$0.00",
                business_question="What were the total incurred costs?",
            )
        )

    # 6. Customer Metrics
    cust_cols = analysis.get_columns_by_role(SemanticRole.CUSTOMER)
    if cust_cols:
        cust_col = cust_cols[0].name
        measures.append(
            DaxMeasure(
                name="Total Customers",
                dax_formula=f"Total Customers = DISTINCTCOUNT('{table_name}'[{cust_col}])",
                display_folder="Customer Intelligence",
                description="Number of distinct active customers.",
                format_string="#,##0",
                business_question="How many distinct customers purchased?",
            )
        )
        if rev_col:
            measures.append(
                DaxMeasure(
                    name="Revenue per Customer",
                    dax_formula="Revenue per Customer = DIVIDE([Total Revenue], [Total Customers], 0)",
                    display_folder="Customer Intelligence",
                    description="Average revenue contribution per customer.",
                    format_string="$#,##0.00",
                    business_question="What is the average customer lifetime / basket value?",
                )
            )

    # 7. Generic Numerics if no standard retail revenue/profit
    if not rev_col and not profit_col and not qty_col:
        generic_cols = analysis.get_columns_by_role(SemanticRole.GENERIC_NUMERIC)
        for g_col in generic_cols[:3]:
            measures.append(
                DaxMeasure(
                    name=f"Total {g_col.name}",
                    dax_formula=f"Total {g_col.name} = SUM('{table_name}'[{g_col.name}])",
                    display_folder="Aggregations",
                    description=f"Sum aggregation for {g_col.name}.",
                    format_string="#,##0.00",
                    business_question=f"What is the total sum of {g_col.name}?",
                )
            )
            measures.append(
                DaxMeasure(
                    name=f"Average {g_col.name}",
                    dax_formula=f"Average {g_col.name} = AVERAGE('{table_name}'[{g_col.name}])",
                    display_folder="Aggregations",
                    description=f"Average value for {g_col.name}.",
                    format_string="#,##0.00",
                    business_question=f"What is the mean value of {g_col.name}?",
                )
            )

    # 8. Time-Intelligence Measures if Date Column exists
    if analysis.date_column and (rev_col or profit_col):
        primary_val_measure = "[Total Revenue]" if rev_col else "[Total Profit]"
        date_c = analysis.date_column
        measures.append(
            DaxMeasure(
                name="Revenue YTD",
                dax_formula=f"Revenue YTD = TOTALYTD({primary_val_measure}, '{table_name}'[{date_c}])",
                display_folder="Time Intelligence",
                description="Year-to-date cumulative primary metric.",
                format_string="$#,##0.00",
                business_question="What is the cumulative year-to-date performance?",
            )
        )

    return measures


def generate_visual_recommendations(
    analysis: PowerBiModelAnalysis,
    table_name: str = "Dataset",
) -> list[VisualRecommendation]:
    """Generate visual layout recommendations tailored to available fields."""
    recs: list[VisualRecommendation] = []

    # 1. Executive KPI Cards
    kpi_fields = []
    rev_col = analysis.get_columns_by_role(SemanticRole.REVENUE)
    profit_col = analysis.get_columns_by_role(SemanticRole.PROFIT)
    qty_col = analysis.get_columns_by_role(SemanticRole.QUANTITY)
    cust_col = analysis.get_columns_by_role(SemanticRole.CUSTOMER)
    id_col = analysis.get_columns_by_role(SemanticRole.TRANSACTION_ID)

    if rev_col:
        kpi_fields.append("[Total Revenue]")
    if profit_col:
        kpi_fields.append("[Total Profit]")
        if rev_col:
            kpi_fields.append("[Profit Margin %]")
    if qty_col:
        kpi_fields.append("[Total Quantity Sold]")
    if cust_col:
        kpi_fields.append("[Total Customers]")
    elif id_col:
        kpi_fields.append("[Total Transactions]")

    if kpi_fields:
        recs.append(
            VisualRecommendation(
                title="Executive KPI Header (Card Visuals)",
                visual_type="Multi-row Card or Individual KPI Cards",
                fields={"KPI Values": kpi_fields},
                description="Top-level summary cards displaying headline performance metrics.",
                configuration_notes="Place across the top band of the dashboard page for immediate executive context.",
                suggested_page="Executive Overview",
            )
        )

    # 2. Time-Series Trend Line Chart
    if analysis.date_column and (rev_col or profit_col or qty_col or analysis.primary_metric_column):
        metric = rev_col[0].name if rev_col else (profit_col[0].name if profit_col else analysis.primary_metric_column)
        recs.append(
            VisualRecommendation(
                title=f"{metric} Over Time Trend",
                visual_type="Line and Clustered Column Chart",
                fields={
                    "X-Axis (Shared Axis)": f"'{table_name}'[{analysis.date_column}]",
                    "Column Values (Primary)": f"[Total {metric}]" if f"[Total {metric}]" in [m.name for m in analysis.dax_measures] else f"'{table_name}'[{metric}]",
                },
                description="Historical progression tracking seasonality, peaks, and long-term trend lines.",
                configuration_notes="Turn on Data Labels and add a Trend Line in the Power BI Analytics pane.",
                suggested_page="Executive Overview",
            )
        )

    # 3. Category / Sub-Category Breakdown Bar Chart
    cat_cols = analysis.get_columns_by_role(SemanticRole.CATEGORY)
    subcat_cols = analysis.get_columns_by_role(SemanticRole.SUBCATEGORY)
    if cat_cols and (rev_col or profit_col or analysis.primary_metric_column):
        cat_c = cat_cols[0].name
        val_c = rev_col[0].name if rev_col else (profit_col[0].name if profit_col else analysis.primary_metric_column)
        drilldown = [f"'{table_name}'[{cat_c}]"]
        if subcat_cols:
            drilldown.append(f"'{table_name}'[{subcat_cols[0].name}]")

        recs.append(
            VisualRecommendation(
                title=f"Performance by {cat_c}",
                visual_type="Clustered Bar Chart",
                fields={
                    "Y-Axis (Category)": drilldown,
                    "X-Axis (Measure)": f"[Total {val_c}]" if rev_col or profit_col else f"'{table_name}'[{val_c}]",
                },
                description="Ranked comparative performance across business categories with drill-down support.",
                configuration_notes="Sort descending by metric. Enable drill-mode to expand into sub-categories.",
                suggested_page="Executive Overview",
            )
        )

    # 4. Regional / Geographic Breakdown
    region_cols = analysis.get_columns_by_role(SemanticRole.REGION)
    if region_cols and (rev_col or profit_col or analysis.primary_metric_column):
        reg_c = region_cols[0].name
        val_c = rev_col[0].name if rev_col else (profit_col[0].name if profit_col else analysis.primary_metric_column)
        recs.append(
            VisualRecommendation(
                title=f"Regional Contribution ({reg_c})",
                visual_type="Filled Map / Treemap / Donut Chart",
                fields={
                    "Location / Category": f"'{table_name}'[{reg_c}]",
                    "Values / Size": f"'{table_name}'[{val_c}]",
                },
                description="Geographical distribution illustrating territorial concentration and whitespace.",
                configuration_notes="Use Treemap if Map visuals are disabled by organizational Power BI policy.",
                suggested_page="Regional Intelligence",
            )
        )

    # 5. Product Performance Matrix
    prod_cols = analysis.get_columns_by_role(SemanticRole.PRODUCT)
    if prod_cols:
        prod_c = prod_cols[0].name
        matrix_vals = []
        if rev_col:
            matrix_vals.append("[Total Revenue]")
        if profit_col:
            matrix_vals.append("[Total Profit]")
        if qty_col:
            matrix_vals.append("[Total Quantity Sold]")

        if not matrix_vals and analysis.primary_metric_column:
            matrix_vals.append(f"'{table_name}'[{analysis.primary_metric_column}]")

        recs.append(
            VisualRecommendation(
                title="Top Products & Items Matrix",
                visual_type="Matrix / Table Visual",
                fields={
                    "Rows": f"'{table_name}'[{prod_c}]",
                    "Values": matrix_vals,
                },
                description="Granular product performance table with sortable columns and conditional formatting.",
                configuration_notes="Apply conditional formatting data bars to Revenue and background color scales to Profit.",
                suggested_page="Product Analytics",
            )
        )

    # 6. Interactive Slicers Panel
    slicer_candidates = []
    if analysis.date_column:
        slicer_candidates.append(f"'{table_name}'[{analysis.date_column}] (Date Range Slider)")
    for c in (cat_cols + region_cols + analysis.get_columns_by_role(SemanticRole.GENERIC_CATEGORICAL))[:3]:
        slicer_candidates.append(f"'{table_name}'[{c.name}] (Dropdown / Tile)")

    if slicer_candidates:
        recs.append(
            VisualRecommendation(
                title="Interactive Global Slicers",
                visual_type="Slicer Controls",
                fields={"Filter Fields": slicer_candidates},
                description="Cross-filtering controls allowing users to isolate specific dates, segments, or regions.",
                configuration_notes="Dock vertically along the left sidebar or in an expandable Power BI bookmark pane.",
                suggested_page="Global (All Pages)",
            )
        )

    return recs


def generate_pbids_content(
    dataset_name: str,
    target_csv_filename: str = "prepared_dataset.csv",
) -> bytes:
    """Generate Microsoft Power BI Data Source (.pbids) JSON content."""
    pbids_doc = {
        "version": "0.1",
        "connections": [
            {
                "details": {
                    "path": target_csv_filename
                },
                "connectionType": "File",
                "options": {
                    "format": "Csv"
                },
                "mode": "Import"
            }
        ]
    }
    return json.dumps(pbids_doc, indent=2).encode("utf-8")


def generate_powerquery_m_script(
    df: pd.DataFrame,
    table_name: str = "PreparedData",
    csv_file_name: str = "prepared_dataset.csv",
) -> str:
    """Generate robust Power Query M code for seamless loading in Power BI Desktop."""
    # Build column type transformations
    type_mappings = []
    for col in df.columns:
        series = df[col]
        escaped_col = str(col).replace('"', '""')
        if pd.api.types.is_integer_dtype(series.dtype):
            type_mappings.append(f'{{"{escaped_col}", Int64.Type}}')
        elif pd.api.types.is_float_dtype(series.dtype):
            type_mappings.append(f'{{"{escaped_col}", type number}}')
        elif pd.api.types.is_datetime64_any_dtype(series.dtype):
            type_mappings.append(f'{{"{escaped_col}", type datetime}}')
        elif pd.api.types.is_bool_dtype(series.dtype):
            type_mappings.append(f'{{"{escaped_col}", type logical}}')
        else:
            type_mappings.append(f'{{"{escaped_col}", type text}}')

    types_str = ",\n        ".join(type_mappings)

    m_script = f"""// Power Query M Script for {table_name}
// Paste this code into Power BI Desktop: Home -> Transform Data -> Advanced Editor

let
    // 1. Ingest clean CSV dataset
    Source = Csv.Document(
        File.Contents("{csv_file_name}"),
        [
            Delimiter = ",",
            Columns = {len(df.columns)},
            Encoding = 65001, // UTF-8
            QuoteStyle = QuoteStyle.Csv
        ]
    ),

    // 2. Promote first row to headers
    #"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),

    // 3. Apply schema data types
    #"Changed Types" = Table.TransformColumnTypes(
        #"Promoted Headers",
        {{
        {types_str}
        }}
    )
in
    #"Changed Types"
"""
    return m_script


def generate_dax_script_content(measures: list[DaxMeasure]) -> str:
    """Generate formatted DAX measures script file content."""
    lines = [
        "/* ==========================================================================",
        "   POWER BI DAX MEASURES & BUSINESS LOGIC",
        "   Copy and paste these measures into Power BI Desktop (Modeling -> New Measure)",
        "   ========================================================================== */\n",
    ]

    current_folder = ""
    for measure in measures:
        if measure.display_folder != current_folder:
            current_folder = measure.display_folder
            lines.append(f"\n// --- [ Folder: {current_folder} ] ---")

        lines.append(f"// Description: {measure.description}")
        if measure.business_question:
            lines.append(f"// Business Insight: {measure.business_question}")
        lines.append(f"// Recommended Format: {measure.format_string}")
        lines.append(f"{measure.dax_formula}\n")

    return "\n".join(lines)


def generate_visual_guide_content(
    analysis: PowerBiModelAnalysis,
    table_name: str = "Dataset",
) -> str:
    """Generate markdown guide with visual recipes and step-by-step instructions."""
    lines = [
        f"# Microsoft Power BI Desktop - Interactive Report Design Guide",
        f"**Dataset**: `{analysis.dataset_name}`  |  **Records**: {analysis.row_count:,}  |  **Columns**: {analysis.column_count}\n",
        "---",
        "## Quick Start (3 Easy Steps to Open in Power BI Desktop)",
        "1. **Extract Bundle**: Unzip all files from the downloaded package into a folder on your computer.",
        "2. **Open Data in Power BI**:",
        "   - **Option A (Automatic)**: Double-click `power_bi_connection.pbids` to launch Power BI Desktop and load the dataset automatically.",
        "   - **Option B (Standard)**: Open Power BI Desktop -> Click **Get data** -> Choose **Text/CSV** -> Select `prepared_dataset.csv` -> Click **Load**.",
        "3. **Add DAX Measures**: In Power BI Desktop, click **Modeling** tab -> **New Measure**, and paste the measures from `dax_measures.dax`.\n",
        "---",
        "## Recommended Visual Layout Recipes\n",
    ]

    for idx, rec in enumerate(analysis.visual_recommendations, 1):
        lines.append(f"### {idx}. {rec.title}")
        lines.append(f"- **Visual Type in Power BI Pane**: `{rec.visual_type}`")
        lines.append(f"- **Report Page**: {rec.suggested_page}")
        lines.append(f"- **Purpose**: {rec.description}")
        lines.append("- **Field Wells Configuration**:")
        for well, field_val in rec.fields.items():
            if isinstance(field_val, list):
                lines.append(f"  - **{well}**: `{', '.join(field_val)}`")
            else:
                lines.append(f"  - **{well}**: `{field_val}`")
        lines.append(f"- **Formatting & Pro-Tips**: {rec.configuration_notes}\n")

    lines.extend([
        "---",
        "## Column Classification & Schema Reference\n",
        "| Column Name | Inferred Type | Detected Semantic Role | Confidence |",
        "| :--- | :--- | :--- | :--- |",
    ])

    for col in analysis.columns:
        lines.append(f"| `{col.name}` | {col.data_type} | {col.semantic_role.value} | {int(col.confidence * 100)}% |")

    lines.extend([
        "\n---",
        "## Data Quality & Operational Notes",
    ])
    if analysis.notes:
        for note in analysis.notes:
            lines.append(f"- {note}")
    else:
        lines.append("- Dataset schema is complete with all required numeric and dimensional columns.")

    return "\n".join(lines)


def build_powerbi_package(
    df: pd.DataFrame,
    dataset_name: str = "Dataset",
    table_name: str = "PreparedData",
) -> PowerBiPackage:
    """Build the complete in-memory Power BI integration package."""
    # 1. Analyze schema & detect roles
    analysis = analyze_dataset_for_powerbi(df, dataset_name=dataset_name)

    # 2. Generate DAX measures & visual recommendations
    analysis.dax_measures = generate_dax_measures(analysis, table_name=table_name)
    analysis.visual_recommendations = generate_visual_recommendations(analysis, table_name=table_name)

    # 3. Export clean CSV bytes (ISO dates, UTF-8 with BOM for Excel/PowerBI compatibility)
    clean_csv_bytes = df.to_csv(index=False).encode("utf-8-sig")

    # 4. Generate PBIDS connection file
    pbids_bytes = generate_pbids_content(dataset_name, target_csv_filename="prepared_dataset.csv")

    # 5. Generate Power Query M code
    powerquery_m_text = generate_powerquery_m_script(df, table_name=table_name, csv_file_name="prepared_dataset.csv")

    # 6. Generate DAX Script
    dax_script_text = generate_dax_script_content(analysis.dax_measures)

    # 7. Generate Visual Guide Markdown
    visual_guide_md_text = generate_visual_guide_content(analysis, table_name=table_name)

    # 8. Create unified ZIP archive
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prepared_dataset.csv", clean_csv_bytes)
        zf.writestr("power_bi_connection.pbids", pbids_bytes)
        zf.writestr("power_query_transformations.m", powerquery_m_text.encode("utf-8"))
        zf.writestr("dax_measures.dax", dax_script_text.encode("utf-8"))
        zf.writestr("dashboard_visual_guide.md", visual_guide_md_text.encode("utf-8"))

    zip_bytes = zip_buffer.getvalue()

    return PowerBiPackage(
        dataset_name=dataset_name,
        clean_csv_bytes=clean_csv_bytes,
        pbids_bytes=pbids_bytes,
        powerquery_m_text=powerquery_m_text,
        dax_script_text=dax_script_text,
        visual_guide_md_text=visual_guide_md_text,
        zip_bundle_bytes=zip_bytes,
    )
