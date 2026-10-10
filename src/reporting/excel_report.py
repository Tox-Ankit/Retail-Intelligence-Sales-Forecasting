"""Generate an adaptive Excel business intelligence workbook."""

from __future__ import annotations

from io import BytesIO
import json
import re

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference, ScatterChart, Series
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from src.analytics.correlation import analyze_correlations
from src.analytics.descriptive import summarize_dataset
from src.insights.engine import InsightsReport


NAVY = "17365D"
BLUE = "4472C4"
LIGHT_BLUE = "D9EAF7"
GREEN = "E2F0D9"
AMBER = "FCE4D6"
WHITE = "FFFFFF"
DARK = "202A35"


# ---------------------------------------------------------------------
# General Excel helpers
# ---------------------------------------------------------------------

def _excel_value(value):
    """Convert values to Excel-compatible values."""
    if value is None:
        return None

    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, default=str)

    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and not np.isfinite(value):
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass

    return str(value)


def _safe_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare a copy of a DataFrame for safe Excel export."""
    result = data.copy()

    for column in result.columns:
        result[column] = result[column].map(_excel_value)

        # Prevent untrusted text from being interpreted as a formula.
        result[column] = result[column].map(
            lambda value: (
                "'" + value
                if isinstance(value, str)
                and value.startswith(("=", "+", "-", "@"))
                else value
            )
        )

    return result


def _write_table(
    worksheet,
    dataframe: pd.DataFrame,
    start_row: int = 1,
    start_col: int = 1,
    table_name: str | None = None,
) -> None:
    """Write a styled Excel table with filters and suitable column widths."""
    if dataframe is None:
        return

    dataframe = _safe_dataframe(dataframe)

    if len(dataframe.columns) == 0:
        return

    for offset, column in enumerate(dataframe.columns):
        cell = worksheet.cell(
            row=start_row,
            column=start_col + offset,
            value=str(column),
        )
        cell.font = Font(bold=True, color=WHITE)
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

    for row_offset, values in enumerate(
        dataframe.itertuples(index=False, name=None),
        start=1,
    ):
        for col_offset, value in enumerate(values):
            cell = worksheet.cell(
                row=start_row + row_offset,
                column=start_col + col_offset,
                value=value,
            )
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    worksheet.row_dimensions[start_row].height = 30

    last_row = start_row + len(dataframe)
    last_col = start_col + len(dataframe.columns) - 1

    if len(dataframe) > 0 and table_name:
        reference = (
            f"{get_column_letter(start_col)}{start_row}:"
            f"{get_column_letter(last_col)}{last_row}"
        )

        table = Table(displayName=table_name, ref=reference)
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        worksheet.add_table(table)

    worksheet.freeze_panes = f"A{start_row + 1}"

    for column_index in range(start_col, last_col + 1):
        letter = get_column_letter(column_index)

        values = [
            worksheet.cell(row=r, column=column_index).value
            for r in range(
                start_row,
                min(last_row, start_row + 100) + 1,
            )
        ]

        max_length = max(
            (len(str(value)) for value in values if value is not None),
            default=10,
        )

        worksheet.column_dimensions[letter].width = min(
            max(max_length + 2, 14),
            42,
        )


def _write_title(worksheet, title: str, subtitle: str) -> None:
    """Add a consistent report heading."""
    worksheet.merge_cells("A1:H1")
    worksheet["A1"] = title
    worksheet["A1"].font = Font(
        name="Aptos Display",
        size=21,
        bold=True,
        color=WHITE,
    )
    worksheet["A1"].fill = PatternFill("solid", fgColor=NAVY)
    worksheet["A1"].alignment = Alignment(
        vertical="center",
        horizontal="left",
    )
    worksheet.row_dimensions[1].height = 38

    worksheet.merge_cells("A2:H2")
    worksheet["A2"] = subtitle
    worksheet["A2"].font = Font(
        name="Aptos",
        size=10,
        italic=True,
        color=DARK,
    )
    worksheet["A2"].alignment = Alignment(
        vertical="center",
        wrap_text=True,
    )
    worksheet.row_dimensions[2].height = 28


# ---------------------------------------------------------------------
# Column detection and selection
# ---------------------------------------------------------------------

def _normalized_column_name(column: object) -> str:
    """Normalize column names for semantic matching."""
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(column).lower(),
    ).strip()


def _is_identifier_column(
    column: object,
    series: pd.Series,
) -> bool:
    """Identify likely identifier or reference-code columns."""
    name = _normalized_column_name(column)
    tokens = set(name.split())

    identifier_tokens = {
        "id",
        "identifier",
        "uuid",
        "guid",
        "sku",
        "upc",
        "ean",
    }

    if tokens & identifier_tokens:
        return True

    if "code" in tokens:
        return True

    if "number" in tokens and tokens & {
        "customer",
        "transaction",
        "invoice",
        "account",
        "order",
        "product",
        "employee",
    }:
        return True

    return False


def _numeric_column_score(column: object) -> int:
    """Rank numeric columns by likely business usefulness."""
    name = _normalized_column_name(column)

    high_value_metrics = (
        "total spent",
        "total sales",
        "total revenue",
        "sales revenue",
        "revenue",
        "sales",
        "turnover",
        "order value",
        "order amount",
        "amount",
    )

    profitability_metrics = (
        "profit",
        "gross margin",
        "net margin",
    )

    cost_metrics = (
        "cost",
        "expense",
        "expenditure",
    )

    quantity_metrics = (
        "quantity",
        "units sold",
        "units",
        "items sold",
    )

    average_metrics = (
        "price per unit",
        "unit price",
        "average price",
        "discount rate",
        "percentage",
        "percent",
        "rate",
        "margin",
        "score",
        "average",
    )

    if any(metric in name for metric in high_value_metrics):
        return 100

    if any(metric in name for metric in profitability_metrics):
        return 90

    if any(metric in name for metric in cost_metrics):
        return 85

    if any(metric in name for metric in quantity_metrics):
        return 80

    if any(metric in name for metric in average_metrics):
        return 60

    return 10


def _aggregation_for_measure(column: object) -> str:
    """Choose a basic aggregation based on the measure's name."""
    name = _normalized_column_name(column)

    average_metrics = (
        "price per unit",
        "unit price",
        "average price",
        "discount rate",
        "percentage",
        "percent",
        "rate",
        "margin",
        "score",
        "average",
        "temperature",
        "age",
    )

    if any(metric in name for metric in average_metrics):
        return "mean"

    return "sum"


def _get_numeric_columns(data: pd.DataFrame) -> list[str]:
    """Return ranked numeric columns suitable for charting."""
    columns = [
        str(column)
        for column in data.columns
        if pd.api.types.is_numeric_dtype(data[column])
        and not pd.api.types.is_bool_dtype(data[column])
        and not _is_identifier_column(column, data[column])
    ]

    return sorted(
        columns,
        key=lambda column: (
            _numeric_column_score(column),
            -data[column].isna().mean(),
        ),
        reverse=True,
    )


def _get_category_columns(data: pd.DataFrame) -> list[str]:
    """Return ranked categorical columns suitable for comparisons."""
    candidates = []

    preferred_terms = (
        "category",
        "region",
        "segment",
        "product",
        "store",
        "channel",
        "department",
        "gender",
        "payment method",
        "payment type",
        "status",
        "city",
        "country",
    )

    for column in data.columns:
        series = data[column]

        is_categorical = (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series)
        )

        if not is_categorical:
            continue

        if _is_identifier_column(column, series):
            continue

        unique_count = series.nunique(dropna=True)

        if not 2 <= unique_count <= 100:
            continue

        name = _normalized_column_name(column)

        semantic_score = (
            50 if any(term in name for term in preferred_terms) else 0
        )

        cardinality_score = (
            30 if unique_count <= 10
            else 20 if unique_count <= 20
            else 10
        )

        missing_penalty = int(10 * series.isna().mean())

        candidates.append(
            (str(column), semantic_score + cardinality_score - missing_penalty)
        )

    candidates.sort(key=lambda item: item[1], reverse=True)

    return [column for column, _ in candidates]


def _find_date_column(
    data: pd.DataFrame,
) -> tuple[str, pd.Series] | None:
    """Find the most plausible date/time column."""
    candidates = []

    for column in data.columns:
        series = data[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            candidates.append((column, series, 1.0))
            continue

        name = str(column).lower()
        suggests_date = any(
            token in name
            for token in ("date", "datetime", "timestamp", "time")
        )

        if not suggests_date:
            continue

        parsed = pd.to_datetime(series, errors="coerce")
        non_missing = int(series.notna().sum())

        if non_missing == 0:
            continue

        ratio = int(parsed.notna().sum()) / non_missing

        if ratio >= 0.7:
            candidates.append((column, parsed, ratio))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[2], reverse=True)

    column, parsed, _ = candidates[0]

    return str(column), parsed


# ---------------------------------------------------------------------
# Adaptive business KPIs
# ---------------------------------------------------------------------

def _calculate_business_kpis(
    data: pd.DataFrame,
) -> list[tuple[str, object]]:
    """Calculate potential business KPIs when suitable columns exist."""
    if data.empty:
        return []

    kpis = []
    numeric_columns = _get_numeric_columns(data)

    normalized_names = {
        str(column): _normalized_column_name(column)
        for column in data.columns
    }

    # Prefer explicit sales and revenue measures over generic amounts.
    sales_terms = (
        "total spent",
        "total sales",
        "total revenue",
        "sales revenue",
        "revenue",
        "sales",
        "turnover",
        "order value",
        "order amount",
        "amount",
    )

    sales_candidates = [
        column
        for column in numeric_columns
        if any(
            term in normalized_names[column]
            for term in sales_terms
        )
    ]

    sales_column = sales_candidates[0] if sales_candidates else None

    if sales_column is not None:
        values = pd.to_numeric(
            data[sales_column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if not values.empty:
            normalized = normalized_names[sales_column]

            if normalized.startswith("total "):
                total_label = sales_column
            else:
                total_label = f"Total {sales_column}"

            kpis.append(
                (total_label, float(values.sum()))
            )

            kpis.append(
                (f"Average {sales_column}", float(values.mean()))
            )

    # Find a quantity measure, if available.
    quantity_terms = (
        "quantity",
        "units sold",
        "items sold",
    )

    quantity_column = next(
        (
            column
            for column in numeric_columns
            if any(
                term in normalized_names[column]
                for term in quantity_terms
            )
        ),
        None,
    )

    if quantity_column is not None:
        values = pd.to_numeric(
            data[quantity_column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if not values.empty:
            kpis.append(
                (f"Total {quantity_column}", float(values.sum()))
            )

    # Find a customer identifier, if available.
    customer_column = next(
        (
            str(column)
            for column in data.columns
            if "customer" in _normalized_column_name(column)
            and _is_identifier_column(column, data[column])
        ),
        None,
    )

    if customer_column is not None:
        unique_customers = int(
            data[customer_column].nunique(dropna=True)
        )
        kpis.append(("Unique Customers", unique_customers))

    return kpis


# ---------------------------------------------------------------------
# Chart data
# ---------------------------------------------------------------------

def _make_chart_data(data: pd.DataFrame, worksheet) -> dict:
    """Create chart source tables using ranked columns."""
    sources = {}
    current_row = 4

    numeric_columns = _get_numeric_columns(data)
    category_columns = _get_category_columns(data)

    # Category comparison
    if category_columns and numeric_columns:
        category_column = category_columns[0]
        numeric_column = numeric_columns[0]
        aggregation = _aggregation_for_measure(numeric_column)

        numeric_values = pd.to_numeric(
            data[numeric_column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

        chart_data = pd.DataFrame(
            {
                "_category": data[category_column],
                "_measure": numeric_values,
            }
        ).dropna(subset=["_measure"])

        grouped = (
            chart_data.groupby("_category", dropna=False)["_measure"]
            .agg(aggregation)
            .sort_values(ascending=False)
            .head(10)
        )

        if len(grouped) >= 2:
            normalized_measure = _normalized_column_name(numeric_column)

            if aggregation == "mean":
                measure_label = f"Average {numeric_column}"
            elif normalized_measure.startswith("total "):
                measure_label = numeric_column
            else:
                measure_label = f"Total {numeric_column}"

            table = pd.DataFrame(
                {
                    str(category_column): [
                        str(value) for value in grouped.index
                    ],
                    measure_label: grouped.values,
                }
            )

            _write_table(
                worksheet,
                table,
                start_row=current_row,
                table_name="CategoryChartData",
            )

            sources["category"] = (
                current_row,
                len(table),
                table.columns.tolist(),
            )

            current_row += len(table) + 5

    # Time-series analysis
    date_info = _find_date_column(data)

    if date_info and numeric_columns:
        _, parsed_dates = date_info
        numeric_column = numeric_columns[0]
        aggregation = _aggregation_for_measure(numeric_column)

        trend_data = pd.DataFrame(
            {
                "Date": parsed_dates,
                numeric_column: pd.to_numeric(
                    data[numeric_column],
                    errors="coerce",
                ),
            }
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if not trend_data.empty:
            trend_data = (
                trend_data.groupby("Date", as_index=False)[numeric_column]
                .agg(aggregation)
                .sort_values("Date")
                .tail(500)
            )

        if len(trend_data) >= 3:
            _write_table(
                worksheet,
                trend_data,
                start_row=current_row,
                table_name="TrendChartData",
            )

            sources["trend"] = (
                current_row,
                len(trend_data),
                trend_data.columns.tolist(),
            )

            current_row += len(trend_data) + 5

    # Numeric distribution
    if numeric_columns:
        numeric_column = numeric_columns[0]

        values = pd.to_numeric(
            data[numeric_column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if len(values) >= 2 and values.nunique() >= 2:
            bin_count = min(10, max(2, int(np.sqrt(len(values)))))
            counts, edges = np.histogram(values, bins=bin_count)

            histogram = pd.DataFrame(
                {
                    f"{numeric_column} Range": [
                        f"{edges[i]:.3g} – {edges[i + 1]:.3g}"
                        for i in range(len(counts))
                    ],
                    "Frequency": counts,
                }
            )

            _write_table(
                worksheet,
                histogram,
                start_row=current_row,
                table_name="DistributionChartData",
            )

            sources["distribution"] = (
                current_row,
                len(histogram),
                histogram.columns.tolist(),
            )

            current_row += len(histogram) + 5

    # Relationship between two numeric measures
    if len(numeric_columns) >= 2:
        x_column, y_column = numeric_columns[:2]

        scatter_data = (
            data[[x_column, y_column]]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .head(500)
        )

        if len(scatter_data) >= 3:
            _write_table(
                worksheet,
                scatter_data,
                start_row=current_row,
                table_name="RelationshipChartData",
            )

            sources["scatter"] = (
                current_row,
                len(scatter_data),
                scatter_data.columns.tolist(),
            )

    return sources


# ---------------------------------------------------------------------
# Editable Excel charts
# ---------------------------------------------------------------------

def _add_charts(
    worksheet,
    source_sheet,
    sources: dict,
    positions: dict | None = None,
) -> None:
    """Add native, editable Excel charts."""
    if positions is None:
        positions = {
            "category": "A5",
            "trend": "J5",
            "distribution": "A22",
            "scatter": "J22",
        }

    if "category" in sources:
        start, count, columns = sources["category"]

        chart = BarChart()
        chart.type = "bar"
        chart.style = 10
        chart.title = f"{columns[1]} by {columns[0]}"
        chart.y_axis.title = columns[0]
        chart.x_axis.title = columns[1]
        chart.height = 8
        chart.width = 15
        chart.legend = None

        data_ref = Reference(
            source_sheet,
            min_col=2,
            min_row=start,
            max_row=start + count,
        )

        labels_ref = Reference(
            source_sheet,
            min_col=1,
            min_row=start + 1,
            max_row=start + count,
        )

        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(labels_ref)
        worksheet.add_chart(chart, positions["category"])

    if "trend" in sources:
        start, count, columns = sources["trend"]

        chart = LineChart()
        chart.style = 13
        chart.title = f"{columns[1]} over time"
        chart.x_axis.title = "Date"
        chart.y_axis.title = columns[1]
        chart.height = 8
        chart.width = 15

        data_ref = Reference(
            source_sheet,
            min_col=2,
            min_row=start,
            max_row=start + count,
        )

        labels_ref = Reference(
            source_sheet,
            min_col=1,
            min_row=start + 1,
            max_row=start + count,
        )

        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(labels_ref)
        worksheet.add_chart(chart, positions["trend"])

    if "distribution" in sources:
        start, count, columns = sources["distribution"]

        chart = BarChart()
        chart.style = 12
        chart.title = (
            f"Distribution of {columns[0].replace(' Range', '')}"
        )
        chart.x_axis.title = columns[0]
        chart.y_axis.title = "Frequency"
        chart.height = 8
        chart.width = 15
        chart.legend = None

        data_ref = Reference(
            source_sheet,
            min_col=2,
            min_row=start,
            max_row=start + count,
        )

        labels_ref = Reference(
            source_sheet,
            min_col=1,
            min_row=start + 1,
            max_row=start + count,
        )

        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(labels_ref)
        worksheet.add_chart(chart, positions["distribution"])

    if "scatter" in sources:
        start, count, columns = sources["scatter"]

        chart = ScatterChart()
        chart.title = f"{columns[1]} vs {columns[0]}"
        chart.x_axis.title = columns[0]
        chart.y_axis.title = columns[1]
        chart.height = 8
        chart.width = 15

        x_values = Reference(
            source_sheet,
            min_col=1,
            min_row=start + 1,
            max_row=start + count,
        )

        y_values = Reference(
            source_sheet,
            min_col=2,
            min_row=start + 1,
            max_row=start + count,
        )

        chart.series.append(
            Series(
                y_values,
                x_values,
                title=f"{columns[1]} vs {columns[0]}",
            )
        )

        worksheet.add_chart(chart, positions["scatter"])


# ---------------------------------------------------------------------
# Main workbook generator
# ---------------------------------------------------------------------

def build_business_insights_excel(
    report: InsightsReport,
    data: pd.DataFrame | None = None,
    forecast_result=None,
    dataset_name: str | None = None,
    data_label: str = "Source Data",
) -> bytes:
    """Build a multi-sheet business intelligence workbook in memory."""
    if not isinstance(report, InsightsReport):
        raise TypeError("report must be an InsightsReport instance.")

    if data is not None and not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame or None.")

    dataset_name = dataset_name or report.dataset_name

    workbook = Workbook()

    dashboard = workbook.active
    dashboard.title = "Executive Dashboard"

    statistics_sheet = workbook.create_sheet("Statistical Summary")
    visual_sheet = workbook.create_sheet("Visual Analysis")
    insights_sheet = workbook.create_sheet("Business Insights")

    data_sheet_name = (
        "Cleaned Data"
        if data_label.strip().lower() == "cleaned data"
        else "Source Data"
    )

    data_sheet = workbook.create_sheet(data_sheet_name)

    forecast_sheet = (
        workbook.create_sheet("Forecasting")
        if forecast_result is not None
        else None
    )

    data_rows = len(data) if data is not None else report.rows_analyzed

    data_columns = (
        len(data.columns) if data is not None
        else report.columns_analyzed
    )

    missing_cells = (
        int(data.isna().sum().sum()) if data is not None
        else None
    )

    numeric_columns = (
        _get_numeric_columns(data) if data is not None
        else []
    )

    warning_count = sum(
        finding.severity.lower() == "warning"
        for finding in report.insights
    )

    info_count = sum(
        finding.severity.lower() == "info"
        for finding in report.insights
    )

    # -----------------------------------------------------------------
    # Executive Dashboard
    # -----------------------------------------------------------------

    _write_title(
        dashboard,
        "RETAIL INTELLIGENCE",
        f"Executive Business Intelligence Report | {dataset_name}",
    )

    dashboard["A4"] = "Dataset"
    dashboard["B4"] = dataset_name

    dashboard["A5"] = "Rows analyzed"
    dashboard["B5"] = data_rows

    dashboard["C5"] = "Columns analyzed"
    dashboard["D5"] = data_columns

    dashboard["E5"] = "Business findings"
    dashboard["F5"] = report.total_insights

    dashboard["G5"] = "Warnings"
    dashboard["H5"] = warning_count

    for cell in dashboard[4][:8]:
        cell.fill = PatternFill("solid", fgColor=LIGHT_BLUE)
        cell.font = Font(bold=True, color=NAVY)

    for cell in dashboard[5][:8]:
        cell.font = Font(bold=True, color=DARK, size=12)

    if missing_cells is not None:
        dashboard["A7"] = "Missing cells"
        dashboard["B7"] = missing_cells

        dashboard["C7"] = "Numerical columns"
        dashboard["D7"] = len(numeric_columns)

        dashboard["E7"] = "Informational findings"
        dashboard["F7"] = info_count

    # Adaptive business KPIs
    dashboard["A9"] = "Business Performance KPIs"
    dashboard["A9"].font = Font(
        bold=True,
        size=14,
        color=NAVY,
    )

    business_kpis = (
        _calculate_business_kpis(data)
        if data is not None
        else []
    )

    if business_kpis:
        for index, (label, value) in enumerate(
            business_kpis,
            start=10,
        ):
            dashboard.cell(index, 1, label)
            dashboard.cell(index, 2, value)

            dashboard.cell(index, 1).font = Font(
                bold=True,
                color=NAVY,
            )

            dashboard.cell(index, 1).fill = PatternFill(
                "solid",
                fgColor=LIGHT_BLUE,
            )

            dashboard.cell(index, 2).font = Font(
                bold=True,
                color=DARK,
                size=12,
            )

            if isinstance(value, float):
                dashboard.cell(index, 2).number_format = "#,##0.00"

        highlights_start_row = 10 + len(business_kpis) + 2
    else:
        dashboard["A10"] = (
            "No confidently identifiable business KPIs were available."
        )
        highlights_start_row = 12

    # Key Business Highlights
    dashboard.cell(
        highlights_start_row,
        1,
        "Key Business Highlights",
    ).font = Font(
        bold=True,
        size=14,
        color=NAVY,
    )

    if report.insights:
        for row_number, finding in enumerate(
            report.insights[:8],
            start=highlights_start_row + 1,
        ):
            dashboard.cell(row_number, 1, finding.title)
            dashboard.cell(row_number, 2, finding.severity.upper())
            dashboard.cell(row_number, 3, finding.description)

            dashboard.cell(row_number, 1).font = Font(
                bold=True,
                color=NAVY,
            )

            dashboard.cell(row_number, 2).fill = PatternFill(
                "solid",
                fgColor=(
                    AMBER
                    if finding.severity.lower() == "warning"
                    else GREEN
                ),
            )

            dashboard.cell(row_number, 3).alignment = Alignment(
                wrap_text=True,
                vertical="top",
            )

            dashboard.row_dimensions[row_number].height = 32
    else:
        dashboard.cell(
            highlights_start_row + 1,
            1,
            "No findings were generated.",
        )

    # -----------------------------------------------------------------
    # Visual Analysis
    # -----------------------------------------------------------------

    _write_title(
        visual_sheet,
        "VISUAL ANALYSIS",
        "Editable Excel charts and their supporting data",
    )

    if data is not None and not data.empty:
        sources = _make_chart_data(data, visual_sheet)

        # Charts on the dashboard are placed below the highlights.
        dashboard_chart_start = max(
            35,
            highlights_start_row + min(len(report.insights), 8) + 5,
        )

        dashboard_positions = {
            "category": f"A{dashboard_chart_start}",
            "trend": f"J{dashboard_chart_start}",
            "distribution": f"A{dashboard_chart_start + 18}",
            "scatter": f"J{dashboard_chart_start + 18}",
        }

        _add_charts(
            dashboard,
            visual_sheet,
            sources,
            positions=dashboard_positions,
        )

        # Visual Analysis retains its own chart layout.
        _add_charts(visual_sheet, visual_sheet, sources)

        if len(numeric_columns) >= 2:
            correlation = analyze_correlations(data)
            matrix = pd.DataFrame(correlation.matrix).T

            if not matrix.empty:
                matrix.index.name = "Column"
                matrix = matrix.reset_index()

                correlation_start = max(
                    4,
                    visual_sheet.max_row + 3,
                )

                visual_sheet.cell(
                    correlation_start - 1,
                    1,
                    "Correlation Matrix",
                ).font = Font(
                    bold=True,
                    size=14,
                    color=NAVY,
                )

                _write_table(
                    visual_sheet,
                    matrix,
                    start_row=correlation_start,
                    table_name="CorrelationMatrix",
                )

                last_row = correlation_start + len(matrix)
                last_col = len(matrix.columns)

                if last_col >= 3 and last_row > correlation_start + 1:
                    visual_sheet.conditional_formatting.add(
                        f"B{correlation_start + 1}:"
                        f"{get_column_letter(last_col)}{last_row}",
                        ColorScaleRule(
                            start_type="num",
                            start_value=-1,
                            start_color="F8696B",
                            mid_type="num",
                            mid_value=0,
                            mid_color="FFEB84",
                            end_type="num",
                            end_value=1,
                            end_color="63BE7B",
                        ),
                    )
    else:
        visual_sheet["A4"] = (
            "No dataset was supplied for chart generation."
        )

    # -----------------------------------------------------------------
    # Statistical Summary
    # -----------------------------------------------------------------

    _write_title(
        statistics_sheet,
        "STATISTICAL SUMMARY",
        "Descriptive statistics calculated from the selected dataset",
    )

    if data is not None:
        summary = summarize_dataset(
            data,
            dataset_name=dataset_name,
        )

        numeric_records = [
            {
                "Column": item.column_name,
                "Count": item.count,
                "Missing": item.missing_count,
                "Mean": item.mean,
                "Median": item.median,
                "Mode": item.mode,
                "Minimum": item.minimum,
                "Maximum": item.maximum,
                "Standard Deviation": item.standard_deviation,
                "Variance": item.variance,
                "Q1": item.q1,
                "Q3": item.q3,
                "IQR": item.iqr,
                "Skewness": item.skewness,
                "Kurtosis": item.kurtosis,
                "Coefficient of Variation (%)": (
                    item.coefficient_of_variation
                ),
            }
            for item in summary.numerical_summaries.values()
        ]

        categorical_records = []

        for item in summary.categorical_summaries.values():
            column_name = str(item.column_name)
            normalized_name = _normalized_column_name(column_name)
            name_tokens = set(normalized_name.split())

            is_identifier = bool(
                name_tokens
                & {
                    "id",
                    "identifier",
                    "uuid",
                    "guid",
                    "sku",
                    "upc",
                    "ean",
                }
            ) or "code" in name_tokens

            if is_identifier:
                mode_display = "Not displayed (identifier column)"
                frequencies_display = "Not displayed (identifier column)"
                percentages_display = "Not displayed (identifier column)"
            else:
                if isinstance(item.mode, (list, tuple, set)):
                    modes = list(item.mode)
                elif item.mode is None:
                    modes = []
                else:
                    modes = [item.mode]

                mode_display = (
                    ", ".join(str(value) for value in modes[:5])
                    if modes
                    else "No mode"
                )

                if len(modes) > 5:
                    mode_display += f" (+{len(modes) - 5} more)"

                frequencies_display = item.frequencies
                percentages_display = item.percentages

            categorical_records.append(
                {
                    "Column": column_name,
                    "Count": item.count,
                    "Missing": item.missing_count,
                    "Unique Values": item.unique_count,
                    "Mode": mode_display,
                    "Top Category Frequencies": frequencies_display,
                    "Top Category Percentages": percentages_display,
                }
            )

        statistics_sheet["A4"] = "Numerical Statistics"
        statistics_sheet["A4"].font = Font(
            bold=True,
            size=14,
            color=NAVY,
        )

        numeric_df = pd.DataFrame(numeric_records)

        if numeric_df.empty:
            numeric_df = pd.DataFrame(
                columns=[
                    "Column",
                    "Count",
                    "Missing",
                    "Mean",
                    "Median",
                ]
            )

        _write_table(
            statistics_sheet,
            numeric_df,
            start_row=5,
            table_name="NumericalStatistics",
        )

        category_start = max(
            9,
            statistics_sheet.max_row + 3,
        )

        statistics_sheet.cell(
            category_start - 1,
            1,
            "Categorical Statistics",
        ).font = Font(
            bold=True,
            size=14,
            color=NAVY,
        )

        category_df = pd.DataFrame(categorical_records)

        if category_df.empty:
            category_df = pd.DataFrame(
                columns=[
                    "Column",
                    "Count",
                    "Missing",
                    "Unique Values",
                    "Mode",
                ]
            )

        _write_table(
            statistics_sheet,
            category_df,
            start_row=category_start,
            table_name="CategoricalStatistics",
        )
    else:
        statistics_sheet["A4"] = (
            "Dataset details were not supplied. "
            "Only report-level metrics are available."
        )

    # -----------------------------------------------------------------
    # Business Insights
    # -----------------------------------------------------------------

    _write_title(
        insights_sheet,
        "BUSINESS INSIGHTS",
        "Findings, evidence, and suggested next steps",
    )

    findings_df = report.to_dataframe()

    if findings_df.empty:
        findings_df = pd.DataFrame(
            columns=[
                "Category",
                "Severity",
                "Title",
                "Description",
                "Evidence",
                "Suggested next step",
            ]
        )

    _write_table(
        insights_sheet,
        findings_df,
        start_row=4,
        table_name="BusinessInsights",
    )

    # -----------------------------------------------------------------
    # Source or Cleaned Data
    # -----------------------------------------------------------------

    _write_title(
        data_sheet,
        data_sheet_name.upper(),
        f"Dataset: {dataset_name} | {data_rows:,} rows",
    )

    if data is not None:
        export_data = data.copy()
        export_data.columns = [
            str(column) for column in export_data.columns
        ]

        _write_table(
            data_sheet,
            export_data,
            start_row=4,
            table_name="DatasetRecords",
        )
    else:
        data_sheet["A4"] = "No dataset was supplied."

    # -----------------------------------------------------------------
    # Optional Forecasting
    # -----------------------------------------------------------------

    if forecast_sheet is not None:
        _write_title(
            forecast_sheet,
            "FORECASTING RESULTS",
            "Results from the supplied forecasting workflow",
        )

        forecast = getattr(forecast_result, "forecast", None)
        comparison = getattr(forecast_result, "model_comparison", None)
        preparation = getattr(forecast_result, "preparation", None)

        forecast_sheet["A4"] = "Selected Model"
        forecast_sheet["B4"] = getattr(
            forecast_result,
            "selected_model_name",
            "Unavailable",
        )

        if isinstance(forecast, pd.Series):
            forecast_df = (
                forecast.rename("Forecast")
                .rename_axis("Period")
                .reset_index()
            )
        elif isinstance(forecast, pd.DataFrame):
            forecast_df = forecast.copy()
        else:
            forecast_df = pd.DataFrame(
                columns=["Period", "Forecast"]
            )

        forecast_sheet["A6"] = "Future Forecast"
        forecast_sheet["A6"].font = Font(
            bold=True,
            size=14,
            color=NAVY,
        )

        _write_table(
            forecast_sheet,
            forecast_df,
            start_row=7,
            table_name="ForecastValues",
        )

        if isinstance(comparison, pd.DataFrame):
            comparison_start = max(
                12,
                forecast_sheet.max_row + 3,
            )

            forecast_sheet.cell(
                comparison_start - 1,
                1,
                "Model Comparison",
            ).font = Font(
                bold=True,
                size=14,
                color=NAVY,
            )

            _write_table(
                forecast_sheet,
                comparison,
                start_row=comparison_start,
                table_name="ForecastModelComparison",
            )

        if preparation is not None:
            details = {
                "Input rows": getattr(
                    preparation, "input_rows", None
                ),
                "Valid rows": getattr(
                    preparation, "valid_rows", None
                ),
                "Dropped rows": getattr(
                    preparation, "dropped_rows", None
                ),
                "Missing periods": getattr(
                    preparation, "missing_periods", None
                ),
                "Filled periods": getattr(
                    preparation, "filled_periods", None
                ),
                "Frequency": getattr(
                    preparation, "frequency", None
                ),
                "Aggregation": getattr(
                    preparation, "aggregation", None
                ),
            }

            details_start = max(
                12,
                forecast_sheet.max_row + 3,
            )

            details_df = pd.DataFrame(
                [
                    {"Metric": key, "Value": value}
                    for key, value in details.items()
                ]
            )

            forecast_sheet.cell(
                details_start - 1,
                1,
                "Preparation Details",
            ).font = Font(
                bold=True,
                size=14,
                color=NAVY,
            )

            _write_table(
                forecast_sheet,
                details_df,
                start_row=details_start,
                table_name="ForecastPreparationDetails",
            )

    # -----------------------------------------------------------------
    # Final workbook styling
    # -----------------------------------------------------------------

    for worksheet in workbook.worksheets:
        worksheet.sheet_view.showGridLines = False

        if worksheet.title in {
            "Executive Dashboard",
            "Statistical Summary",
            "Visual Analysis",
        }:
            worksheet.sheet_properties.pageSetUpPr.fitToPage = True
            worksheet.page_setup.fitToWidth = 1
            worksheet.page_setup.fitToHeight = 0
            worksheet.page_setup.orientation = "landscape"

    dashboard.freeze_panes = "A4"
    dashboard.sheet_view.zoomScale = 85

    dashboard.sheet_properties.tabColor = NAVY
    statistics_sheet.sheet_properties.tabColor = BLUE
    visual_sheet.sheet_properties.tabColor = "70AD47"
    insights_sheet.sheet_properties.tabColor = "ED7D31"
    data_sheet.sheet_properties.tabColor = "A5A5A5"

    for column, width in {
        "A": 30,
        "B": 24,
        "C": 28,
        "D": 18,
        "E": 24,
        "F": 18,
        "G": 18,
        "H": 18,
    }.items():
        dashboard.column_dimensions[column].width = width

    buffer = BytesIO()
    workbook.save(buffer)

    return buffer.getvalue()