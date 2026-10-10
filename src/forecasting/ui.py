
"""Streamlit interface for retail forecasting."""

import pandas as pd
import streamlit as st

from src.forecasting.detection import detect_forecasting_columns
from src.forecasting.workflow import run_forecasting_workflow


SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


def get_excel_sheet_names(uploaded_file) -> list[str]:
    """Return sheet names from an uploaded Excel workbook."""

    uploaded_file.seek(0)
    workbook = pd.ExcelFile(uploaded_file)
    return workbook.sheet_names


def load_uploaded_data(uploaded_file, sheet_name: str | None = None) -> pd.DataFrame:
    """Load a CSV or a selected Excel worksheet."""

    if uploaded_file is None:
        raise ValueError("Upload a CSV or Excel file first.")

    filename = getattr(uploaded_file, "name", "")
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only CSV and XLSX files are supported.")

    uploaded_file.seek(0)

    if extension == ".csv":
        return pd.read_csv(uploaded_file)

    workbook = pd.ExcelFile(uploaded_file)

    if sheet_name is None:
        if len(workbook.sheet_names) != 1:
            raise ValueError(
                "Choose a worksheet when the workbook contains multiple sheets."
            )
        sheet_name = workbook.sheet_names[0]

    if sheet_name not in workbook.sheet_names:
        raise ValueError(f"Worksheet '{sheet_name}' was not found.")

    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file, sheet_name=sheet_name)


def render_forecasting_ui() -> None:
    """Render the complete forecasting interface."""

    st.header("Retail Sales Forecasting")
    st.write(
        "Upload historical data, review the detected columns, "
        "configure your forecast, and compare model performance."
    )

    uploaded_file = st.file_uploader(
        "Upload your historical data",
        type=["csv", "xlsx"],
        key="forecasting_upload",
    )

    if uploaded_file is None:
        st.info("Upload a CSV or XLSX file to begin.")
        return

    try:
        extension = uploaded_file.name.rsplit(".", 1)[-1].lower()

        if extension == "xlsx":
            sheet_names = get_excel_sheet_names(uploaded_file)

            if len(sheet_names) > 1:
                sheet_name = st.selectbox(
                    "Select worksheet",
                    options=sheet_names,
                    key="forecasting_sheet",
                )
            else:
                sheet_name = sheet_names[0]

            data = load_uploaded_data(uploaded_file, sheet_name)

        else:
            data = load_uploaded_data(uploaded_file)

    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        return

    if data.empty:
        st.error("The uploaded dataset is empty.")
        return

    if not data.columns.is_unique:
        st.error("The dataset contains duplicate column names.")
        return

    st.subheader("Dataset overview")
    st.write(f"Rows: {len(data):,} | Columns: {len(data.columns):,}")
    st.dataframe(data.head(10), use_container_width=True)

    try:
        detection = detect_forecasting_columns(data)
    except (TypeError, ValueError) as exc:
        st.error(f"Could not inspect the dataset: {exc}")
        return

    with st.expander("Automatic detection suggestions", expanded=True):
        if detection.date_candidates:
            st.write("**Possible date columns**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Column": item.column,
                            "Confidence score": item.confidence,
                            "Reason": item.reason,
                        }
                        for item in detection.date_candidates
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("No likely date column was detected.")

        if detection.target_candidates:
            st.write("**Possible numerical targets**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Column": item.column,
                            "Confidence score": item.confidence,
                            "Reason": item.reason,
                        }
                        for item in detection.target_candidates
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("No suitable numerical target was detected.")

    all_columns = list(data.columns)
    suggested_dates = [item.column for item in detection.date_candidates]
    numeric_targets = [
        column
        for column in all_columns
        if pd.api.types.is_numeric_dtype(data[column].dtype)
        and not pd.api.types.is_bool_dtype(data[column].dtype)
    ]

    if not numeric_targets:
        st.error(
            "No numeric target columns are available. "
            "Clean or convert the desired target column first."
        )
        return

    st.subheader("Forecast configuration")

    date_default = (
        all_columns.index(suggested_dates[0])
        if suggested_dates and suggested_dates[0] in all_columns
        else 0
    )

    date_column = st.selectbox(
        "Date column",
        options=all_columns,
        index=date_default,
        key="forecasting_date_column",
    )

    target_candidates = [
        item.column
        for item in detection.target_candidates
        if item.column in numeric_targets
    ]

    target_default = (
        numeric_targets.index(target_candidates[0])
        if target_candidates and target_candidates[0] in numeric_targets
        else 0
    )

    value_column = st.selectbox(
        "Numerical target to forecast",
        options=numeric_targets,
        index=target_default,
        key="forecasting_target_column",
    )

    frequency_options = ["D", "W", "ME", "QE", "YE"]
    suggested_frequency = detection.suggested_frequency

    frequency_default = (
        frequency_options.index(suggested_frequency)
        if suggested_frequency in frequency_options
        else frequency_options.index("ME")
    )

    frequency = st.selectbox(
        "Forecast frequency",
        options=frequency_options,
        index=frequency_default,
        format_func={
            "D": "Daily",
            "W": "Weekly",
            "ME": "Monthly",
            "QE": "Quarterly",
            "YE": "Yearly",
        }.get,
        key="forecasting_frequency",
    )

    aggregation = st.selectbox(
        "Aggregation when multiple observations occur in a period",
        options=["sum", "mean", "count"],
        format_func={
            "sum": "Sum",
            "mean": "Mean",
            "count": "Count of valid observations",
        }.get,
        key="forecasting_aggregation",
    )

    missing_strategy = st.selectbox(
        "How should missing periods be handled?",
        options=["none", "zero", "ffill", "interpolate"],
        format_func={
            "none": "Keep missing periods and stop if any remain",
            "zero": "Fill missing periods with zero",
            "ffill": "Use the previous observed value",
            "interpolate": "Interpolate between observations",
        }.get,
        key="forecasting_missing_strategy",
    )

    horizon_defaults = {"D": 7, "W": 4, "ME": 3, "QE": 1, "YE": 1}
    horizon_maximum = {"D": 365, "W": 104, "ME": 60, "QE": 40, "YE": 20}

    forecast_horizon = st.number_input(
        "Number of future periods",
        min_value=1,
        max_value=horizon_maximum[frequency],
        value=horizon_defaults[frequency],
        step=1,
        key="forecasting_horizon",
    )

    include_seasonal = st.checkbox(
        "Include seasonal forecasting models",
        value=False,
        help=(
            "Seasonal models require enough historical data. "
            "You can configure the seasonal period below."
        ),
        key="forecasting_include_seasonal",
    )

    default_seasons = {"D": 7, "W": 52, "ME": 12, "QE": 4, "YE": 2}
    season_length = None

    if include_seasonal:
        season_length = st.number_input(
            "Observations per season",
            min_value=2,
            max_value=366,
            value=default_seasons[frequency],
            step=1,
            key="forecasting_season_length",
        )

    st.caption(
        "These are initial settings. Verify the selected frequency and "
        "seasonal period against the business reporting cycle."
    )

    run_clicked = st.button(
        "Run forecasting",
        type="primary",
        use_container_width=True,
        key="run_forecasting",
    )

    if not run_clicked:
        return

    with st.spinner("Preparing data, evaluating models, and generating forecasts..."):
        try:
            result = run_forecasting_workflow(
                data=data,
                date_column=date_column,
                value_column=value_column,
                frequency=frequency,
                aggregation=aggregation,
                missing_period_strategy=missing_strategy,
                forecast_horizon=int(forecast_horizon),
                season_length=(
                    int(season_length) if season_length is not None else None
                ),
                include_seasonal=include_seasonal,
            )
        except Exception as exc:
            st.error(f"Forecasting could not be completed: {exc}")
            return

    st.success("Forecast generated successfully.")

    st.subheader("Forecast summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Selected model", result.selected_model_name)
    col2.metric("Forecast periods", len(result.forecast))
    col3.metric(
        "Backtesting mean RMSE",
        f"{result.model_comparison.iloc[0]['Mean RMSE']:,.3f}"
        if "Mean RMSE" in result.model_comparison.columns
        else "Unavailable",
    )

    st.subheader("Forecast chart")

    historical = result.preparation.series.rename("Historical")
    future = result.forecast.rename("Forecast")
    chart_data = pd.concat([historical, future], axis=1)
    st.line_chart(chart_data, y=["Historical", "Forecast"])

    st.subheader("Future predictions")
    forecast_table = result.forecast_dataframe()
    st.dataframe(forecast_table, use_container_width=True)

    st.download_button(
        "Download forecast as CSV",
        data=forecast_table.to_csv(index=False).encode("utf-8"),
        file_name="retail_sales_forecast.csv",
        mime="text/csv",
        key="download_forecast_csv",
    )

    st.subheader("Model comparison")
    st.dataframe(
        result.model_comparison,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Preparation details and notes"):
        st.write(
            {
                "Input rows": result.preparation.input_rows,
                "Valid rows": result.preparation.valid_rows,
                "Dropped rows": result.preparation.dropped_rows,
                "Missing periods found": result.preparation.missing_periods,
                "Periods filled": result.preparation.filled_periods,
                "Frequency": result.preparation.frequency,
                "Aggregation": result.preparation.aggregation,
            }
        )

        for note in result.notes:
            st.write(f"- {note}")
