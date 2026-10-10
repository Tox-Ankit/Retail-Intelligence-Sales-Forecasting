"""Dedicated page views for Data Upload, Understanding, Cleaning, Excel Report, PDF Report, and Settings."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.cleaning.cleaner import clean_dataset
from src.ingestion.loader import load_file
from src.ingestion.models import Dataset
from src.insights.engine import generate_business_insights
from src.profiling.profiler import profile_dataset
from src.quality.checker import check_quality
from src.reporting.excel_report import build_business_insights_excel
from src.reporting.pdf_report import build_business_insights_pdf
from src.ui.overview import _generate_synthetic_baseline
from src.ui.theme import (
    COLOR_ACCENT_TEAL,
    COLOR_BG_PRIMARY,
    COLOR_BORDER,
    COLOR_CARD,
    COLOR_TEXT_MUTED,
)
from src.workspace_ui import (
    cleaning_actions_to_dataframe,
    load_uploaded_files,
    profiles_to_dataframe,
    quality_findings_to_dataframe,
)


def _get_active_dataset_index() -> int:
    """Safely get selected dataset index."""
    datasets = st.session_state.get("workspace_datasets", [])
    if not datasets:
        return 0
    idx = st.session_state.get("workspace_selected_dataset", 0)
    return min(max(0, idx), len(datasets) - 1)


# =====================================================================
# 1. DATA UPLOAD PAGE
# =====================================================================
def render_upload_page() -> None:
    """Render the Data Upload page with drag-and-drop file ingestion."""

    st.title("Data Upload")
    st.markdown(
        "Upload your CSV or Excel files (`.csv`, `.xlsx`, `.xls`) to populate your analytics workspace."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Upload Files")
        uploaded_files = st.file_uploader(
            "Drag and drop business datasets here",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="page_upload_uploader",
        )

        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            if st.button("Load Uploaded Files", type="primary", use_container_width=True):
                if not uploaded_files:
                    st.warning("Please select at least one CSV or Excel file to upload.")
                else:
                    try:
                        datasets = load_uploaded_files(uploaded_files)
                        st.session_state["workspace_datasets"] = datasets
                        st.session_state["workspace_cleaning_results"] = {}
                        st.session_state["workspace_selected_dataset"] = 0
                        st.success(f"Successfully loaded {len(datasets)} dataset(s).")
                    except Exception as exc:
                        st.error(f"Error loading files: {exc}")

        with col_b2:
            if st.button("Load Sample Retail Dataset", use_container_width=True):
                sample_df = _generate_synthetic_baseline()
                sample_ds = Dataset(
                    name="Superstore_Retail_Sample",
                    source_file="sample_retail_data.csv",
                    data=sample_df,
                    source_type="csv",
                )
                st.session_state["workspace_datasets"] = [sample_ds]
                st.session_state["workspace_cleaning_results"] = {}
                st.session_state["workspace_selected_dataset"] = 0
                st.success("Loaded Sample Retail Intelligence Dataset with 486 transactions.")

    with col2:
        st.subheader("Active Session Datasets")
        datasets = st.session_state.get("workspace_datasets", [])
        if not datasets:
            st.info("No datasets loaded in memory.")
        else:
            for idx, ds in enumerate(datasets):
                with st.container(border=True):
                    st.markdown(f"**{idx + 1}. {ds.name}**")
                    st.caption(f"Source: `{ds.source_file}` ({ds.source_type})")
                    st.caption(f"Dimensions: {len(ds.data):,} rows × {len(ds.data.columns)} columns")

    st.divider()

    # Quick preview of currently active dataset
    datasets = st.session_state.get("workspace_datasets", [])
    if datasets:
        selected_idx = _get_active_dataset_index()
        selected_ds = datasets[selected_idx]
        st.subheader(f"Data Preview — {selected_ds.name}")
        st.dataframe(selected_ds.data.head(50), use_container_width=True)


# =====================================================================
# 2. DATA UNDERSTANDING (PROFILING & QUALITY)
# =====================================================================
def render_understanding_page() -> None:
    """Render the Data Understanding (profiling and quality checks) page."""

    st.title("Data Understanding & Quality Assessment")
    st.markdown(
        "Inspect inferred data types, completeness, cardinality, and quality audits."
    )

    datasets = st.session_state.get("workspace_datasets", [])
    if not datasets:
        st.info("Please upload a dataset first in **Data Upload**.")
        return

    dataset_labels = [f"{idx + 1}. {ds.name}" for idx, ds in enumerate(datasets)]
    selected_label = st.selectbox("Choose dataset to inspect", dataset_labels, key="understanding_dataset_select")
    selected_index = dataset_labels.index(selected_label)
    dataset = datasets[selected_index]
    data = dataset.data

    tabs = st.tabs(["Column Profiling & Types", "Data Quality Audit"])

    with tabs[0]:
        st.subheader("Detailed Column Profiles")
        try:
            profile = profile_dataset(dataset)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Memory Footprint", f"{profile.memory_usage_bytes / 1024**2:.2f} MB")
            m2.metric("Duplicate Rows", profile.duplicate_rows)
            m3.metric("Columns with Nulls", len(profile.columns_with_missing_values))
            m4.metric("Potential ID Keys", len(profile.potential_id_columns))

            df_profile = profiles_to_dataframe(profile)
            st.dataframe(df_profile, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Profiling error: {exc}")

    with tabs[1]:
        st.subheader("Automated Quality Rules & Anomalies")
        try:
            report = check_quality(dataset)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Findings", report.total_findings)
            m2.metric("Critical Errors", len(report.critical_findings))
            m3.metric("Warnings", len(report.warning_findings))
            m4.metric("Informational", len(report.info_findings))

            findings = quality_findings_to_dataframe(report)
            if findings.empty:
                st.success("No critical quality violations detected in this dataset.")
            else:
                st.dataframe(findings, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Quality check error: {exc}")


# =====================================================================
# 3. DATA CLEANING
# =====================================================================
def render_cleaning_page() -> None:
    """Render the configurable Data Cleaning workspace."""

    st.title("Data Cleaning & Transformation")
    st.markdown(
        "Apply audited cleaning operations without altering your original source files."
    )

    datasets = st.session_state.get("workspace_datasets", [])
    if not datasets:
        st.info("Please upload a dataset first in **Data Upload**.")
        return

    dataset_labels = [f"{idx + 1}. {ds.name}" for idx, ds in enumerate(datasets)]
    selected_label = st.selectbox("Choose dataset to clean", dataset_labels, key="cleaning_dataset_select")
    selected_index = dataset_labels.index(selected_label)
    dataset = datasets[selected_index]
    data = dataset.data

    c_col1, c_col2 = st.columns([1, 1])

    with c_col1:
        st.subheader("Configure Cleaning Rules")
        apply_missing = st.checkbox("Impute missing values", value=True, key=f"clean_p_missing_{selected_index}")
        missing_strategy = st.selectbox("Imputation strategy", ["auto", "mean", "median", "mode", "constant"], disabled=not apply_missing)
        remove_duplicates = st.checkbox("Remove duplicate rows", value=True, key=f"clean_p_dup_{selected_index}")
        standardize_whitespace = st.checkbox("Trim leading/trailing whitespace", value=True, key=f"clean_p_ws_{selected_index}")
        normalize_case = st.checkbox("Normalize text to lowercase", value=False, key=f"clean_p_case_{selected_index}")

    with c_col2:
        st.subheader("Outlier Treatment")
        outlier_strategy = st.selectbox(
            "Outlier strategy",
            ["none", "cap_iqr", "remove_iqr", "remove_zscore"],
            format_func={"none": "No treatment", "cap_iqr": "Cap outliers with IQR", "remove_iqr": "Remove rows with IQR", "remove_zscore": "Remove rows with Z-score"}.get,
        )
        numeric_columns = data.select_dtypes(include="number").columns.tolist()
        selected_outliers = st.multiselect("Columns for outlier handling", options=numeric_columns, default=numeric_columns, disabled=(outlier_strategy == "none"))

    if st.button("Apply Cleaning Pipeline", type="primary", use_container_width=True):
        try:
            result = clean_dataset(
                data=data,
                dataset_name=dataset.name,
                apply_missing_values=apply_missing,
                missing_strategy=missing_strategy,
                remove_duplicates=remove_duplicates,
                standardize_whitespace=standardize_whitespace,
                normalize_case=normalize_case,
                outlier_strategy=outlier_strategy,
                outlier_columns=selected_outliers if outlier_strategy != "none" else None,
            )
            cleaning_results = st.session_state.get("workspace_cleaning_results", {})
            cleaning_results[selected_index] = result
            st.session_state["workspace_cleaning_results"] = cleaning_results
            st.success("Cleaning pipeline executed successfully.")
        except Exception as exc:
            st.error(f"Cleaning failed: {exc}")

    result = st.session_state.get("workspace_cleaning_results", {}).get(selected_index)
    if result is not None:
        st.divider()
        st.subheader("Cleaned Dataset Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Actions Applied", result.total_actions)
        m2.metric("Rows Before", result.original_shape[0])
        m3.metric("Rows After", result.cleaned_shape[0])
        m4.metric("Rows Removed", result.rows_removed)

        st.dataframe(result.cleaned_data.head(50), use_container_width=True)

        st.download_button(
            "Download Cleaned CSV",
            data=result.cleaned_data.to_csv(index=False).encode("utf-8"),
            file_name=f"{dataset.name}_cleaned.csv",
            mime="text/csv",
        )


# =====================================================================
# 4. EXCEL REPORT PAGE
# =====================================================================
def render_excel_report_page() -> None:
    """Render the dedicated Excel BI report builder."""

    st.title("Excel Business Intelligence Report")
    st.markdown(
        "Generate a formatted Microsoft Excel (`.xlsx`) workbook with native OpenPyXL charts, "
        "executive summary cards, statistics, and business insights."
    )

    datasets = st.session_state.get("workspace_datasets", [])
    if not datasets:
        st.info("Please upload a dataset in **Data Upload** to generate an Excel report.")
        return

    dataset_labels = [f"{idx + 1}. {ds.name}" for idx, ds in enumerate(datasets)]
    selected_label = st.selectbox("Select dataset for Excel export", dataset_labels, key="excel_dataset_select")
    dataset_index = dataset_labels.index(selected_label)
    selected_dataset = datasets[dataset_index]

    cleaning_results = st.session_state.get("workspace_cleaning_results", {})
    cleaned_res = cleaning_results.get(dataset_index)

    version_opts = ["Original"]
    if cleaned_res is not None:
        version_opts.append("Cleaned")

    selected_ver = st.radio("Choose version", version_opts, horizontal=True)
    active_data = cleaned_res.cleaned_data if (selected_ver == "Cleaned" and cleaned_res) else selected_dataset.data

    with st.container(border=True):
        st.subheader("Excel Workbook Structure")
        st.write("The generated `.xlsx` workbook includes:")
        st.write("1. **Executive Dashboard**: Key KPIs and native bar/line charts.")
        st.write("2. **Descriptive Statistics**: Statistical summaries of all metrics.")
        st.write("3. **Category Breakdown**: Categorical aggregations and distribution tables.")
        st.write("4. **Business Insights**: AI-generated findings and recommendations.")
        st.write("5. **Cleaned Data Table**: Formatted raw data rows.")

        try:
            insights_report = generate_business_insights(active_data, dataset_name=selected_dataset.name)
            excel_bytes = build_business_insights_excel(
                insights_report,
                data=active_data,
                dataset_name=selected_dataset.name,
                data_label="Cleaned Data" if selected_ver == "Cleaned" else "Source Data",
            )

            st.download_button(
                label="Download Executive Excel Report (.xlsx)",
                data=excel_bytes,
                file_name=f"{selected_dataset.name}_BI_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Could not generate Excel report: {exc}")


# =====================================================================
# 5. PDF REPORT PAGE
# =====================================================================
def render_pdf_report_page() -> None:
    """Render the dedicated PDF report generator."""

    st.title("PDF Business Intelligence Report")
    st.markdown(
        "Export your business findings, evidence, and recommendations to a formatted PDF document."
    )

    datasets = st.session_state.get("workspace_datasets", [])
    if not datasets:
        st.info("Please upload a dataset in **Data Upload** to generate a PDF report.")
        return

    dataset_labels = [f"{idx + 1}. {ds.name}" for idx, ds in enumerate(datasets)]
    selected_label = st.selectbox("Select dataset for PDF export", dataset_labels, key="pdf_dataset_select")
    dataset_index = dataset_labels.index(selected_label)
    selected_dataset = datasets[dataset_index]

    cleaning_results = st.session_state.get("workspace_cleaning_results", {})
    cleaned_res = cleaning_results.get(dataset_index)
    active_data = cleaned_res.cleaned_data if cleaned_res else selected_dataset.data

    with st.container(border=True):
        st.subheader("PDF Report Specification")
        st.write("Includes formatted tables, executive summary, statistical findings, and severity badges.")

        try:
            report = generate_business_insights(active_data, dataset_name=selected_dataset.name)
            pdf_bytes = build_business_insights_pdf(report)

            st.download_button(
                label="Download PDF Report (.pdf)",
                data=pdf_bytes,
                file_name=f"{selected_dataset.name}_Insights_Report.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Could not generate PDF report: {exc}")


# =====================================================================
# 6. SETTINGS PAGE
# =====================================================================
def render_settings_page() -> None:
    """Render application preferences and cache controls."""

    st.title("Settings & Workspace Preferences")
    st.markdown("Configure local environment preferences, memory cache, and display formats.")

    col1, col2 = st.columns([1, 1])

    with col1:
        with st.container(border=True):
            st.subheader("Theme & Visuals")
            st.write("**Current Theme**: Dark Intelligence (SaaS Edition)")
            st.write("**Accent Palette**: Teal (`#19D9C5`) & Cyan (`#12BDF5`)")
            st.write("**Chart Engine**: Plotly Dark Pro")

    with col2:
        with st.container(border=True):
            st.subheader("Workspace Memory Cache")
            datasets = st.session_state.get("workspace_datasets", [])
            st.write(f"**Loaded Datasets in RAM**: {len(datasets)}")

            if st.button("Clear Session Datasets", type="primary"):
                st.session_state["workspace_datasets"] = []
                st.session_state["workspace_cleaning_results"] = {}
                st.session_state["workspace_selected_dataset"] = 0
                st.success("Session memory cleared successfully.")
