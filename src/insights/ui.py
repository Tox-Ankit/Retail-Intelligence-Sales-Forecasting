"""Streamlit interface for automated business insights."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.insights.engine import (
    InsightsReport,
    generate_business_insights,
)
from src.reporting.pdf_report import build_business_insights_pdf
from src.reporting.excel_report import build_business_insights_excel

def filter_insights_dataframe(
    report: InsightsReport,
    severities: list[str] | None = None,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """Return a filtered copy of the report's findings as a DataFrame."""

    findings = report.to_dataframe()

    if severities is not None:
        findings = findings[findings["Severity"].isin(severities)]

    if categories is not None:
        findings = findings[findings["Category"].isin(categories)]

    return findings.reset_index(drop=True)


def _get_dataset_name(dataset: object, index: int) -> str:
    """Get a readable name from a workspace dataset."""

    name = getattr(dataset, "name", None)
    return str(name) if name else f"Dataset {index + 1}"


def render_business_insights_page() -> None:
    """Render the Business Insights page using datasets in the Data Workspace."""

    st.title("Automated Business Insights")
    st.markdown(
        "Generate evidence-based findings about data quality, distributions, "
        "categories, unusual values, and relationships in your dataset."
    )

    datasets = st.session_state.get("workspace_datasets", [])
    if not datasets:
        st.info(
            "No datasets are available yet. Open **Data Workspace** and upload "
            "a CSV or Excel file first."
        )
        return

    dataset_labels = [
        f"{index + 1}. {_get_dataset_name(dataset, index)}"
        for index, dataset in enumerate(datasets)
    ]
    selected_label = st.selectbox(
        "Choose a dataset",
        options=dataset_labels,
        key="business_insights_dataset",
    )
    dataset_index = dataset_labels.index(selected_label)
    selected_dataset = datasets[dataset_index]

    original_data = getattr(selected_dataset, "data", None)
    if not isinstance(original_data, pd.DataFrame):
        st.error("The selected workspace item does not contain a valid DataFrame.")
        return

    cleaning_results = st.session_state.get("workspace_cleaning_results", {})
    cleaning_result = cleaning_results.get(dataset_index)

    version_options = ["Original"]
    if cleaning_result is not None:
        version_options.append("Cleaned")

    selected_version = st.radio(
        "Analyze which version?",
        options=version_options,
        horizontal=True,
        key=f"business_insights_version_{dataset_index}",
    )

    analysis_data = original_data
    if selected_version == "Cleaned" and cleaning_result is not None:
        analysis_data = cleaning_result.cleaned_data

    st.caption(
        f"Selected data: {analysis_data.shape[0]:,} rows × "
        f"{analysis_data.shape[1]:,} columns"
    )

    if analysis_data.empty:
        st.warning("This dataset is empty. Select another dataset or clean/reload it.")
        return

    if st.button(
        "Generate business insights",
        type="primary",
        key=f"generate_business_insights_{dataset_index}_{selected_version}",
    ):
        try:
            report = generate_business_insights(
                analysis_data,
                dataset_name=_get_dataset_name(selected_dataset, dataset_index),
            )
            st.session_state["business_insights_report"] = report
            st.session_state["business_insights_source"] = (
                dataset_index,
                selected_version,
            )
        except (TypeError, ValueError) as exc:
            st.error(f"Could not generate insights: {exc}")
            return
        except Exception as exc:
            st.error(f"An unexpected error occurred while generating insights: {exc}")
            return

    report = st.session_state.get("business_insights_report")
    report_source = st.session_state.get("business_insights_source")
    if report is None or report_source != (dataset_index, selected_version):
        st.info("Select **Generate business insights** to analyze this dataset.")
        return

    st.divider()
    st.subheader(f"Insights for {_get_dataset_name(selected_dataset, dataset_index)}")

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Total findings", report.total_insights)
    metric2.metric("Rows analyzed", f"{report.rows_analyzed:,}")
    metric3.metric("Columns analyzed", f"{report.columns_analyzed:,}")

    report_df = report.to_dataframe()
    if report_df.empty:
        st.info("The analysis did not produce any findings.")
        return

    available_severities = [
        severity
        for severity in ["warning", "info"]
        if severity in report_df["Severity"].tolist()
    ]
    available_categories = sorted(report_df["Category"].dropna().unique().tolist())

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_severities = st.multiselect(
            "Filter by severity",
            options=available_severities,
            default=available_severities,
            key="business_insights_severity_filter",
        )
    with filter_col2:
        selected_categories = st.multiselect(
            "Filter by category",
            options=available_categories,
            default=available_categories,
            key="business_insights_category_filter",
        )

    filtered_df = filter_insights_dataframe(
        report,
        severities=selected_severities,
        categories=selected_categories,
    )

    st.caption(f"Showing {len(filtered_df)} of {report.total_insights} findings.")

    if filtered_df.empty:
        st.info("No findings match the selected filters.")
    else:
        for finding in report.insights:
            if finding.severity not in selected_severities:
                continue
            if finding.category not in selected_categories:
                continue

            with st.container(border=True):
                st.markdown(f"### {finding.title}")
                st.markdown(
                    f"**Category:** {finding.category}  ·  "
                    f"**Severity:** {finding.severity}"
                )
                st.write(finding.description)

                if finding.evidence:
                    with st.expander("View supporting evidence"):
                        st.json(finding.evidence)

                if finding.recommendation:
                    st.markdown(f"**Suggested next step:** {finding.recommendation}")

    st.download_button(
        "Download filtered insights (CSV)",
        data=filtered_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="business_insights.csv",
        mime="text/csv",
        key="download_business_insights_csv",
    )

    st.divider()
    st.subheader("Download Full Analysis Report")

    st.write(
        "Export all findings, including supporting evidence and "
        "suggested next steps, into a formatted PDF report."
    )

    try:
        pdf_bytes = build_business_insights_pdf(report)

        st.download_button(
            label="Download Business Insights Report (PDF)",
            data=pdf_bytes,
            file_name="business_insights_report.pdf",
            mime="application/pdf",
            key="download_business_insights_pdf",
        )

    except Exception as exc:
        st.error(f"Could not generate the PDF report: {exc}")

    st.divider()
    st.subheader("Download Excel Business Intelligence Workbook")

    st.write(
        "Download a formatted Excel workbook containing an executive "
        "dashboard, statistical summaries, editable charts, business "
        "insights, and the selected dataset."
    )

    try:
        excel_bytes = build_business_insights_excel(
            report,
            data=analysis_data,
            dataset_name=_get_dataset_name(
                selected_dataset,
                dataset_index,
            ),
            data_label=(
                "Cleaned Data"
                if selected_version == "Cleaned"
                else "Source Data"
            ),
        )

        st.download_button(
            label="Download Excel Report (.xlsx)",
            data=excel_bytes,
            file_name="business_intelligence_report.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            key="download_business_insights_excel",
        )

    except Exception as exc:
        st.error(f"Could not generate the Excel report: {exc}")