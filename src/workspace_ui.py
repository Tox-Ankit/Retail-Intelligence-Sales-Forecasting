
"""Streamlit workspace for ingestion, profiling, quality, and cleaning."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st

from src.cleaning.cleaner import clean_dataset
from src.ingestion.loader import load_file
from src.ingestion.models import Dataset
from src.profiling.profiler import profile_dataset
from src.quality.checker import check_quality


def load_uploaded_files(uploaded_files: list) -> list[Dataset]:
    """Load uploaded CSV and Excel files using temporary disk storage."""

    if not uploaded_files:
        raise ValueError("Upload at least one CSV or Excel file.")

    loaded_datasets: list[Dataset] = []

    with TemporaryDirectory(prefix="retail_workspace_") as temporary_dir:
        temp_path = Path(temporary_dir)

        for index, uploaded_file in enumerate(uploaded_files):
            original_name = Path(uploaded_file.name).name

            if not original_name:
                raise ValueError("An uploaded file has an invalid filename.")

            destination = temp_path / f"{index}_{original_name}"
            destination.write_bytes(uploaded_file.getvalue())

            loaded_datasets.extend(load_file(destination))

    return loaded_datasets


def profiles_to_dataframe(profile) -> pd.DataFrame:
    """Convert column profile results into a displayable DataFrame."""

    rows = [
        {
            "Column": column.name,
            "Data Type": column.data_type,
            "Inferred Type": column.inferred_type,
            "Non-null": column.non_null_count,
            "Missing": column.null_count,
            "Missing %": round(column.null_percentage, 2),
            "Unique": column.unique_count,
            "Unique %": round(column.unique_percentage, 2),
            "Constant": column.is_constant,
            "Potential ID": column.is_potential_id,
            "Sample Values": ", ".join(
                str(value) for value in column.sample_values
            ),
        }
        for column in profile.column_profiles
    ]

    return pd.DataFrame(rows)


def quality_findings_to_dataframe(report) -> pd.DataFrame:
    """Convert quality findings into a displayable DataFrame."""

    rows = [
        {
            "Issue": finding.issue_type,
            "Severity": finding.severity,
            "Column": finding.column_name or "Dataset",
            "Message": finding.message,
            "Affected Rows": finding.affected_rows,
            "Affected %": round(finding.affected_percentage, 2),
            "Suggested Action": finding.suggested_action or "",
        }
        for finding in report.findings
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "Issue",
            "Severity",
            "Column",
            "Message",
            "Affected Rows",
            "Affected %",
            "Suggested Action",
        ],
    )


def cleaning_actions_to_dataframe(result) -> pd.DataFrame:
    """Convert the cleaning audit trail into a displayable DataFrame."""

    rows = [
        {
            "Action": action.action_type,
            "Column": action.column_name or "Dataset",
            "Description": action.description,
            "Affected Rows": action.affected_rows,
            "Reason": action.reason or "",
            "Parameters": str(action.parameters),
        }
        for action in result.actions
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "Action",
            "Column",
            "Description",
            "Affected Rows",
            "Reason",
            "Parameters",
        ],
    )


def render_data_workspace() -> None:
    """Render the data ingestion, profiling, quality, and cleaning workspace."""

    st.title("Data Workspace")
    st.write(
        "Upload datasets, inspect their structure, identify quality issues, "
        "and apply configurable cleaning operations."
    )

    uploaded_files = st.file_uploader(
        "Upload CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key="workspace_upload",
    )

    if st.button("Load uploaded datasets", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            try:
                datasets = load_uploaded_files(uploaded_files)
                st.session_state["workspace_datasets"] = datasets
                st.session_state["workspace_cleaning_results"] = {}
                st.success(f"Loaded {len(datasets)} dataset(s).")
            except Exception as exc:
                st.error(f"Could not load the uploaded files: {exc}")

    datasets = st.session_state.get("workspace_datasets", [])

    if not datasets:
        st.info("Upload files and select Load uploaded datasets to begin.")
        return

    st.subheader("Choose a dataset")

    selected_index = st.selectbox(
        "Dataset",
        options=list(range(len(datasets))),
        format_func=lambda index: (
            f"{datasets[index].name} — {datasets[index].source_file}"
        ),
        key="workspace_selected_dataset",
    )

    dataset = datasets[selected_index]
    data = dataset.data

    overview = st.columns(4)
    overview[0].metric("Rows", f"{len(data):,}")
    overview[1].metric("Columns", f"{len(data.columns):,}")
    overview[2].metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
    overview[3].metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")

    st.caption(
        f"Source: {dataset.source_file} | "
        f"Format: {dataset.source_type}"
        + (f" | Worksheet: {dataset.sheet_name}" if dataset.sheet_name else "")
    )

    tabs = st.tabs(
        ["Preview", "Profiling", "Data Quality", "Cleaning"]
    )

    with tabs[0]:
        st.subheader("Dataset preview")
        st.dataframe(data.head(100), use_container_width=True)

        st.download_button(
            "Download original dataset as CSV",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name=f"{dataset.name}_original.csv",
            mime="text/csv",
            key=f"download_original_{selected_index}",
        )

    with tabs[1]:
        st.subheader("Column profiling")

        try:
            profile = profile_dataset(dataset)

            metrics = st.columns(4)
            metrics[0].metric("Memory usage", f"{profile.memory_usage_bytes / 1024**2:.2f} MB")
            metrics[1].metric("Duplicate rows", profile.duplicate_rows)
            metrics[2].metric(
                "Columns with missing values",
                len(profile.columns_with_missing_values),
            )
            metrics[3].metric(
                "Potential ID columns",
                len(profile.potential_id_columns),
            )

            st.dataframe(
                profiles_to_dataframe(profile),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.error(f"Profiling failed: {exc}")

    with tabs[2]:
        st.subheader("Data quality report")

        try:
            report = check_quality(dataset)

            metrics = st.columns(4)
            metrics[0].metric("Total findings", report.total_findings)
            metrics[1].metric("Critical", len(report.critical_findings))
            metrics[2].metric("Warnings", len(report.warning_findings))
            metrics[3].metric("Informational", len(report.info_findings))

            findings = quality_findings_to_dataframe(report)

            if findings.empty:
                st.success("No issues were reported by the configured quality rules.")
            else:
                st.dataframe(findings, use_container_width=True, hide_index=True)

                st.download_button(
                    "Download quality report as CSV",
                    data=findings.to_csv(index=False).encode("utf-8"),
                    file_name=f"{dataset.name}_quality_report.csv",
                    mime="text/csv",
                    key=f"download_quality_{selected_index}",
                )
        except Exception as exc:
            st.error(f"Quality assessment failed: {exc}")

    with tabs[3]:
        st.subheader("Configure cleaning")

        st.info(
            "Cleaning starts from the original dataset each time. "
            "The original data is not overwritten."
        )

        apply_missing = st.checkbox(
            "Treat missing values",
            value=True,
            key=f"clean_missing_{selected_index}",
        )

        missing_strategy = st.selectbox(
            "Missing-value strategy",
            ["auto", "mean", "median", "mode", "constant"],
            key=f"clean_missing_strategy_{selected_index}",
            disabled=not apply_missing,
        )

        remove_duplicates = st.checkbox(
            "Remove duplicate rows",
            value=True,
            key=f"clean_duplicates_{selected_index}",
        )

        standardize_whitespace = st.checkbox(
            "Trim leading and trailing whitespace",
            value=True,
            key=f"clean_whitespace_{selected_index}",
        )

        normalize_case = st.checkbox(
            "Normalize categorical text to lowercase",
            value=False,
            key=f"clean_case_{selected_index}",
        )

        outlier_strategy = st.selectbox(
            "Outlier treatment",
            ["none", "cap_iqr", "remove_iqr", "remove_zscore"],
            format_func={
                "none": "No outlier treatment",
                "cap_iqr": "Cap outliers using IQR",
                "remove_iqr": "Remove outlier rows using IQR",
                "remove_zscore": "Remove outlier rows using Z-score",
            }.get,
            key=f"clean_outliers_{selected_index}",
        )

        numeric_columns = data.select_dtypes(include="number").columns.tolist()

        selected_outlier_columns = st.multiselect(
            "Columns for outlier treatment",
            options=numeric_columns,
            default=numeric_columns,
            disabled=outlier_strategy == "none",
            key=f"clean_outlier_columns_{selected_index}",
        )

        if st.button(
            "Apply cleaning",
            type="primary",
            key=f"apply_cleaning_{selected_index}",
        ):
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
                    outlier_columns=(
                        selected_outlier_columns
                        if outlier_strategy != "none"
                        else None
                    ),
                )

                cleaning_results = st.session_state.get(
                    "workspace_cleaning_results", {}
                )
                cleaning_results[selected_index] = result
                st.session_state["workspace_cleaning_results"] = cleaning_results

            except Exception as exc:
                st.error(f"Cleaning failed: {exc}")

        result = st.session_state.get(
            "workspace_cleaning_results", {}
        ).get(selected_index)

        if result is not None:
            st.subheader("Cleaning results")

            metrics = st.columns(4)
            metrics[0].metric("Actions recorded", result.total_actions)
            metrics[1].metric("Rows before", result.original_shape[0])
            metrics[2].metric("Rows after", result.cleaned_shape[0])
            metrics[3].metric("Rows removed", result.rows_removed)

            st.write("**Cleaned dataset preview**")
            st.dataframe(
                result.cleaned_data.head(100),
                use_container_width=True,
            )

            actions = cleaning_actions_to_dataframe(result)

            st.write("**Cleaning audit trail**")

            if actions.empty:
                st.info("No cleaning actions were recorded.")
            else:
                st.dataframe(
                    actions,
                    use_container_width=True,
                    hide_index=True,
                )

            st.download_button(
                "Download cleaned dataset as CSV",
                data=result.cleaned_data.to_csv(index=False).encode("utf-8"),
                file_name=f"{dataset.name}_cleaned.csv",
                mime="text/csv",
                key=f"download_cleaned_{selected_index}",
            )

            if not result.cleaned_data.isna().any().any():
                st.success("The cleaned dataset contains no missing values.")
            else:
                st.warning(
                    "Some missing values remain. Review the selected strategy "
                    "and the columns that could not be imputed."
                )
