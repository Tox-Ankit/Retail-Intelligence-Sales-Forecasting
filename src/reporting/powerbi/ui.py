"""Streamlit user interface for Microsoft Power BI Desktop integration."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.reporting.powerbi.detector import analyze_dataset_for_powerbi
from src.reporting.powerbi.generator import (
    build_powerbi_package,
    generate_dax_measures,
    generate_visual_recommendations,
)


def _get_dataset_name(dataset: object, index: int) -> str:
    """Safely obtain dataset display name."""
    name = getattr(dataset, "name", None)
    return str(name) if name else f"Dataset {index + 1}"


def render_powerbi_page() -> None:
    """Render the dedicated Power BI Desktop integration workspace."""

    st.title("Power BI Dashboard Integration")
    st.markdown(
        """
        Prepare and bridge your datasets into **Microsoft Power BI Desktop**.
        This workspace analyzes your data schema, detects business metrics and dimensions,
        generates optimized DAX formulas, crafts visual recipes, and exports verified
        Power BI data bundles.
        """
    )

    # 1. Dataset Selection
    datasets = st.session_state.get("workspace_datasets", [])
    if not datasets:
        st.info(
            "No datasets are loaded in the current session. "
            "Please go to **Data Workspace**, upload your CSV or Excel file, "
            "and load your data."
        )
        return

    dataset_labels = [
        f"{idx + 1}. {_get_dataset_name(ds, idx)} ({getattr(ds, 'source_file', 'file')})"
        for idx, ds in enumerate(datasets)
    ]

    col_select, col_version = st.columns([2, 1])

    with col_select:
        selected_label = st.selectbox(
            "Select Source Dataset for Power BI",
            options=dataset_labels,
            key="pbi_selected_dataset",
        )
        dataset_index = dataset_labels.index(selected_label)
        selected_dataset = datasets[dataset_index]

    original_data = getattr(selected_dataset, "data", None)
    if not isinstance(original_data, pd.DataFrame):
        st.error("The selected workspace item does not contain a valid DataFrame.")
        return

    cleaning_results = st.session_state.get("workspace_cleaning_results", {})
    cleaning_result = cleaning_results.get(dataset_index)

    with col_version:
        version_options = ["Original"]
        if cleaning_result is not None:
            version_options.append("Cleaned")

        selected_version = st.radio(
            "Data Version",
            options=version_options,
            horizontal=True,
            key=f"pbi_version_radio_{dataset_index}",
        )

    active_df = original_data
    if selected_version == "Cleaned" and cleaning_result is not None:
        active_df = cleaning_result.cleaned_data

    # Overview Metrics
    overview_cols = st.columns(4)
    overview_cols[0].metric("Rows Ready", f"{len(active_df):,}")
    overview_cols[1].metric("Columns", f"{len(active_df.columns):,}")
    overview_cols[2].metric("Missing Values", f"{int(active_df.isna().sum().sum()):,}")
    overview_cols[3].metric("Data Version", f"{selected_version}")

    if active_df.empty:
        st.warning("The selected dataset contains 0 rows. Please select another dataset or clean/re-upload it.")
        return

    st.divider()

    # 2. Build Power BI Package
    dataset_name = _get_dataset_name(selected_dataset, dataset_index)
    with st.spinner("Analyzing schema and generating Power BI artifacts..."):
        try:
            analysis = analyze_dataset_for_powerbi(active_df, dataset_name=dataset_name)
            analysis.dax_measures = generate_dax_measures(analysis, table_name="PreparedData")
            analysis.visual_recommendations = generate_visual_recommendations(analysis, table_name="PreparedData")
            pkg = build_powerbi_package(active_df, dataset_name=dataset_name, table_name="PreparedData")
        except Exception as exc:
            st.error(f"Failed to prepare Power BI bundle: {exc}")
            return

    # 3. Quick Action - Download All-in-One Package
    st.subheader("1. Download Power BI Package")
    st.write(
        "Download the complete all-in-one integration archive containing the prepared CSV, "
        "PBIDS connection file, Power Query M script, DAX measures, and design guide."
    )

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        st.download_button(
            label="Download Complete Power BI Bundle (.zip)",
            data=pkg.zip_bundle_bytes,
            file_name=f"{dataset_name}_PowerBI_Bundle.zip",
            mime="application/zip",
            type="primary",
            key="pbi_download_zip_button",
            use_container_width=True,
        )
    with col_info:
        st.caption(
            "Includes: `prepared_dataset.csv`, `power_bi_connection.pbids`, "
            "`power_query_transformations.m`, `dax_measures.dax`, and `dashboard_visual_guide.md`"
        )

    st.divider()

    # 4. Tabs for Detailed Exploration & Individual Assets
    tabs = st.tabs([
        "Semantic Schema Analysis",
        "Recommended DAX Measures",
        "Visual Layout Recipes",
        "Power Query (M) Script",
        "Power BI Desktop Guide",
    ])

    # Tab 1: Semantic Schema Analysis
    with tabs[0]:
        st.subheader("Detected Column Roles & Data Types")
        st.write(
            "Each column in your dataset is classified into business semantic roles "
            "to automatically tailor Power BI aggregations, hierarchies, and visuals."
        )

        schema_rows = []
        for col in analysis.columns:
            schema_rows.append({
                "Column Name": col.name,
                "Inferred Type": col.data_type,
                "Semantic Role": col.semantic_role.value,
                "Confidence": f"{int(col.confidence * 100)}%",
                "Missing Values": col.null_count,
                "Unique Values": col.unique_count,
                "Sample Values": ", ".join(map(str, col.sample_values[:3])),
                "Reason / Context": col.reason,
            })
        schema_df = pd.DataFrame(schema_rows)
        st.dataframe(schema_df, use_container_width=True, hide_index=True)

        if analysis.notes:
            for note in analysis.notes:
                st.info(f"**Notice:** {note}")

    # Tab 2: DAX Measures
    with tabs[1]:
        st.subheader("Tailored DAX Measures & Business Logic")
        st.write(
            "Copy and paste these DAX measures into Power BI Desktop via "
            "**Modeling -> New Measure** to power your KPIs and analytics."
        )

        if not analysis.dax_measures:
            st.info("No numeric fields available to generate automated DAX measures.")
        else:
            for measure in analysis.dax_measures:
                with st.expander(f"**{measure.name}** ({measure.display_folder})", expanded=False):
                    st.caption(f"**Purpose**: {measure.description}")
                    if measure.business_question:
                        st.caption(f"**Answers**: {measure.business_question}")
                    st.code(measure.dax_formula, language="dax")
                    st.caption(f"**Recommended Format**: `{measure.format_string}`")

            st.download_button(
                label="Download DAX Measures Script (.dax)",
                data=pkg.dax_script_text.encode("utf-8"),
                file_name=f"{dataset_name}_measures.dax",
                mime="text/plain",
                key="pbi_download_dax_button",
            )

    # Tab 3: Visual Layout Recipes
    with tabs[2]:
        st.subheader("Recommended Dashboard Visuals")
        st.write(
            "Use these visual recipes to arrange your interactive dashboard in Power BI Desktop."
        )

        if not analysis.visual_recommendations:
            st.info("No visual recommendations could be generated for this schema.")
        else:
            for idx, rec in enumerate(analysis.visual_recommendations, 1):
                with st.container(border=True):
                    st.markdown(f"#### {idx}. {rec.title}")
                    col_v1, col_v2 = st.columns([1, 1])
                    with col_v1:
                        st.markdown(f"**Visual Type in Power BI**: `{rec.visual_type}`")
                        st.markdown(f"**Recommended Page**: *{rec.suggested_page}*")
                        st.write(rec.description)
                    with col_v2:
                        st.markdown("**Field Configuration**:")
                        for well, fld in rec.fields.items():
                            if isinstance(fld, list):
                                st.write(f"- **{well}**: `{', '.join(fld)}`")
                            else:
                                st.write(f"- **{well}**: `{fld}`")
                    st.caption(f"**Design Tip**: {rec.configuration_notes}")

    # Tab 4: Power Query M Script
    with tabs[3]:
        st.subheader("Power Query (M) Transformation Code")
        st.write(
            "If you want exact reproducible type coercion and schema enforcement, "
            "paste this M code into Power BI's **Advanced Editor**."
        )
        st.code(pkg.powerquery_m_text, language="powerquery")

        st.download_button(
            label="Download Power Query Script (.m)",
            data=pkg.powerquery_m_text.encode("utf-8"),
            file_name=f"{dataset_name}_powerquery.m",
            mime="text/plain",
            key="pbi_download_pq_button",
        )

    # Tab 5: Step-by-Step Power BI Desktop Guide
    with tabs[4]:
        st.subheader("Step-by-Step Power BI Desktop Walkthrough")
        st.markdown(
            """
            ### How to create your dashboard in Microsoft Power BI Desktop:

            1. **Extract your downloaded ZIP package**:
               - Unzip `{dataset_name}_PowerBI_Bundle.zip` to a folder on your computer.

            2. **Connect Power BI Desktop**:
               - **Option A (Instant)**: Double-click `power_bi_connection.pbids`. Power BI Desktop will automatically open and prompt to import `prepared_dataset.csv`.
               - **Option B (Manual)**: Launch Power BI Desktop -> click **Get data** -> select **Text/CSV** -> pick `prepared_dataset.csv` -> click **Load**.

            3. **Create DAX Measures**:
               - In Power BI Desktop's top ribbon, open the **Modeling** tab.
               - Click **New Measure**, open `dax_measures.dax` (or view the DAX tab above), and paste your formulas.

            4. **Assemble Visuals**:
               - Refer to the **Visual Layout Recipes** tab to add Card visuals, Line charts, Bar charts, and Matrix tables.
               - Drag and drop the recommended fields into Power BI's **X-Axis**, **Y-Axis**, and **Values** wells.

            5. **Publish & Refresh**:
               - Whenever you clean or upload an updated dataset in this application, download the updated CSV into the same folder and click **Refresh** in Power BI Desktop.
            """.format(dataset_name=dataset_name)
        )

        st.download_button(
            label="Download Visual Design Guide (.md)",
            data=pkg.visual_guide_md_text.encode("utf-8"),
            file_name=f"{dataset_name}_PowerBI_Guide.md",
            mime="text/markdown",
            key="pbi_download_guide_button",
        )
