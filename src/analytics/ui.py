"""Interactive exploratory data analysis interface."""

from dataclasses import asdict

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics.correlation import analyze_correlations
from src.analytics.descriptive import summarize_dataset
from src.analytics.distribution import analyze_dataset_distributions
from src.analytics.time_series import analyze_time_series
from src.analytics.statistical_tests import (
    chi_square_test,
    confidence_interval_mean,
    independent_t_test,
    one_sample_t_test,
    one_way_anova,
)


def numerical_summaries_to_dataframe(summary) -> pd.DataFrame:
    """Convert numerical summaries into a table."""
    return pd.DataFrame(
        [asdict(item) for item in summary.numerical_summaries.values()]
    )


def categorical_summaries_to_dataframe(summary) -> pd.DataFrame:
    """Convert categorical summaries into a table."""
    rows = []
    for item in summary.categorical_summaries.values():
        rows.append(
            {
                "Column": item.column_name,
                "Valid values": item.count,
                "Missing values": item.missing_count,
                "Unique values": item.unique_count,
                "Mode": ", ".join(map(str, item.mode)),
                "Top frequencies": str(item.frequencies),
                "Top percentages": str(
                    {key: round(value, 2) for key, value in item.percentages.items()}
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "Column",
            "Valid values",
            "Missing values",
            "Unique values",
            "Mode",
            "Top frequencies",
            "Top percentages",
        ],
    )


def distributions_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert distribution results into a table."""
    rows = []
    for result in results.values():
        row = asdict(result)
        row["Interpretation"] = " ".join(row["interpretation"])
        row["Percentiles"] = str(row["percentiles"])
        del row["interpretation"]
        del row["percentiles"]
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_matrix_to_dataframe(result) -> pd.DataFrame:
    """Convert a correlation result into a matrix DataFrame."""
    return pd.DataFrame(result.matrix).T


def build_frequency_dataframe(
    data: pd.DataFrame, column: str, top_n: int = 20
) -> pd.DataFrame:
    """Build a category frequency table, including missing values."""
    if column not in data.columns:
        raise ValueError(f"Column not found: {column}")
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    counts = data[column].value_counts(dropna=False).head(top_n)
    rows = [
        {
            "Category": "(Missing)" if pd.isna(value) else str(value),
            "Count": int(count),
            "Percentage": round(float(count) / len(data) * 100, 2)
            if len(data)
            else 0.0,
        }
        for value, count in counts.items()
    ]
    return pd.DataFrame(rows, columns=["Category", "Count", "Percentage"])


def _get_numeric_columns(data: pd.DataFrame) -> list[str]:
    """Return numeric columns excluding Boolean columns."""
    return [
        column
        for column in data.select_dtypes(include="number").columns
        if not pd.api.types.is_bool_dtype(data[column].dtype)
    ]


def _get_categorical_columns(data: pd.DataFrame) -> list[str]:
    """Return non-numeric columns suitable for category charts."""
    return [
        column
        for column in data.columns
        if (
            pd.api.types.is_object_dtype(data[column].dtype)
            or isinstance(data[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(data[column].dtype)
            or pd.api.types.is_bool_dtype(data[column].dtype)
        )
    ]


def _get_date_columns(data: pd.DataFrame) -> list[str]:
    """Identify date-like non-numeric columns."""
    date_columns = []
    minimum_valid = max(2, int(len(data) * 0.5))

    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column].dtype):
            continue
        try:
            parsed = pd.to_datetime(data[column], errors="coerce", utc=True)
        except (TypeError, ValueError, OverflowError):
            continue
        if int(parsed.notna().sum()) >= minimum_valid:
            date_columns.append(column)

    return date_columns


def _show_test_result(result) -> None:
    """Render a consistent hypothesis-test result."""
    left, right = st.columns(2)
    left.metric(
        "Test statistic",
        f"{result.statistic:,.6g}" if result.statistic is not None else "N/A",
    )
    right.metric(
        "p-value",
        f"{result.p_value:.6g}" if result.p_value is not None else "N/A",
    )
    if result.significant is True:
        st.success(result.interpretation)
    elif result.significant is False:
        st.info(result.interpretation)
    else:
        st.warning(result.interpretation)
    if result.details:
        with st.expander("Test details and assumptions"):
            st.json(result.details)


def _render_time_series_tab(
    data: pd.DataFrame, dataset_index: int, numeric_columns: list[str]
) -> None:
    """Render interactive time-series analysis."""
    st.subheader("Time-Series Analysis")
    st.write(
        "Aggregate historical data by time period, inspect trends, "
        "and explore rolling averages."
    )

    date_columns = _get_date_columns(data)
    if not date_columns:
        st.info(
            "No suitable date column was detected. Upload data containing "
            "dates or timestamps."
        )
        return
    if not numeric_columns:
        st.info("A numerical column is required for time-series analysis.")
        return

    col1, col2 = st.columns(2)
    with col1:
        date_column = st.selectbox(
            "Date column",
            options=date_columns,
            key=f"eda_ts_date_{dataset_index}",
        )
        frequency = st.selectbox(
            "Time frequency",
            options=["D", "W", "M", "Q", "Y"],
            format_func={
                "D": "Daily",
                "W": "Weekly",
                "M": "Monthly",
                "Q": "Quarterly",
                "Y": "Yearly",
            }.get,
            index=2,
            key=f"eda_ts_frequency_{dataset_index}",
        )
    with col2:
        value_column = st.selectbox(
            "Numerical value",
            options=numeric_columns,
            key=f"eda_ts_value_{dataset_index}",
        )
        aggregation = st.selectbox(
            "Aggregation",
            options=["sum", "mean", "count"],
            format_func={"sum": "Sum", "mean": "Mean", "count": "Count"}.get,
            key=f"eda_ts_aggregation_{dataset_index}",
        )

    rolling_window = st.number_input(
        "Rolling average window (periods)",
        min_value=1,
        max_value=100,
        value=3,
        key=f"eda_ts_window_{dataset_index}",
    )

    if not st.button("Analyze time series", key=f"eda_ts_run_{dataset_index}"):
        return

    try:
        result = analyze_time_series(
            data=data,
            date_column=date_column,
            value_column=value_column,
            frequency=frequency,
            rolling_window=int(rolling_window),
            aggregation=aggregation,
        )
    except (ValueError, TypeError, KeyError) as exc:
        st.error(f"Time-series analysis failed: {exc}")
        return

    if result.observations == 0 or not result.period_summary:
        st.warning("No valid date-and-value periods were available.")
        for note in result.interpretation:
            st.write(f"- {note}")
        return

    metrics = st.columns(4)
    metrics[0].metric("Periods", f"{result.observations:,}")
    metrics[1].metric(
        "Mean", f"{result.mean:,.2f}" if result.mean is not None else "N/A"
    )
    metrics[2].metric("Trend", result.trend.replace("_", " ").title())
    metrics[3].metric(
        "Change from first to last",
        (
            f"{result.period_change_percentage:,.2f}%"
            if result.period_change_percentage is not None
            else "N/A"
        ),
    )

    periods = pd.DataFrame(result.period_summary)
    periods["period"] = pd.to_datetime(periods["period"], errors="coerce", utc=True)
    periods["value"] = pd.to_numeric(periods["value"], errors="coerce")

    # Match rolling averages by normalized ISO timestamps. Missing rolling values
    # remain NaN until enough periods are available for the requested window.
    rolling = result.rolling_average
    periods["rolling_average"] = periods["period"].map(
        lambda timestamp: rolling.get(timestamp.isoformat())
        if pd.notna(timestamp)
        else None
    )

    fig = px.line(
        periods,
        x="period",
        y="value",
        markers=True,
        title=f"{aggregation.title()} of {value_column} over time",
    )
    fig.add_scatter(
        x=periods["period"],
        y=periods["rolling_average"],
        mode="lines",
        name="Rolling average",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(periods, use_container_width=True, hide_index=True)
    st.download_button(
        "Download time-series results",
        data=periods.to_csv(index=False).encode("utf-8"),
        file_name="time_series_analysis.csv",
        mime="text/csv",
        key=f"eda_ts_download_{dataset_index}",
    )
    for note in result.interpretation:
        st.write(f"- {note}")


def _render_statistical_testing_tab(
    data: pd.DataFrame,
    dataset_index: int,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> None:
    """Render confidence intervals and hypothesis tests."""
    st.subheader("Statistical Testing")
    st.write(
        "Calculate confidence intervals and test hypotheses using "
        "numerical or categorical data."
    )

    test_type = st.selectbox(
        "Choose a statistical method",
        options=[
            "Confidence Interval",
            "One-sample t-test",
            "Welch's independent t-test",
            "Chi-square test",
            "One-way ANOVA",
        ],
        key=f"eda_test_type_{dataset_index}",
    )
    confidence_level = st.slider(
        "Confidence level",
        min_value=90,
        max_value=99,
        value=95,
        step=1,
        key=f"eda_test_confidence_{dataset_index}",
    )
    alpha = 1 - confidence_level / 100

    if test_type in {"Confidence Interval", "One-sample t-test"}:
        if not numeric_columns:
            st.info("A numerical column is required.")
            return

        value_column = st.selectbox(
            "Numerical column",
            options=numeric_columns,
            key=f"eda_test_numeric_{dataset_index}",
        )

        if test_type == "Confidence Interval":
            if st.button(
                "Calculate confidence interval",
                key=f"eda_ci_run_{dataset_index}",
            ):
                try:
                    result = confidence_interval_mean(
                        data[value_column],
                        confidence_level=confidence_level / 100,
                        column_name=value_column,
                    )
                    st.metric("Sample mean", f"{result.sample_mean:,.4f}")
                    st.write(
                        f"**{confidence_level}% confidence interval:** "
                        f"[{result.lower_bound:,.4f}, {result.upper_bound:,.4f}]"
                    )
                    st.write(f"Margin of error: {result.margin_of_error:,.4f}")
                    st.write(result.interpretation)
                except (ValueError, TypeError) as exc:
                    st.error(str(exc))
        else:
            population_mean = st.number_input(
                "Hypothesized population mean",
                value=0.0,
                key=f"eda_test_mean_{dataset_index}",
            )
            alternative = st.selectbox(
                "Alternative hypothesis",
                options=["two-sided", "less", "greater"],
                key=f"eda_test_alternative_{dataset_index}",
            )
            if st.button("Run t-test", key=f"eda_ttest_run_{dataset_index}"):
                try:
                    result = one_sample_t_test(
                        data[value_column],
                        population_mean=population_mean,
                        alternative=alternative,
                        alpha=alpha,
                    )
                    _show_test_result(result)
                except (ValueError, TypeError) as exc:
                    st.error(str(exc))
        return

    if test_type in {"Welch's independent t-test", "One-way ANOVA"}:
        if not numeric_columns or not categorical_columns:
            st.info(
                "A numerical column and a categorical grouping column "
                "are required."
            )
            return

        value_column = st.selectbox(
            "Numerical value",
            options=numeric_columns,
            key=f"eda_group_value_{dataset_index}",
        )
        group_column = st.selectbox(
            "Group by",
            options=categorical_columns,
            key=f"eda_group_column_{dataset_index}",
        )
        group_values = (
            data[group_column].dropna().value_counts().head(20).index.tolist()
        )

        if test_type == "Welch's independent t-test":
            if len(group_values) < 2:
                st.info("At least two groups are required.")
                return
            selected_groups = st.multiselect(
                "Select exactly two groups",
                options=group_values,
                default=group_values[:2],
                max_selections=2,
                key=f"eda_welch_groups_{dataset_index}",
            )
            if st.button(
                "Compare group means", key=f"eda_welch_run_{dataset_index}"
            ):
                if len(selected_groups) != 2:
                    st.warning("Select exactly two groups first.")
                    return
                try:
                    group_a = data.loc[
                        data[group_column] == selected_groups[0], value_column
                    ]
                    group_b = data.loc[
                        data[group_column] == selected_groups[1], value_column
                    ]
                    result = independent_t_test(
                        group_a, group_b, alpha=alpha
                    )
                    _show_test_result(result)
                except (ValueError, TypeError) as exc:
                    st.error(str(exc))
        else:
            st.caption(
                "ANOVA assumes independent observations, approximately normal "
                "within-group residuals, and equal population variances."
            )
            selected_groups = st.multiselect(
                "Select at least three groups",
                options=group_values,
                default=group_values[:3],
                key=f"eda_anova_groups_{dataset_index}",
            )
            if st.button("Run ANOVA", key=f"eda_anova_run_{dataset_index}"):
                if len(selected_groups) < 3:
                    st.warning("Select at least three groups first.")
                    return
                try:
                    groups = [
                        data.loc[data[group_column] == group, value_column]
                        for group in selected_groups
                    ]
                    result = one_way_anova(groups, alpha=alpha)
                    _show_test_result(result)
                    st.caption(
                        "A significant result means at least one group mean "
                        "differs; it does not identify which groups differ."
                    )
                except (ValueError, TypeError) as exc:
                    st.error(str(exc))
        return

    if len(categorical_columns) < 2:
        st.info("Two categorical columns are required for the chi-square test.")
        return

    col1, col2 = st.columns(2)
    with col1:
        category_a = st.selectbox(
            "First categorical variable",
            options=categorical_columns,
            key=f"eda_chi_a_{dataset_index}",
        )
    category_b_options = [
        column for column in categorical_columns if column != category_a
    ]
    with col2:
        category_b = st.selectbox(
            "Second categorical variable",
            options=category_b_options,
            key=f"eda_chi_b_{dataset_index}",
        )

    if st.button("Run chi-square test", key=f"eda_chi_run_{dataset_index}"):
        try:
            contingency_table = pd.crosstab(
                data[category_a], data[category_b]
            )
            result = chi_square_test(contingency_table, alpha=alpha)
            st.dataframe(contingency_table, use_container_width=True)
            _show_test_result(result)
            st.caption(
                "Check expected cell frequencies; small expected counts can "
                "make the chi-square approximation unreliable."
            )
        except (ValueError, TypeError) as exc:
            st.error(str(exc))


def render_eda_page() -> None:
    """Render the exploratory data analysis page."""
    st.title("Exploratory Data Analysis")
    st.write(
        "Explore statistical summaries, numerical distributions, category "
        "frequencies, correlations, time-series trends, statistical tests, "
        "and interactive charts."
    )

    datasets = st.session_state.get("workspace_datasets", [])
    if not datasets:
        st.info(
            "Load a dataset in Data Workspace first. Your loaded datasets "
            "will then be available here."
        )
        return

    dataset_index = st.selectbox(
        "Select dataset",
        options=list(range(len(datasets))),
        format_func=lambda index: datasets[index].name,
        key="eda_dataset",
    )
    dataset = datasets[dataset_index]
    data = dataset.data

    cleaning_results = st.session_state.get("workspace_cleaning_results", {})
    cleaning_result = cleaning_results.get(dataset_index)
    versions = ["Original"]
    if cleaning_result is not None:
        versions.append("Cleaned")

    version = st.radio(
        "Analyze which version?",
        options=versions,
        horizontal=True,
        key=f"eda_version_{dataset_index}",
    )
    if version == "Cleaned" and cleaning_result is not None:
        data = cleaning_result.cleaned_data

    if data.empty:
        st.warning("The selected dataset has no rows.")
        return

    numeric_columns = _get_numeric_columns(data)
    categorical_columns = _get_categorical_columns(data)

    st.caption(
        f"Analyzing {dataset.name} — {len(data):,} rows, "
        f"{len(data.columns):,} columns."
    )
    try:
        summary = summarize_dataset(data, dataset_name=dataset.name)
    except Exception as exc:
        st.error(f"Could not summarize the dataset: {exc}")
        return

    metrics = st.columns(4)
    metrics[0].metric("Rows", f"{len(data):,}")
    metrics[1].metric("Columns", f"{len(data.columns):,}")
    metrics[2].metric("Numerical columns", summary.numerical_column_count)
    metrics[3].metric("Categorical columns", summary.categorical_column_count)

    tabs = st.tabs(
        [
            "Statistical Summary",
            "Distributions",
            "Correlations",
            "Visual Explorer",
            "Time-Series Analysis",
            "Statistical Testing",
        ]
    )

    with tabs[0]:
        st.subheader("Numerical statistics")
        numerical_table = numerical_summaries_to_dataframe(summary)
        if numerical_table.empty:
            st.info("No numerical columns were found.")
        else:
            st.dataframe(
                numerical_table, use_container_width=True, hide_index=True
            )
            st.download_button(
                "Download numerical summary",
                data=numerical_table.to_csv(index=False).encode("utf-8"),
                file_name="numerical_summary.csv",
                mime="text/csv",
                key=f"eda_download_numeric_{dataset_index}",
            )

        st.subheader("Categorical statistics")
        categorical_table = categorical_summaries_to_dataframe(summary)
        if categorical_table.empty:
            st.info("No categorical columns were found.")
        else:
            st.dataframe(
                categorical_table, use_container_width=True, hide_index=True
            )
            st.download_button(
                "Download categorical summary",
                data=categorical_table.to_csv(index=False).encode("utf-8"),
                file_name="categorical_summary.csv",
                mime="text/csv",
                key=f"eda_download_categorical_{dataset_index}",
            )

    with tabs[1]:
        st.subheader("Numerical distribution analysis")
        if not numeric_columns:
            st.info("This dataset has no numerical columns to analyze.")
        else:
            selected_columns = st.multiselect(
                "Select numerical columns",
                options=numeric_columns,
                default=numeric_columns,
                key=f"eda_distribution_columns_{dataset_index}",
            )
            if not selected_columns:
                st.info("Select at least one numerical column.")
            else:
                try:
                    results = analyze_dataset_distributions(
                        data, columns=selected_columns
                    )
                    distribution_table = distributions_to_dataframe(results)
                    st.dataframe(
                        distribution_table,
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        "Download distribution analysis",
                        data=distribution_table.to_csv(index=False).encode("utf-8"),
                        file_name="distribution_analysis.csv",
                        mime="text/csv",
                        key=f"eda_download_distribution_{dataset_index}",
                    )
                    for result in results.values():
                        with st.expander(
                            f"{result.column_name}: "
                            f"{result.shape.replace('_', ' ').title()}"
                        ):
                            for interpretation in result.interpretation:
                                st.write(f"- {interpretation}")
                except Exception as exc:
                    st.error(f"Distribution analysis failed: {exc}")

    with tabs[2]:
        st.subheader("Correlation analysis")
        if len(numeric_columns) < 2:
            st.info(
                "At least two numerical columns are required for correlation analysis."
            )
        else:
            selected_columns = st.multiselect(
                "Columns to compare",
                options=numeric_columns,
                default=numeric_columns,
                key=f"eda_correlation_columns_{dataset_index}",
            )
            method = st.selectbox(
                "Correlation method",
                options=["pearson", "spearman"],
                format_func={
                    "pearson": "Pearson — linear association",
                    "spearman": "Spearman — rank-based monotonic association",
                }.get,
                key=f"eda_correlation_method_{dataset_index}",
            )
            top_n = st.number_input(
                "Number of strongest pairs to show",
                min_value=1,
                max_value=100,
                value=10,
                key=f"eda_correlation_top_n_{dataset_index}",
            )
            if len(selected_columns) < 2:
                st.info("Select at least two numerical columns.")
            else:
                try:
                    result = analyze_correlations(
                        data,
                        columns=selected_columns,
                        method=method,
                        top_n=int(top_n),
                    )
                    matrix = correlation_matrix_to_dataframe(result)
                    st.write("**Correlation matrix**")
                    st.dataframe(
                        matrix.style.format("{:.3f}", na_rep="—"),
                        use_container_width=True,
                    )
                    st.plotly_chart(
                        px.imshow(
                            matrix,
                            text_auto=".2f",
                            aspect="auto",
                            color_continuous_scale="RdBu",
                            zmin=-1,
                            zmax=1,
                            title=f"{method.title()} correlation heatmap",
                        ),
                        use_container_width=True,
                    )
                    st.write("**Strongest relationships**")
                    if result.strongest_pairs:
                        st.dataframe(
                            pd.DataFrame(result.strongest_pairs),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info(
                            "No valid column pairs were available for comparison."
                        )
                    for interpretation in result.interpretation:
                        st.write(f"- {interpretation}")
                    st.caption(
                        "Correlation measures association, not causation. "
                        "Constant columns or insufficient overlapping values "
                        "may produce undefined correlations."
                    )
                except Exception as exc:
                    st.error(f"Correlation analysis failed: {exc}")

    with tabs[3]:
        st.subheader("Interactive visual explorer")
        chart_options = [
            "Histogram",
            "Box plot",
            "Category frequency",
            "Scatter plot",
        ]
        if len(numeric_columns) >= 2:
            chart_options.append("Correlation heatmap")

        chart_type = st.selectbox(
            "Chart type",
            options=chart_options,
            key=f"eda_chart_type_{dataset_index}",
        )

        if chart_type in {"Histogram", "Box plot"}:
            if not numeric_columns:
                st.warning("No numerical columns are available.")
            else:
                column = st.selectbox(
                    "Numerical column",
                    options=numeric_columns,
                    key=f"eda_single_numeric_{dataset_index}",
                )
                if chart_type == "Histogram":
                    fig = px.histogram(
                        data,
                        x=column,
                        title=f"Distribution of {column}",
                        marginal="box",
                    )
                else:
                    fig = px.box(
                        data,
                        y=column,
                        points="outliers",
                        title=f"Box plot of {column}",
                    )
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Category frequency":
            if not categorical_columns:
                st.warning("No categorical columns are available.")
            else:
                column = st.selectbox(
                    "Categorical column",
                    options=categorical_columns,
                    key=f"eda_single_category_{dataset_index}",
                )
                top_n = st.slider(
                    "Number of categories",
                    min_value=5,
                    max_value=30,
                    value=10,
                    key=f"eda_category_top_n_{dataset_index}",
                )
                frequency_table = build_frequency_dataframe(
                    data, column, top_n=top_n
                )
                st.plotly_chart(
                    px.bar(
                        frequency_table,
                        x="Category",
                        y="Count",
                        title=f"Most frequent values in {column}",
                    ),
                    use_container_width=True,
                )
                st.dataframe(
                    frequency_table, use_container_width=True, hide_index=True
                )

        elif chart_type == "Scatter plot":
            if len(numeric_columns) < 2:
                st.warning("At least two numerical columns are required.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    x_column = st.selectbox(
                        "X-axis",
                        options=numeric_columns,
                        index=0,
                        key=f"eda_scatter_x_{dataset_index}",
                    )
                y_options = [
                    column for column in numeric_columns if column != x_column
                ]
                with col2:
                    y_column = st.selectbox(
                        "Y-axis",
                        options=y_options,
                        key=f"eda_scatter_y_{dataset_index}",
                    )
                try:
                    fig = px.scatter(
                        data,
                        x=x_column,
                        y=y_column,
                        title=f"{y_column} vs {x_column}",
                        trendline="ols",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except (ValueError, TypeError, ImportError) as exc:
                    st.error(f"Could not generate the scatter plot: {exc}")

        elif chart_type == "Correlation heatmap":
            try:
                result = analyze_correlations(data, method="pearson")
                matrix = correlation_matrix_to_dataframe(result)
                st.plotly_chart(
                    px.imshow(
                        matrix,
                        text_auto=".2f",
                        aspect="auto",
                        color_continuous_scale="RdBu",
                        zmin=-1,
                        zmax=1,
                        title="Numerical correlation heatmap",
                    ),
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Could not generate the heatmap: {exc}")

    with tabs[4]:
        _render_time_series_tab(data, dataset_index, numeric_columns)

    with tabs[5]:
        _render_statistical_testing_tab(
            data, dataset_index, numeric_columns, categorical_columns
        )
