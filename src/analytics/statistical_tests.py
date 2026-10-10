
"""Statistical tests and confidence intervals for analytics datasets."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StatisticalTestResult:
    """Stores the result of a statistical hypothesis test."""

    test_name: str
    statistic: float | None
    p_value: float | None
    alpha: float
    significant: bool | None
    interpretation: str
    details: dict = field(default_factory=dict)


@dataclass
class ConfidenceIntervalResult:
    """Stores a confidence interval for a population mean."""

    column_name: str
    sample_size: int
    confidence_level: float
    sample_mean: float
    lower_bound: float
    upper_bound: float
    margin_of_error: float
    interpretation: str


def _validate_alpha(alpha):
    """Ensure alpha is a valid significance level."""
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float, np.integer, np.floating))
        or not np.isfinite(alpha)
        or not 0 < alpha < 1
    ):
        raise ValueError("alpha must be a number strictly between 0 and 1.")


def _clean_numeric_data(values, name="data"):
    """Convert input to finite numeric values and remove invalid entries."""
    if isinstance(values, pd.DataFrame):
        raise TypeError(f"{name} must be a one-dimensional series or array.")

    try:
        series = pd.Series(values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a one-dimensional series or array.") from exc

    if isinstance(values, (str, bytes)) or series.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional series or array.")

    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()

    return numeric.to_numpy(dtype=float)


def _validate_minimum_sample_size(values, minimum=2, name="data"):
    """Require a minimum number of valid observations."""
    if len(values) < minimum:
        raise ValueError(
            f"{name} must contain at least {minimum} valid numeric observations."
        )


def _safe_float(value):
    """Convert a result to a finite float, or None."""
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _build_test_result(
    test_name,
    statistic,
    p_value,
    alpha,
    details=None,
):
    """Create a consistent result for a hypothesis test."""
    statistic = _safe_float(statistic)
    p_value = _safe_float(p_value)

    if p_value is None:
        significant = None
        interpretation = (
            "The test did not produce a finite p-value, "
            "so statistical significance could not be determined."
        )
    else:
        significant = p_value < alpha

        if significant:
            interpretation = (
                f"The result is statistically significant at alpha={alpha}. "
                "Reject the null hypothesis."
            )
        else:
            interpretation = (
                f"The result is not statistically significant at alpha={alpha}. "
                "Fail to reject the null hypothesis."
            )

    return StatisticalTestResult(
        test_name=test_name,
        statistic=statistic,
        p_value=p_value,
        alpha=float(alpha),
        significant=significant,
        interpretation=interpretation,
        details=details or {},
    )


def confidence_interval_mean(
    values,
    confidence_level=0.95,
    column_name="value",
):
    """
    Calculate a t-based confidence interval for a population mean.

    Missing and non-finite values are excluded.
    At least two valid observations are required.
    """
    if (
        isinstance(confidence_level, bool)
        or not isinstance(
            confidence_level,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError("confidence_level must be between 0 and 1.")

    data = _clean_numeric_data(values, "values")
    _validate_minimum_sample_size(data, minimum=2, name="values")

    sample_size = len(data)
    sample_mean = float(np.mean(data))
    sample_std = float(np.std(data, ddof=1))

    standard_error = sample_std / np.sqrt(sample_size)
    degrees_of_freedom = sample_size - 1

    critical_value = stats.t.ppf(
        (1 + confidence_level) / 2,
        df=degrees_of_freedom,
    )

    margin_of_error = float(critical_value * standard_error)
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    return ConfidenceIntervalResult(
        column_name=str(column_name),
        sample_size=sample_size,
        confidence_level=float(confidence_level),
        sample_mean=sample_mean,
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        margin_of_error=margin_of_error,
        interpretation=(
            f"The {confidence_level:.0%} confidence interval for the "
            f"population mean of '{column_name}' is "
            f"[{lower_bound:.4f}, {upper_bound:.4f}]."
        ),
    )


def one_sample_t_test(
    values,
    population_mean=0,
    alternative="two-sided",
    alpha=0.05,
):
    """
    Test whether a sample mean differs from a specified population mean.

    alternative: 'two-sided', 'less', or 'greater'.
    """
    _validate_alpha(alpha)

    if (
        isinstance(population_mean, bool)
        or not isinstance(
            population_mean,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(population_mean)
    ):
        raise ValueError("population_mean must be a finite number.")

    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError(
            "alternative must be 'two-sided', 'less', or 'greater'."
        )

    data = _clean_numeric_data(values, "values")
    _validate_minimum_sample_size(data, name="values")

    test = stats.ttest_1samp(
        data,
        popmean=float(population_mean),
        alternative=alternative,
    )

    return _build_test_result(
        test_name="One-sample t-test",
        statistic=test.statistic,
        p_value=test.pvalue,
        alpha=alpha,
        details={
            "sample_size": len(data),
            "sample_mean": float(np.mean(data)),
            "population_mean": float(population_mean),
            "alternative": alternative,
            "null_hypothesis": (
                f"The population mean equals {population_mean}."
            ),
        },
    )


def independent_t_test(
    group_a,
    group_b,
    alternative="two-sided",
    alpha=0.05,
):
    """
    Compare the means of two independent groups.

    Welch's t-test is used because it does not assume equal variances.
    alternative: 'two-sided', 'less', or 'greater'.
    """
    _validate_alpha(alpha)

    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError(
            "alternative must be 'two-sided', 'less', or 'greater'."
        )

    first = _clean_numeric_data(group_a, "group_a")
    second = _clean_numeric_data(group_b, "group_b")

    _validate_minimum_sample_size(first, name="group_a")
    _validate_minimum_sample_size(second, name="group_b")

    test = stats.ttest_ind(
        first,
        second,
        equal_var=False,
        alternative=alternative,
    )

    return _build_test_result(
        test_name="Independent two-sample Welch's t-test",
        statistic=test.statistic,
        p_value=test.pvalue,
        alpha=alpha,
        details={
            "group_a_size": len(first),
            "group_b_size": len(second),
            "group_a_mean": float(np.mean(first)),
            "group_b_mean": float(np.mean(second)),
            "mean_difference": float(np.mean(first) - np.mean(second)),
            "alternative": alternative,
            "null_hypothesis": "The two population means are equal.",
        },
    )


def chi_square_test(contingency_table, alpha=0.05):
    """
    Test the association between two categorical variables.

    Input must be a contingency table of non-negative observed counts.
    Rows and columns represent category combinations.
    """
    _validate_alpha(alpha)

    if not isinstance(contingency_table, pd.DataFrame):
        raise TypeError("contingency_table must be a pandas DataFrame.")

    if contingency_table.empty:
        raise ValueError("contingency_table cannot be empty.")

    try:
        numeric_table = contingency_table.apply(
            pd.to_numeric,
            errors="raise",
        )
        observed = numeric_table.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "The contingency table must contain numeric counts."
        ) from exc

    if not np.isfinite(observed).all():
        raise ValueError("The contingency table cannot contain missing or infinite values.")

    if (observed < 0).any():
        raise ValueError("Observed counts cannot be negative.")

    if not np.equal(observed, np.floor(observed)).all():
        raise ValueError("Observed counts must be whole numbers.")

    # Remove categories with zero totals; they provide no information.
    observed = observed[
        observed.sum(axis=1) > 0
    ]
    if observed.size:
        observed = observed[
            :, observed.sum(axis=0) > 0
        ]

    if observed.shape[0] < 2 or observed.shape[1] < 2:
        raise ValueError(
            "At least two non-empty categories are required in both dimensions."
        )

    statistic, p_value, degrees_of_freedom, expected = (
        stats.chi2_contingency(observed)
    )

    return _build_test_result(
        test_name="Chi-square test of independence",
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
        details={
            "degrees_of_freedom": int(degrees_of_freedom),
            "sample_size": int(observed.sum()),
            "expected_frequencies": expected.tolist(),
            "minimum_expected_frequency": float(expected.min()),
            "null_hypothesis": (
                "The two categorical variables are independent."
            ),
            "assumption_note": (
                "Check expected cell frequencies. Small expected counts "
                "can make the chi-square approximation unreliable."
            ),
        },
    )


def one_way_anova(groups, alpha=0.05):
    """
    Compare the means of three or more independent groups.

    Each item in groups must be a one-dimensional series or array.
    Standard one-way ANOVA assumes independent observations,
    approximately normal within-group residuals, and equal variances.
    """
    _validate_alpha(alpha)

    if isinstance(groups, (str, bytes, pd.DataFrame)):
        raise TypeError("groups must be a sequence of numeric groups.")

    try:
        raw_groups = list(groups)
    except TypeError as exc:
        raise TypeError(
            "groups must be a sequence of numeric groups."
        ) from exc

    if len(raw_groups) < 3:
        raise ValueError("At least three groups are required for ANOVA.")

    cleaned_groups = []

    for index, group in enumerate(raw_groups, start=1):
        cleaned = _clean_numeric_data(group, f"group_{index}")
        _validate_minimum_sample_size(
            cleaned,
            name=f"group_{index}",
        )
        cleaned_groups.append(cleaned)

    test = stats.f_oneway(*cleaned_groups)

    group_means = [
        float(np.mean(group))
        for group in cleaned_groups
    ]

    return _build_test_result(
        test_name="One-way ANOVA",
        statistic=test.statistic,
        p_value=test.pvalue,
        alpha=alpha,
        details={
            "number_of_groups": len(cleaned_groups),
            "group_sizes": [len(group) for group in cleaned_groups],
            "group_means": group_means,
            "total_observations": sum(
                len(group) for group in cleaned_groups
            ),
            "null_hypothesis": (
                "All group population means are equal."
            ),
            "assumption_note": (
                "Assumes independent observations, approximately normal "
                "within-group residuals, and equal population variances. "
                "A significant result indicates that at least one mean "
                "differs, not which specific groups differ."
            ),
        },
    )
