
import pandas as pd
import pytest

from src.analytics.distribution import (
    analyze_distribution,
    analyze_dataset_distributions,
)


def test_right_skewed_distribution():
    series = pd.Series([10, 11, 12, 13, 100])

    result = analyze_distribution(series, "sales")

    assert result.count == 5
    assert result.shape == "right_skewed"
    assert result.mean > result.median


def test_missing_values_are_counted():
    series = pd.Series([10.0, 20.0, None, 40.0])

    result = analyze_distribution(series, "sales")

    assert result.count == 3
    assert result.missing_count == 1


def test_constant_distribution():
    result = analyze_distribution(
        pd.Series([5, 5, 5, 5]),
        "quantity",
    )

    assert result.shape == "constant"
    assert result.outlier_count == 0


def test_empty_distribution():
    result = analyze_distribution(
        pd.Series([None, None]),
        "sales",
    )

    assert result.count == 0
    assert result.shape == "insufficient_data"


def test_dataset_distribution_analysis():
    data = pd.DataFrame({
        "sales": [10, 11, 12, 13, 100],
        "region": ["East", "West", "East", "North", "West"],
    })

    results = analyze_dataset_distributions(data)

    assert "sales" in results
    assert "region" not in results


def test_invalid_column():
    data = pd.DataFrame({"sales": [10, 20, 30]})

    with pytest.raises(ValueError):
        analyze_dataset_distributions(data, ["missing_column"])


def test_invalid_iqr_multiplier():
    with pytest.raises(ValueError):
        analyze_distribution(
            pd.Series([1, 2, 3]),
            "sales",
            iqr_multiplier=0,
        )
