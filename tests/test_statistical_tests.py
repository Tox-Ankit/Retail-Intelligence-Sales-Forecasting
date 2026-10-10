
import numpy as np
import pandas as pd
import pytest

from src.analytics.statistical_tests import (
    confidence_interval_mean,
    one_sample_t_test,
    independent_t_test,
    chi_square_test,
    one_way_anova,
)


def test_confidence_interval_contains_sample_mean():
    result = confidence_interval_mean([10, 12, 13, 15, 16])

    assert result.lower_bound <= result.sample_mean
    assert result.sample_mean <= result.upper_bound
    assert result.sample_size == 5
    assert result.confidence_level == 0.95


def test_confidence_interval_excludes_missing_values():
    result = confidence_interval_mean([10, 12, np.nan, 14])

    assert result.sample_size == 3
    assert result.sample_mean == pytest.approx(12.0)


def test_confidence_interval_invalid_level():
    with pytest.raises(ValueError):
        confidence_interval_mean([1, 2, 3], confidence_level=1.0)


def test_confidence_interval_requires_two_values():
    with pytest.raises(ValueError):
        confidence_interval_mean([5])


def test_one_sample_t_test_detects_difference():
    result = one_sample_t_test(
        [20, 21, 22, 23, 24],
        population_mean=10,
    )

    assert result.test_name == "One-sample t-test"
    assert result.p_value < 0.05
    assert result.significant is True


def test_one_sample_t_test_invalid_alternative():
    with pytest.raises(ValueError):
        one_sample_t_test([1, 2, 3], alternative="invalid")


def test_independent_t_test_compares_groups():
    result = independent_t_test(
        [10, 11, 12, 10, 11],
        [20, 21, 22, 20, 21],
    )

    assert result.p_value < 0.05
    assert result.significant is True
    assert result.details["mean_difference"] < 0


def test_independent_t_test_requires_two_groups_with_enough_data():
    with pytest.raises(ValueError):
        independent_t_test([1], [2, 3])


def test_chi_square_test_detects_association():
    table = pd.DataFrame(
        [
            [40, 10],
            [10, 40],
        ],
        index=["Group A", "Group B"],
        columns=["Category X", "Category Y"],
    )

    result = chi_square_test(table)

    assert result.p_value < 0.05
    assert result.significant is True
    assert result.details["degrees_of_freedom"] == 1


def test_chi_square_rejects_negative_counts():
    table = pd.DataFrame([[10, -2], [3, 4]])

    with pytest.raises(ValueError):
        chi_square_test(table)


def test_chi_square_requires_two_categories_each_way():
    table = pd.DataFrame([[10, 20]])

    with pytest.raises(ValueError):
        chi_square_test(table)


def test_anova_detects_different_group_means():
    result = one_way_anova(
        [
            [10, 11, 12, 10, 11],
            [20, 21, 22, 20, 21],
            [30, 31, 32, 30, 31],
        ]
    )

    assert result.p_value < 0.05
    assert result.significant is True
    assert result.details["number_of_groups"] == 3


def test_anova_requires_three_groups():
    with pytest.raises(ValueError):
        one_way_anova([[1, 2], [3, 4]])


def test_invalid_alpha_is_rejected():
    with pytest.raises(ValueError):
        one_sample_t_test([1, 2, 3], alpha=0)


def test_statistical_tests_do_not_modify_input():
    data = pd.Series([10, 12, 14, 16])
    original = data.copy(deep=True)

    one_sample_t_test(data, population_mean=10)

    pd.testing.assert_series_equal(data, original)
