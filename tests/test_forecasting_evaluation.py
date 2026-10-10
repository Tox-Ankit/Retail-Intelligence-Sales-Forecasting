
import numpy as np
import pandas as pd
import pytest

from src.forecasting.evaluation import walk_forward_backtest


@pytest.fixture
def monthly_series():
    dates = pd.date_range(
        "2024-01-31",
        periods=18,
        freq="ME",
        tz="UTC",
    )

    values = [
        100, 105, 110, 115, 120, 125,
        130, 135, 140, 145, 150, 155,
        160, 165, 170, 175, 180, 185,
    ]

    return pd.Series(values, index=dates, name="sales")


def test_backtest_compares_three_models(monthly_series):
    result = walk_forward_backtest(
        monthly_series,
        initial_train_size=10,
        horizon=2,
        step_size=2,
    )

    assert set(result.fold_results["Model"]) == {
        "Naive",
        "Moving Average",
        "Exponential Smoothing",
    }


def test_backtest_creates_expected_number_of_folds(monthly_series):
    result = walk_forward_backtest(
        monthly_series,
        initial_train_size=10,
        horizon=2,
        step_size=2,
    )

    # Folds begin at positions 10, 12, 14, and 16.
    assert result.fold_results["Fold"].nunique() == 4


def test_each_fold_contains_all_models(monthly_series):
    result = walk_forward_backtest(
        monthly_series,
        initial_train_size=10,
        horizon=2,
        step_size=2,
    )

    counts = result.fold_results.groupby("Fold")["Model"].nunique()

    assert (counts == 3).all()


def test_summary_contains_expected_metrics(monthly_series):
    result = walk_forward_backtest(
        monthly_series,
        initial_train_size=10,
        horizon=2,
    )

    expected_columns = {
        "Model",
        "Mean MAE",
        "Mean RMSE",
        "Mean MAPE (%)",
        "Folds Evaluated",
    }

    assert expected_columns.issubset(result.summary.columns)


def test_best_model_has_lowest_mean_rmse(monthly_series):
    result = walk_forward_backtest(
        monthly_series,
        initial_train_size=10,
        horizon=2,
    )

    best = result.summary.iloc[0]

    assert best["Model"] == result.best_model_name
    assert best["Mean RMSE"] == result.summary["Mean RMSE"].min()


def test_fold_test_periods_follow_training_periods(monthly_series):
    result = walk_forward_backtest(
        monthly_series,
        initial_train_size=10,
        horizon=2,
        step_size=2,
    )

    for _, row in result.fold_results.iterrows():
        assert row["Train End"] < row["Test Start"]


def test_backtest_rejects_missing_values(monthly_series):
    series = monthly_series.copy()
    series.iloc[4] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        walk_forward_backtest(series, initial_train_size=10)


def test_backtest_rejects_unsorted_series(monthly_series):
    series = monthly_series.iloc[::-1]

    with pytest.raises(ValueError, match="sorted"):
        walk_forward_backtest(series, initial_train_size=10)


def test_backtest_rejects_insufficient_data(monthly_series):
    with pytest.raises(ValueError, match="Not enough observations"):
        walk_forward_backtest(
            monthly_series,
            initial_train_size=17,
            horizon=2,
        )


def test_backtest_rejects_invalid_horizon(monthly_series):
    with pytest.raises(ValueError, match="horizon"):
        walk_forward_backtest(
            monthly_series,
            initial_train_size=10,
            horizon=0,
        )


def test_backtest_rejects_invalid_step_size(monthly_series):
    with pytest.raises(ValueError, match="step_size"):
        walk_forward_backtest(
            monthly_series,
            initial_train_size=10,
            step_size=0,
        )


def test_backtest_rejects_non_datetime_index():
    series = pd.Series([10, 20, 30, 40, 50])

    with pytest.raises(ValueError, match="DatetimeIndex"):
        walk_forward_backtest(series, initial_train_size=3)


def test_backtest_returns_evaluation_metadata(monthly_series):
    result = walk_forward_backtest(
        monthly_series,
        initial_train_size=10,
        horizon=2,
        step_size=2,
    )

    assert result.initial_train_size == 10
    assert result.horizon == 2
    assert result.step_size == 2


def test_backtest_rejects_non_integer_initial_window(monthly_series):
    with pytest.raises(TypeError, match="initial_train_size"):
        walk_forward_backtest(
            monthly_series,
            initial_train_size=10.5,
        )

def test_seasonal_models_can_be_enabled(monthly_series):
    result = walk_forward_backtest(
        monthly_series,
        initial_train_size=12,
        horizon=2,
        step_size=2,
        include_seasonal=True,
        season_length=6,
    )

    assert {
        "Naive",
        "Moving Average",
        "Exponential Smoothing",
        "Seasonal Naive",
        "Holt-Winters",
    }.issubset(set(result.fold_results["Model"]))


def test_seasonal_models_are_optional_by_default(monthly_series):
    result = walk_forward_backtest(
        monthly_series,
        initial_train_size=10,
        horizon=2,
    )

    assert "Seasonal Naive" not in set(result.fold_results["Model"])
    assert "Holt-Winters" not in set(result.fold_results["Model"])


def test_seasonal_mode_requires_season_length(monthly_series):
    with pytest.raises(ValueError, match="season_length is required"):
        walk_forward_backtest(
            monthly_series,
            initial_train_size=12,
            include_seasonal=True,
        )


def test_seasonal_mode_requires_two_training_seasons(monthly_series):
    with pytest.raises(ValueError, match="two complete seasons"):
        walk_forward_backtest(
            monthly_series,
            initial_train_size=10,
            include_seasonal=True,
            season_length=6,
        )


def test_integrated_summary_contains_successful_model_metrics(monthly_series):
    result = walk_forward_backtest(
        monthly_series,
        initial_train_size=12,
        horizon=2,
        step_size=2,
        include_seasonal=True,
        season_length=6,
    )

    assert "Mean RMSE" in result.summary.columns
    assert "Folds Evaluated" in result.summary.columns
    assert result.best_model_name in set(result.summary["Model"])
