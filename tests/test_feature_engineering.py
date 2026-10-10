
import numpy as np
import pandas as pd
import pytest

from src.analytics.feature_engineering import (
    add_date_features,
    add_retail_metrics,
    log_transform_numeric,
    encode_categorical_features,
)


def test_add_date_features():
    data = pd.DataFrame({
        "order_date": ["2026-01-03", "2026-01-05"]
    })

    result = add_date_features(data, "order_date")

    assert result["order_date_year"].tolist() == [2026, 2026]
    assert result["order_date_month"].tolist() == [1, 1]
    assert result["order_date_quarter"].tolist() == [1, 1]
    assert result["order_date_day"].tolist() == [3, 5]

    assert "order_date_dayofweek" not in result.columns
    assert "order_date_is_weekend" not in result.columns


def test_invalid_dates_produce_missing_features():
    data = pd.DataFrame({"date": ["2026-01-01", "invalid"]})
    result = add_date_features(data, "date", features=["year"])

    assert result["date_year"].iloc[0] == 2026
    assert pd.isna(result["date_year"].iloc[1])


def test_date_feature_missing_column():
    with pytest.raises(ValueError):
        add_date_features(
            pd.DataFrame({"date": ["2026-01-01"]}),
            "missing_date",
        )


def test_date_features_can_drop_original_column():
    data = pd.DataFrame({"date": ["2026-01-01"]})

    result = add_date_features(
        data, "date", features=["year"], drop_original=True
    )

    assert "date" not in result.columns
    assert result["date_year"].iloc[0] == 2026


def test_add_retail_metrics():
    data = pd.DataFrame({
        "quantity": [2, 3],
        "unit_price": [100.0, 50.0],
        "unit_cost": [60.0, 30.0],
    })

    result = add_retail_metrics(
        data,
        quantity_column="quantity",
        unit_price_column="unit_price",
        revenue_column="revenue",
        unit_cost_column="unit_cost",
        cost_column="cost",
        profit_column="profit",
    )

    assert result["revenue"].tolist() == [200.0, 150.0]
    assert result["cost"].tolist() == [120.0, 90.0]
    assert result["profit"].tolist() == [80.0, 60.0]
    assert result["profit_margin"].tolist() == pytest.approx([40.0, 40.0])


def test_profit_margin_is_missing_when_revenue_is_zero():
    data = pd.DataFrame({
        "quantity": [0],
        "unit_price": [100],
        "unit_cost": [20],
    })

    result = add_retail_metrics(
        data,
        quantity_column="quantity",
        unit_price_column="unit_price",
        revenue_column="revenue",
        unit_cost_column="unit_cost",
        cost_column="cost",
        profit_column="profit",
    )

    assert pd.isna(result["profit_margin"].iloc[0])


def test_retail_metrics_require_input_columns():
    data = pd.DataFrame({"quantity": [2]})

    with pytest.raises(ValueError):
        add_retail_metrics(
            data,
            quantity_column="quantity",
            revenue_column="revenue",
        )


def test_retail_metrics_do_not_overwrite_existing_columns():
    data = pd.DataFrame({
        "quantity": [2],
        "unit_price": [100],
        "revenue": [999],
    })

    with pytest.raises(ValueError):
        add_retail_metrics(
            data,
            quantity_column="quantity",
            unit_price_column="unit_price",
            revenue_column="revenue",
        )


def test_log_transform():
    data = pd.DataFrame({"sales": [0, 1, 9]})
    result = log_transform_numeric(data, "sales")

    assert result["sales_log1p"].tolist() == pytest.approx(
        np.log1p([0, 1, 9]).tolist()
    )
    assert "sales" in result.columns


def test_log_transform_rejects_negative_values():
    with pytest.raises(ValueError):
        log_transform_numeric(pd.DataFrame({"sales": [10, -1, 5]}), "sales")


def test_log_transform_preserves_missing_values():
    data = pd.DataFrame({"sales": [1, np.nan, 9]})
    result = log_transform_numeric(data, "sales")

    assert pd.isna(result["sales_log1p"].iloc[1])


def test_one_hot_encoding():
    data = pd.DataFrame({
        "region": ["North", "South", "North"],
        "sales": [100, 200, 150],
    })

    result = encode_categorical_features(data, columns=["region"])

    assert "region=North" in result.columns
    assert "region=South" in result.columns
    assert result["region=North"].tolist() == [1, 0, 1]
    assert result["sales"].tolist() == [100, 200, 150]


def test_encoding_rejects_numeric_columns():
    with pytest.raises(ValueError):
        encode_categorical_features(
            pd.DataFrame({"sales": [100, 200]}),
            columns=["sales"],
        )


def test_empty_encoding_selection_returns_copy():
    data = pd.DataFrame({"sales": [100, 200]})
    result = encode_categorical_features(data, columns=[])

    pd.testing.assert_frame_equal(result, data)
    assert result is not data


def test_feature_engineering_does_not_modify_original_data():
    data = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "sales": [100, 200],
        "region": ["North", "South"],
    })
    original = data.copy(deep=True)

    add_date_features(data, "date")
    log_transform_numeric(data, "sales")
    encode_categorical_features(data, ["region"])

    pd.testing.assert_frame_equal(data, original)
