
import pandas as pd
import pytest

from src.analytics.correlation import analyze_correlations


def test_perfect_positive_correlation():
    data = pd.DataFrame({
        "sales": [10, 20, 30, 40],
        "quantity": [1, 2, 3, 4],
    })

    result = analyze_correlations(data)

    assert result.matrix["sales"]["quantity"] == pytest.approx(1.0)


def test_perfect_negative_correlation():
    data = pd.DataFrame({
        "sales": [10, 20, 30, 40],
        "returns": [4, 3, 2, 1],
    })

    result = analyze_correlations(data)

    assert result.matrix["sales"]["returns"] == pytest.approx(-1.0)


def test_spearman_method():
    data = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [10, 20, 30, 40],
    })

    result = analyze_correlations(data, method="spearman")

    assert result.method == "spearman"
    assert result.matrix["x"]["y"] == pytest.approx(1.0)


def test_constant_column_returns_none_for_pair():
    data = pd.DataFrame({
        "sales": [10, 20, 30],
        "constant": [5, 5, 5],
    })

    result = analyze_correlations(data)

    assert result.matrix["sales"]["constant"] is None


def test_missing_column_raises_error():
    data = pd.DataFrame({"sales": [10, 20, 30]})

    with pytest.raises(ValueError):
        analyze_correlations(data, columns=["unknown"])


def test_invalid_method_raises_error():
    data = pd.DataFrame({"sales": [10, 20, 30]})

    with pytest.raises(ValueError):
        analyze_correlations(data, method="invalid")


def test_input_is_not_modified():
    data = pd.DataFrame({
        "sales": [10, 20, 30],
        "quantity": [1, None, 3],
    })

    original = data.copy(deep=True)

    analyze_correlations(data)

    pd.testing.assert_frame_equal(data, original)
