"""Unit tests for Dark Intelligence UI theme and overview components."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.ui.overview import _generate_synthetic_baseline, render_overview_page
from src.ui.theme import (
    CHART_COLORWAY,
    COLOR_ACCENT_TEAL,
    COLOR_BG_PRIMARY,
    apply_dark_theme,
)


def test_synthetic_baseline_generation():
    df = _generate_synthetic_baseline()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 100
    assert "Sales" in df.columns
    assert "Order_Date" in df.columns
    assert "Category" in df.columns
    assert "Product" in df.columns


def test_apply_dark_theme_to_plotly_figure():
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[10, 20, 30]))
    fig = apply_dark_theme(fig, title="Test Chart")

    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert "Test Chart" in fig.layout.title.text
    assert len(CHART_COLORWAY) >= 5


def test_overview_module_exports_and_callable():
    assert callable(render_overview_page)
