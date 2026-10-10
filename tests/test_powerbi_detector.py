"""Unit tests for Power BI semantic column detection."""

import numpy as np
import pandas as pd
import pytest

from src.reporting.powerbi.detector import (
    analyze_dataset_for_powerbi,
    detect_column_semantic_role,
)
from src.reporting.powerbi.models import SemanticRole


def test_detect_date_column():
    series = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])
    profile = detect_column_semantic_role(series, "Order_Date", total_rows=4)
    assert profile.semantic_role == SemanticRole.DATE
    assert profile.confidence >= 0.8


def test_detect_revenue_and_profit():
    sales = pd.Series([100.5, 200.0, 350.75, 420.1])
    prof = pd.Series([20.5, 40.0, 65.25, 80.0])

    p_sales = detect_column_semantic_role(sales, "Sales_Amount", total_rows=4)
    p_prof = detect_column_semantic_role(prof, "Net_Profit", total_rows=4)

    assert p_sales.semantic_role == SemanticRole.REVENUE
    assert p_prof.semantic_role == SemanticRole.PROFIT


def test_detect_quantity_and_cost():
    qty = pd.Series([1, 5, 10, 2])
    cost = pd.Series([50.0, 100.0, 200.0, 75.0])

    p_qty = detect_column_semantic_role(qty, "Order_Quantity", total_rows=4)
    p_cost = detect_column_semantic_role(cost, "Unit_Cost", total_rows=4)

    assert p_qty.semantic_role == SemanticRole.QUANTITY
    assert p_cost.semantic_role == SemanticRole.COST


def test_detect_categories_and_regions():
    cats = pd.Series(["Electronics", "Furniture", "Technology", "Office Supplies"])
    regs = pd.Series(["North", "South", "East", "West"])
    custs = pd.Series(["Acme Corp", "Beta Inc", "Gamma LLC", "Delta Co"])

    p_cats = detect_column_semantic_role(cats, "Product_Category", total_rows=4)
    p_regs = detect_column_semantic_role(regs, "Region", total_rows=4)
    p_custs = detect_column_semantic_role(custs, "Customer_Name", total_rows=4)

    assert p_cats.semantic_role == SemanticRole.CATEGORY
    assert p_regs.semantic_role == SemanticRole.REGION
    assert p_custs.semantic_role == SemanticRole.CUSTOMER


def test_detect_id_key():
    ids = pd.Series(["ORD-001", "ORD-002", "ORD-003", "ORD-004"])
    p_id = detect_column_semantic_role(ids, "Order_ID", total_rows=4)

    assert p_id.semantic_role == SemanticRole.TRANSACTION_ID
    assert p_id.is_key is True


def test_detect_generic_fallback():
    metric = pd.Series([12.3, 45.6, 78.9])
    dim = pd.Series(["Alpha", "Beta", "Gamma"])

    p_m = detect_column_semantic_role(metric, "Metric_X", total_rows=3)
    p_d = detect_column_semantic_role(dim, "Dimension_Y", total_rows=3)

    assert p_m.semantic_role == SemanticRole.GENERIC_NUMERIC
    assert p_d.semantic_role == SemanticRole.GENERIC_CATEGORICAL


def test_analyze_dataset_for_powerbi_full_retail():
    df = pd.DataFrame({
        "Order_Date": pd.date_range("2023-01-01", periods=10, freq="D"),
        "Order_ID": [f"ORD-{i}" for i in range(10)],
        "Customer_Name": [f"Cust {i}" for i in range(10)],
        "Region": ["North", "South"] * 5,
        "Category": ["Technology", "Furniture"] * 5,
        "Sales": [100.0 * (i + 1) for i in range(10)],
        "Profit": [20.0 * (i + 1) for i in range(10)],
        "Quantity": [1 + (i % 3) for i in range(10)],
    })

    analysis = analyze_dataset_for_powerbi(df, "Retail_Sample")
    assert analysis.row_count == 10
    assert analysis.column_count == 8
    assert analysis.date_column == "Order_Date"
    assert analysis.primary_metric_column == "Sales"
    assert len(analysis.category_columns) >= 3


def test_analyze_empty_dataframe():
    empty_df = pd.DataFrame()
    analysis = analyze_dataset_for_powerbi(empty_df, "Empty")
    assert analysis.row_count == 0
    assert analysis.column_count == 0
    assert len(analysis.columns) == 0
    assert len(analysis.notes) > 0


def test_analyze_none_dataframe_raises():
    with pytest.raises(ValueError):
        analyze_dataset_for_powerbi(None, "None")
