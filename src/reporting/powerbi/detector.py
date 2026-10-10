"""Intelligent semantic detection for dynamic Power BI data modeling."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from src.reporting.powerbi.models import (
    ColumnSemanticProfile,
    PowerBiModelAnalysis,
    SemanticRole,
)


_KEYWORD_PATTERNS: dict[SemanticRole, list[str]] = {
    SemanticRole.DATE: [
        "date", "time", "timestamp", "datetime", "day", "month", "year",
        "order_date", "ship_date", "transaction_date", "invoice_date", "period", "trans_date"
    ],
    SemanticRole.REVENUE: [
        "sales", "revenue", "turnover", "gross_sales", "net_sales", "amount",
        "total_amount", "order_amount", "sales_amount", "invoice_amount", "income"
    ],
    SemanticRole.QUANTITY: [
        "quantity", "qty", "units", "units_sold", "volume", "order_quantity",
        "count", "pieces", "items", "item_count"
    ],
    SemanticRole.PROFIT: [
        "profit", "net_profit", "gross_profit", "margin", "earnings", "net_income"
    ],
    SemanticRole.COST: [
        "cost", "cogs", "expense", "expenses", "unit_cost", "total_cost",
        "purchase_price", "spending"
    ],
    SemanticRole.PRICE: [
        "price", "unit_price", "unit_rate", "rate", "selling_price", "mrp", "retail_price"
    ],
    SemanticRole.DISCOUNT: [
        "discount", "discount_amount", "discount_pct", "discount_rate", "rebate"
    ],
    SemanticRole.CATEGORY: [
        "category", "segment", "department", "sector", "division", "group",
        "category_name", "type", "class", "industry"
    ],
    SemanticRole.SUBCATEGORY: [
        "subcategory", "sub_category", "sub-category", "sub_dept", "sub_segment"
    ],
    SemanticRole.PRODUCT: [
        "product", "product_name", "item", "item_name", "sku", "description",
        "product_description", "title", "brand", "model"
    ],
    SemanticRole.REGION: [
        "region", "country", "state", "city", "territory", "province",
        "zone", "district", "market", "area", "postal_code", "zip_code", "zip", "location"
    ],
    SemanticRole.CUSTOMER: [
        "customer", "customer_name", "client", "buyer", "account_name",
        "customer_id", "client_id", "user_id", "shopper", "consumer"
    ],
    SemanticRole.TRANSACTION_ID: [
        "order_id", "invoice_id", "transaction_id", "trans_id", "receipt_no",
        "invoice_number", "order_number", "uuid", "row_id", "order_num", "trans_no"
    ],
}


def _normalize_name(name: str) -> str:
    """Normalize a column name for pattern matching."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower())
    return cleaned.strip("_")


def _is_date_series(series: pd.Series) -> bool:
    """Check if a series can be parsed as dates with sufficient validity."""
    if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_object_dtype(series.dtype):
        return False

    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return True

    valid_series = series.dropna()
    if len(valid_series) == 0:
        return False

    sample_size = min(len(valid_series), 50)
    sample = valid_series.iloc[:sample_size]

    try:
        parsed = pd.to_datetime(sample, errors="coerce", utc=True, format="mixed")
        valid_ratio = float(parsed.notna().sum()) / len(sample)
        return valid_ratio >= 0.7
    except Exception:
        return False


def _has_keyword(normalized: str, role: SemanticRole) -> bool:
    """Check if normalized name contains keywords for the given role."""
    keywords = _KEYWORD_PATTERNS.get(role, [])
    tokens = set(normalized.split("_"))
    for kw in keywords:
        if kw in tokens or kw == normalized:
            return True
        # For multi-word keywords like order_date
        if "_" in kw and kw in normalized:
            return True
    return False


def detect_column_semantic_role(
    series: pd.Series,
    name: str,
    total_rows: int,
) -> ColumnSemanticProfile:
    """Detect the semantic role and confidence for a single DataFrame column."""
    normalized = _normalize_name(name)
    tokens = set(normalized.split("_"))
    data_type_str = str(series.dtype)
    null_count = int(series.isna().sum())
    unique_count = int(series.nunique(dropna=True))

    sample_vals = [
        val for val in series.dropna().head(5).tolist()
        if not (isinstance(val, float) and np.isnan(val))
    ]

    is_numeric = pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype)
    is_date = _is_date_series(series)

    # 1. Date Detection
    if is_date:
        conf = 0.95 if _has_keyword(normalized, SemanticRole.DATE) else 0.8
        return ColumnSemanticProfile(
            name=name,
            data_type=data_type_str,
            semantic_role=SemanticRole.DATE,
            confidence=conf,
            reason="Column contains parseable date/time timestamps.",
            sample_values=sample_vals,
            null_count=null_count,
            unique_count=unique_count,
            is_key=False,
        )

    # 2. Specific Transaction ID keyword match
    if _has_keyword(normalized, SemanticRole.TRANSACTION_ID) or (
        "id" in tokens and not any(k in tokens for k in ["customer", "client", "user", "product", "item"])
    ):
        return ColumnSemanticProfile(
            name=name,
            data_type=data_type_str,
            semantic_role=SemanticRole.TRANSACTION_ID,
            confidence=0.9,
            reason="Transaction, order, or invoice identifier.",
            sample_values=sample_vals,
            null_count=null_count,
            unique_count=unique_count,
            is_key=(unique_count == (total_rows - null_count) and unique_count > 0),
        )

    # 3. Numeric specific business roles
    if is_numeric:
        if _has_keyword(normalized, SemanticRole.DISCOUNT):
            return ColumnSemanticProfile(
                name=name,
                data_type=data_type_str,
                semantic_role=SemanticRole.DISCOUNT,
                confidence=0.9,
                reason="Numeric field matching discount keywords.",
                sample_values=sample_vals,
                null_count=null_count,
                unique_count=unique_count,
            )

        if _has_keyword(normalized, SemanticRole.PROFIT):
            return ColumnSemanticProfile(
                name=name,
                data_type=data_type_str,
                semantic_role=SemanticRole.PROFIT,
                confidence=0.9,
                reason="Numeric field representing net or gross profit/margin.",
                sample_values=sample_vals,
                null_count=null_count,
                unique_count=unique_count,
            )

        if _has_keyword(normalized, SemanticRole.COST):
            return ColumnSemanticProfile(
                name=name,
                data_type=data_type_str,
                semantic_role=SemanticRole.COST,
                confidence=0.9,
                reason="Numeric field representing cost or expense.",
                sample_values=sample_vals,
                null_count=null_count,
                unique_count=unique_count,
            )

        if _has_keyword(normalized, SemanticRole.QUANTITY):
            return ColumnSemanticProfile(
                name=name,
                data_type=data_type_str,
                semantic_role=SemanticRole.QUANTITY,
                confidence=0.9,
                reason="Numeric field representing counts or unit quantities.",
                sample_values=sample_vals,
                null_count=null_count,
                unique_count=unique_count,
            )

        if _has_keyword(normalized, SemanticRole.PRICE):
            return ColumnSemanticProfile(
                name=name,
                data_type=data_type_str,
                semantic_role=SemanticRole.PRICE,
                confidence=0.9,
                reason="Numeric field representing individual unit pricing.",
                sample_values=sample_vals,
                null_count=null_count,
                unique_count=unique_count,
            )

        if _has_keyword(normalized, SemanticRole.REVENUE):
            return ColumnSemanticProfile(
                name=name,
                data_type=data_type_str,
                semantic_role=SemanticRole.REVENUE,
                confidence=0.9,
                reason="Primary monetary revenue or sales measure.",
                sample_values=sample_vals,
                null_count=null_count,
                unique_count=unique_count,
            )

        # Fallback for generic numeric
        return ColumnSemanticProfile(
            name=name,
            data_type=data_type_str,
            semantic_role=SemanticRole.GENERIC_NUMERIC,
            confidence=0.6,
            reason="Numeric measurement suitable for aggregations (SUM, AVG, MIN, MAX).",
            sample_values=sample_vals,
            null_count=null_count,
            unique_count=unique_count,
        )

    # 4. Categorical / Dimension Column Roles
    if _has_keyword(normalized, SemanticRole.CUSTOMER):
        return ColumnSemanticProfile(
            name=name,
            data_type=data_type_str,
            semantic_role=SemanticRole.CUSTOMER,
            confidence=0.85,
            reason="Customer, client, or account dimension.",
            sample_values=sample_vals,
            null_count=null_count,
            unique_count=unique_count,
        )

    if _has_keyword(normalized, SemanticRole.REGION):
        return ColumnSemanticProfile(
            name=name,
            data_type=data_type_str,
            semantic_role=SemanticRole.REGION,
            confidence=0.85,
            reason="Geographical or regional dimension.",
            sample_values=sample_vals,
            null_count=null_count,
            unique_count=unique_count,
        )

    if _has_keyword(normalized, SemanticRole.SUBCATEGORY):
        return ColumnSemanticProfile(
            name=name,
            data_type=data_type_str,
            semantic_role=SemanticRole.SUBCATEGORY,
            confidence=0.9,
            reason="Secondary hierarchical category dimension.",
            sample_values=sample_vals,
            null_count=null_count,
            unique_count=unique_count,
        )

    if _has_keyword(normalized, SemanticRole.CATEGORY):
        return ColumnSemanticProfile(
            name=name,
            data_type=data_type_str,
            semantic_role=SemanticRole.CATEGORY,
            confidence=0.9,
            reason="Primary grouping or categorical dimension.",
            sample_values=sample_vals,
            null_count=null_count,
            unique_count=unique_count,
        )

    if _has_keyword(normalized, SemanticRole.PRODUCT):
        return ColumnSemanticProfile(
            name=name,
            data_type=data_type_str,
            semantic_role=SemanticRole.PRODUCT,
            confidence=0.85,
            reason="Product or item identifier dimension.",
            sample_values=sample_vals,
            null_count=null_count,
            unique_count=unique_count,
        )

    # Low-to-medium cardinality fallback
    return ColumnSemanticProfile(
        name=name,
        data_type=data_type_str,
        semantic_role=SemanticRole.GENERIC_CATEGORICAL,
        confidence=0.6,
        reason="Categorical text or status dimension for slicing and filtering.",
        sample_values=sample_vals,
        null_count=null_count,
        unique_count=unique_count,
    )


def analyze_dataset_for_powerbi(
    df: pd.DataFrame,
    dataset_name: str = "Dataset",
) -> PowerBiModelAnalysis:
    """Analyze a DataFrame and detect all semantic column roles and structure."""
    if df is None:
        raise ValueError("DataFrame cannot be None.")

    total_rows = len(df)
    total_cols = len(df.columns)

    if total_rows == 0 or total_cols == 0:
        return PowerBiModelAnalysis(
            dataset_name=dataset_name,
            row_count=total_rows,
            column_count=total_cols,
            columns=[],
            dax_measures=[],
            visual_recommendations=[],
            notes=["The dataset is empty. Upload or clean data with rows and columns to generate Power BI assets."],
        )

    columns_profiles: list[ColumnSemanticProfile] = []
    notes: list[str] = []

    for col in df.columns:
        profile = detect_column_semantic_role(df[col], str(col), total_rows)
        columns_profiles.append(profile)

    # Find candidate primary date column
    date_cols = [c for c in columns_profiles if c.semantic_role == SemanticRole.DATE]
    selected_date_col = date_cols[0].name if date_cols else None

    # Find candidate primary metric column (Revenue -> Profit -> Quantity -> Price -> first generic numeric)
    revenue_cols = [c for c in columns_profiles if c.semantic_role == SemanticRole.REVENUE]
    profit_cols = [c for c in columns_profiles if c.semantic_role == SemanticRole.PROFIT]
    qty_cols = [c for c in columns_profiles if c.semantic_role == SemanticRole.QUANTITY]
    generic_num = [c for c in columns_profiles if c.semantic_role == SemanticRole.GENERIC_NUMERIC]

    primary_metric: str | None = None
    if revenue_cols:
        primary_metric = revenue_cols[0].name
    elif profit_cols:
        primary_metric = profit_cols[0].name
    elif qty_cols:
        primary_metric = qty_cols[0].name
    elif generic_num:
        primary_metric = generic_num[0].name

    # Identify category columns
    cat_roles = {
        SemanticRole.CATEGORY,
        SemanticRole.SUBCATEGORY,
        SemanticRole.PRODUCT,
        SemanticRole.REGION,
        SemanticRole.CUSTOMER,
        SemanticRole.GENERIC_CATEGORICAL,
    }
    category_columns = [c.name for c in columns_profiles if c.semantic_role in cat_roles]

    if not date_cols:
        notes.append("No date/time column was detected. Trend/Time-series visuals will be disabled.")
    if not primary_metric:
        notes.append("No numeric metrics were detected. Measure calculations will be limited to row counts.")

    return PowerBiModelAnalysis(
        dataset_name=dataset_name,
        row_count=total_rows,
        column_count=total_cols,
        columns=columns_profiles,
        date_column=selected_date_col,
        primary_metric_column=primary_metric,
        category_columns=category_columns,
        notes=notes,
    )
