"""Data models and structures for Power BI integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SemanticRole(str, Enum):
    """Semantic roles that columns can fulfill in a Power BI data model."""

    DATE = "Date / Time"
    REVENUE = "Revenue / Sales"
    QUANTITY = "Quantity / Volume"
    PROFIT = "Profit"
    COST = "Cost / Expense"
    PRICE = "Unit Price"
    DISCOUNT = "Discount"
    CATEGORY = "Category / Group"
    SUBCATEGORY = "Sub-Category"
    PRODUCT = "Product / Item"
    REGION = "Region / Geography"
    CUSTOMER = "Customer / Account"
    TRANSACTION_ID = "Identifier / Key"
    GENERIC_NUMERIC = "Numeric Metric"
    GENERIC_CATEGORICAL = "Categorical Dimension"


@dataclass
class ColumnSemanticProfile:
    """Detailed semantic analysis for a single column."""

    name: str
    data_type: str
    semantic_role: SemanticRole
    confidence: float
    reason: str
    sample_values: list[Any] = field(default_factory=list)
    null_count: int = 0
    unique_count: int = 0
    is_key: bool = False


@dataclass
class DaxMeasure:
    """Specification of a dynamically generated DAX measure."""

    name: str
    dax_formula: str
    display_folder: str
    description: str
    format_string: str = "#,0.00"
    business_question: str = ""


@dataclass
class VisualRecommendation:
    """Specification for a recommended Power BI report visual."""

    title: str
    visual_type: str  # Card, Line Chart, Clustered Bar Chart, Matrix, Scatter Chart, Treemap, Slicer
    fields: dict[str, str | list[str]]
    description: str
    configuration_notes: str
    suggested_page: str = "Overview Dashboard"


@dataclass
class PowerBiModelAnalysis:
    """Complete analysis and Power BI artifact package for a dataset."""

    dataset_name: str
    row_count: int
    column_count: int
    columns: list[ColumnSemanticProfile] = field(default_factory=list)
    dax_measures: list[DaxMeasure] = field(default_factory=list)
    visual_recommendations: list[VisualRecommendation] = field(default_factory=list)
    date_column: str | None = None
    primary_metric_column: str | None = None
    category_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def get_columns_by_role(self, role: SemanticRole) -> list[ColumnSemanticProfile]:
        """Return all columns matching a given semantic role."""
        return [col for col in self.columns if col.semantic_role == role]


@dataclass
class PowerBiPackage:
    """In-memory bundle containing all generated Power BI assets."""

    dataset_name: str
    clean_csv_bytes: bytes
    pbids_bytes: bytes
    powerquery_m_text: str
    dax_script_text: str
    visual_guide_md_text: str
    zip_bundle_bytes: bytes
