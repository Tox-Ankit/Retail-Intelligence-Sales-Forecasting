"""Power BI Desktop reporting and integration package."""

from src.reporting.powerbi.detector import (
    analyze_dataset_for_powerbi,
    detect_column_semantic_role,
)
from src.reporting.powerbi.generator import (
    build_powerbi_package,
    generate_dax_measures,
    generate_pbids_content,
    generate_powerquery_m_script,
    generate_visual_guide_content,
    generate_visual_recommendations,
)
from src.reporting.powerbi.models import (
    ColumnSemanticProfile,
    DaxMeasure,
    PowerBiModelAnalysis,
    PowerBiPackage,
    SemanticRole,
    VisualRecommendation,
)

__all__ = [
    "SemanticRole",
    "ColumnSemanticProfile",
    "DaxMeasure",
    "VisualRecommendation",
    "PowerBiModelAnalysis",
    "PowerBiPackage",
    "detect_column_semantic_role",
    "analyze_dataset_for_powerbi",
    "generate_dax_measures",
    "generate_visual_recommendations",
    "generate_pbids_content",
    "generate_powerquery_m_script",
    "generate_visual_guide_content",
    "build_powerbi_package",
]
