"""Unit tests for Power BI artifact generation and bundle packaging."""

import io
import json
import zipfile
import pandas as pd
import pytest

from src.reporting.powerbi.detector import analyze_dataset_for_powerbi
from src.reporting.powerbi.generator import (
    build_powerbi_package,
    generate_dax_measures,
    generate_pbids_content,
    generate_powerquery_m_script,
    generate_visual_guide_content,
    generate_visual_recommendations,
)
from src.reporting.powerbi.models import SemanticRole


@pytest.fixture
def sample_retail_df():
    return pd.DataFrame({
        "Order_Date": pd.date_range("2023-01-01", periods=15, freq="D"),
        "Order_ID": [f"ORD-{i:03d}" for i in range(15)],
        "Customer_Name": [f"Customer {i % 4}" for i in range(15)],
        "Region": ["North", "South", "East", "West", "Central"] * 3,
        "Category": ["Electronics", "Furniture", "Supplies"] * 5,
        "Sub_Category": ["Phones", "Chairs", "Paper"] * 5,
        "Sales": [150.0 + (i * 10) for i in range(15)],
        "Profit": [30.0 + (i * 2) for i in range(15)],
        "Quantity": [1 + (i % 5) for i in range(15)],
        "Unit_Cost": [100.0 + (i * 8) for i in range(15)],
    })


def test_generate_pbids_valid_json():
    pbids_bytes = generate_pbids_content("SalesData", "prepared_dataset.csv")
    doc = json.loads(pbids_bytes.decode("utf-8"))
    assert doc["version"] == "0.1"
    assert len(doc["connections"]) == 1
    assert doc["connections"][0]["connectionType"] == "File"
    assert doc["connections"][0]["details"]["path"] == "prepared_dataset.csv"
    assert doc["connections"][0]["mode"] == "Import"


def test_generate_powerquery_m_contains_types(sample_retail_df):
    m_script = generate_powerquery_m_script(sample_retail_df, "SalesTable", "sales.csv")
    assert "Source = Csv.Document" in m_script
    assert "Table.PromoteHeaders" in m_script
    assert "Table.TransformColumnTypes" in m_script
    assert "Sales" in m_script
    assert "type number" in m_script or "Int64.Type" in m_script


def test_generate_dax_measures_retail(sample_retail_df):
    analysis = analyze_dataset_for_powerbi(sample_retail_df, "SalesData")
    measures = generate_dax_measures(analysis, table_name="SalesTable")

    measure_names = [m.name for m in measures]
    assert "Total Revenue" in measure_names
    assert "Total Profit" in measure_names
    assert "Profit Margin %" in measure_names
    assert "Total Quantity Sold" in measure_names
    assert "Total Customers" in measure_names
    assert "Revenue YTD" in measure_names

    # Check DAX formula syntax
    rev_measure = next(m for m in measures if m.name == "Total Revenue")
    assert "SUM('SalesTable'[Sales])" in rev_measure.dax_formula


def test_generate_dax_measures_generic_fallback():
    generic_df = pd.DataFrame({
        "Score_A": [10.0, 20.0, 30.0],
        "Score_B": [1.5, 2.5, 3.5],
        "Group": ["A", "B", "C"],
    })
    analysis = analyze_dataset_for_powerbi(generic_df, "GenericData")
    measures = generate_dax_measures(analysis, table_name="Data")

    measure_names = [m.name for m in measures]
    assert "Total Score_A" in measure_names
    assert "Average Score_A" in measure_names


def test_generate_visual_recommendations(sample_retail_df):
    analysis = analyze_dataset_for_powerbi(sample_retail_df, "SalesData")
    analysis.dax_measures = generate_dax_measures(analysis, table_name="SalesTable")
    recs = generate_visual_recommendations(analysis, table_name="SalesTable")

    assert len(recs) >= 4
    visual_titles = [r.title for r in recs]
    assert any("KPI Header" in t for t in visual_titles)
    assert any("Over Time Trend" in t for t in visual_titles)
    assert any("Performance by Category" in t for t in visual_titles)


def test_generate_visual_guide_content(sample_retail_df):
    analysis = analyze_dataset_for_powerbi(sample_retail_df, "SalesData")
    analysis.dax_measures = generate_dax_measures(analysis, table_name="SalesTable")
    analysis.visual_recommendations = generate_visual_recommendations(analysis, table_name="SalesTable")

    guide_md = generate_visual_guide_content(analysis, table_name="SalesTable")
    assert "Microsoft Power BI Desktop" in guide_md
    assert "Quick Start" in guide_md
    assert "Executive KPI Header" in guide_md
    assert "Column Classification" in guide_md


def test_build_powerbi_package_creates_valid_zip(sample_retail_df):
    pkg = build_powerbi_package(sample_retail_df, dataset_name="TestDataset", table_name="PreparedData")

    assert pkg.dataset_name == "TestDataset"
    assert len(pkg.clean_csv_bytes) > 0
    assert len(pkg.pbids_bytes) > 0
    assert len(pkg.powerquery_m_text) > 0
    assert len(pkg.dax_script_text) > 0
    assert len(pkg.visual_guide_md_text) > 0

    # Test that the zip file can be unzipped and has all required files
    zip_stream = io.BytesIO(pkg.zip_bundle_bytes)
    with zipfile.ZipFile(zip_stream, "r") as zf:
        namelist = zf.namelist()
        assert "prepared_dataset.csv" in namelist
        assert "power_bi_connection.pbids" in namelist
        assert "power_query_transformations.m" in namelist
        assert "dax_measures.dax" in namelist
        assert "dashboard_visual_guide.md" in namelist

        csv_content = zf.read("prepared_dataset.csv").decode("utf-8-sig")
        assert "Sales" in csv_content
        assert "Profit" in csv_content


def test_edge_cases_special_chars_and_nans():
    df = pd.DataFrame({
        "Order # / ID": ["A1", "A2", None, "A4"],
        "Gross Sales ($)": [100.0, None, 300.0, 400.0],
        "Special & Region": ["East", "West", "North", None],
    })

    pkg = build_powerbi_package(df, dataset_name="Special_Chars_Dataset")
    assert pkg.dataset_name == "Special_Chars_Dataset"
    assert len(pkg.zip_bundle_bytes) > 0

    zip_stream = io.BytesIO(pkg.zip_bundle_bytes)
    with zipfile.ZipFile(zip_stream, "r") as zf:
        m_code = zf.read("power_query_transformations.m").decode("utf-8")
        assert "Gross Sales ($)" in m_code
        assert "Special & Region" in m_code


def test_edge_case_single_column():
    df = pd.DataFrame({"Values": [1, 2, 3, 4, 5]})
    pkg = build_powerbi_package(df, dataset_name="SingleCol")
    assert len(pkg.zip_bundle_bytes) > 0

