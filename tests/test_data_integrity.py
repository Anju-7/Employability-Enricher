import os
import xml.etree.ElementTree as ET
import pytest

DATA_FILES = [
    "todays_market_drift.xml",
    "historical_market_intelligence.xml",
]

@pytest.mark.parametrize("file_path", DATA_FILES)
def test_xml_files_exist_and_not_empty(file_path):
    """Ensure generated XML data files exist and are greater than 0 bytes."""
    assert os.path.exists(file_path), f"Missing data file: {file_path}"
    assert os.path.getsize(file_path) > 0, f"Data file is empty (0 bytes): {file_path}"

@pytest.mark.parametrize("file_path", DATA_FILES)
def test_xml_structure_is_valid(file_path):
    """Ensure the XML file is well-formed and contains record elements."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Verify the root node has child elements (articles/records)
    assert len(root) > 0, f"XML file {file_path} contains a valid root but 0 records."

def test_analytics_json_exists_and_valid():
    """Ensure consolidated analytics JSON exists and has valid content."""
    import json
    json_path = "market_summary_analytics.json"
    assert os.path.exists(json_path), f"Missing analytics JSON file: {json_path}"
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert isinstance(data, dict), "Analytics JSON root should be a JSON object"