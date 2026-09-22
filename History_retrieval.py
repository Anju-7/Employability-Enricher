import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

HISTORICAL_FILE = "historical_market_intelligence.xml"
TODAYS_FILE = "todays_market_drift.xml"
TEMP_FILE = "historical_market_intelligence.tmp"

def extract_existing_urls_streaming(file_path):
    """Memory-efficient streaming extraction of existing URLs from a large XML file."""
    existing_urls = set()
    if not os.path.exists(file_path):
        return existing_urls

    try:
        # iterparse avoids loading the whole DOM into memory at once
        context = ET.iterparse(file_path, events=("end",))
        for event, elem in context:
            if elem.tag == "ReferenceURL" and elem.text:
                existing_urls.add(elem.text.strip())
            # Clear element from memory once processed
            elem.clear()
    except (ET.ParseError, Exception) as e:
        print(f"[!] Warning: Could not perform streaming read on {file_path}: {e}")
    
    return existing_urls

def append_new_trends_to_historical():
    """Ingests todays_market_drift.xml and safely updates the historical ledger."""
    if not os.path.exists(TODAYS_FILE):
        print(f"[!] Error: {TODAYS_FILE} not found. Execute fetch script first.")
        return

    # 1. Load or Initialize Master Historical Ledger
    if not os.path.exists(HISTORICAL_FILE):
        print(f"[*] Initializing a brand-new historical ledger: {HISTORICAL_FILE}")
        historical_root = ET.Element("MarketIngestion", {"type": "HISTORICAL_LEDGER"})
        existing_urls = set()
    else:
        print(f"[*] Extracting existing index from: {HISTORICAL_FILE}")
        existing_urls = extract_existing_urls_streaming(HISTORICAL_FILE)
        
        try:
            historical_tree = ET.parse(HISTORICAL_FILE)
            historical_root = historical_tree.getroot()
        except ET.ParseError:
            print("[!] Critical: Historical XML corrupted. Creating fresh master ledger.")
            historical_root = ET.Element("MarketIngestion", {"type": "HISTORICAL_LEDGER"})
            existing_urls = set()

    # 2. Parse Incoming Daily Window
    try:
        todays_tree = ET.parse(TODAYS_FILE)
        todays_root = todays_tree.getroot()
    except ET.ParseError as e:
        print(f"[!] Error parsing {TODAYS_FILE}: {e}")
        return

    new_records_added = 0

    # 3. Deduplicate and Append
    for item in todays_root.findall("TrendItem"):
        url_node = item.find("ReferenceURL")
        url = url_node.text.strip() if url_node is not None and url_node.text else ""

        if url and url not in existing_urls:
            # Clean inner whitespace for consistent serialization
            for elem in item.iter():
                if elem.text:
                    elem.text = elem.text.strip()
            
            historical_root.append(item)
            existing_urls.add(url)
            new_records_added += 1

    # 4. Safe Atomic Write to Disk
    if new_records_added > 0:
        print(f"[+] Discovered {new_records_added} new unique trends! Updating ledger...")
        
        # Serialize to temp file first to prevent partial write corruption
        try:
            # Format single item nodes safely
            raw_bytes = ET.tostring(historical_root, encoding="utf-8")
            parsed = minidom.parseString(raw_bytes)
            
            # Format and strip redundant blank lines safely
            clean_lines = [
                line for line in parsed.toprettyxml(indent="  ").splitlines() 
                if line.strip()
            ]
            
            with open(TEMP_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(clean_lines))

            # Atomic swap (overwrites target file safely)
            os.replace(TEMP_FILE, HISTORICAL_FILE)
            print(f"[+] Sync successful. Total Cumulative Records: {len(existing_urls)}")

        except Exception as e:
            print(f"[!] Critical Error during file write: {e}")
            if os.path.exists(TEMP_FILE):
                os.remove(TEMP_FILE)
    else:
        print("[+] All incoming items already exist in the historical baseline. Synchronization clean.")

if __name__ == "__main__":
    append_new_trends_to_historical()