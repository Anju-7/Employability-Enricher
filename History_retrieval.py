import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

HISTORICAL_FILE = "historical_market_intelligence.xml"
TODAYS_FILE = "todays_market_drift.xml"

def append_new_trends_to_historical():
    """Ingests data from todays_market_drift.xml and incrementally builds 
    the 4-year cumulative ledger without overwriting past history.
    """
    if not os.path.exists(TODAYS_FILE):
        print(f"[!] Error: {TODAYS_FILE} not found. Execute fetch_news.py first.")
        return

    # 1. Initialize or load the Master Historical Ledger
    if not os.path.exists(HISTORICAL_FILE):
        print(f"[*] Initializing a brand-new historical ledger: {HISTORICAL_FILE}")
        historical_root = ET.Element("MarketIngestion", {"type": "HISTORICAL_LEDGER"})
        existing_urls = set()
    else:
        print(f"[*] Loading existing historical ledger: {HISTORICAL_FILE}")
        try:
            historical_tree = ET.parse(HISTORICAL_FILE)
            historical_root = historical_tree.getroot()
            # Track existing URLs to ensure absolute deduplication over the years
            existing_urls = {url.text.strip() for url in historical_root.findall(".//ReferenceURL") if url.text}
        except ET.ParseError:
            print("[!] Critical: Historical XML corrupted. Resetting clean master ledger backup.")
            historical_root = ET.Element("MarketIngestion", {"type": "HISTORICAL_LEDGER"})
            existing_urls = set()

    # 2. Parse today's incoming 5-day sliding window data
    todays_tree = ET.parse(TODAYS_FILE)
    todays_root = todays_tree.getroot()
    
    new_records_added = 0

    # 3. Filter and Append Unique Trends
    for item in todays_root.findall('TrendItem'):
        url_node = item.find('ReferenceURL')
        url = url_node.text.strip() if url_node is not None else ""
        
        # Only append if this specific technology instance has not been recorded before
        if url and url not in existing_urls:
            historical_root.append(item)
            existing_urls.add(url)
            new_records_added += 1

    # 4. Save the expanded ledger back to disk cleanly
    if new_records_added > 0:
        print(f"[+] Discovered {new_records_added} new unique trends! Compounding ledger...")
        
        # Format cleanly with tostring
        raw_string = ET.tostring(historical_root, encoding="utf-8")
        parsed_string = minidom.parseString(raw_string)
        pretty_xml = parsed_string.toprettyxml(indent="  ")
        
        # Clean up double line breaks caused by minidom pretty printing empty elements
        pretty_xml = "\n".join([line for line in pretty_xml.splitlines() if line.strip()])
        
        with open(HISTORICAL_FILE, "w", encoding="utf-8") as f:
            f.write(pretty_xml)
        print(f"[+] Historical ledger updated. Total Cumulative Records: {len(existing_urls)}")
    else:
        print("[+] All incoming items already exist in the 4-year baseline. Synchronization clean.")

if __name__ == "__main__":
    append_new_trends_to_historical()
