import datetime
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import urllib.request
import feedparser

# 1. CURATED HIGH-SIGNAL TECH FEEDS
TARGET_FEEDS = {
    "Meta Engineering": "https://engineering.fb.com/feed/",
    "Netflix Tech Blog": "https://netflixtechblog.com/feed",
    "Stack Overflow Blog": "https://stackoverflow.blog/feed/",
    "AWS Architecture": "https://aws.amazon.com/blogs/architecture/feed/",
    "InfoQ Cloud Computing": "https://feed.infoq.com/cloud-computing/news",
}

# Add a User-Agent header so blogs don't block the connection
CUSTOM_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def fetch_tech_trends_last_5_days():
    """Parses target RSS feeds, isolates entries from the LAST 5 DAYS,
    and returns a structured list of dictionaries.
    """
    # Calculate the date boundary for the last 5 days
    today = datetime.datetime.utcnow().date()
    five_days_ago = today - datetime.timedelta(days=5)
    
    print(f"[*] Initializing Ingestion Pipeline")
    print(f"[*] Date Range: {five_days_ago} to {today} (Last 5 Days)")

    todays_updates = []

    for source_name, feed_url in TARGET_FEEDS.items():
        print(f" -> Fetching feeds from: {source_name}...")
        try:
            # Fetch the feed content over HTTP with a strict 5-second timeout
            req = urllib.request.Request(feed_url, headers=CUSTOM_HEADERS)
            with urllib.request.urlopen(req, timeout=5) as response:
                html_or_xml_content = response.read()
            
            # Parse the fetched string content via feedparser
            feed = feedparser.parse(html_or_xml_content)

            for entry in feed.entries:
                published_parsed = entry.get("published_parsed") or entry.get("updated_parsed")

                if published_parsed:
                    entry_date = datetime.date(
                        published_parsed.tm_year,
                        published_parsed.tm_mon,
                        published_parsed.tm_mday,
                    )

                    # Check if the article date falls within the 5-day window
                    if five_days_ago <= entry_date <= today:
                        title = entry.get("title", "No Title")
                        link = entry.get("link", "")
                        summary = entry.get("summary", "No summary available.")

                        tags = [tag.term for tag in entry.get("tags", []) if "term" in tag]

                        todays_updates.append({
                            "source": source_name,
                            "title": title,
                            "link": link,
                            "summary": summary,
                            "tags": ", ".join(tags) if tags else "General Tech",
                        })
        except Exception as e:
            print(f" [!] Skipped/Failed {source_name}: Connection Timeout or Error")

    return todays_updates


def export_to_structured_xml(data_list, filename="todays_market_drift.xml"):
    """Structures extracted dictionary elements into a neat XML layout."""
    root = ET.Element("MarketIngestion")
    root.set("date", str(datetime.datetime.utcnow().date()))
    root.set("status", "SUCCESS")

    if not data_list:
        comment = ET.Comment("No raw data adjustments found for the specified date window.")
        root.append(comment)
    else:
        for item in data_list:
            trend_element = ET.SubElement(root, "TrendItem")

            source = ET.SubElement(trend_element, "Source")
            source.text = item["source"]

            title = ET.SubElement(trend_element, "Title")
            title.text = item["title"]

            tags = ET.SubElement(trend_element, "ExtractedTags")
            tags.text = item["tags"]

            link = ET.SubElement(trend_element, "ReferenceURL")
            link.text = item["link"]

            summary = ET.SubElement(trend_element, "SummaryPayload")
            summary.text = item["summary"]

    xml_string = ET.tostring(root, encoding="utf-8")
    parsed_string = minidom.parseString(xml_string)
    pretty_xml = parsed_string.toprettyxml(indent="  ")

    with open(filename, "w", encoding="utf-8") as xml_file:
        xml_file.write(pretty_xml)

    print(f"\n[+] Ingestion Complete. {len(data_list)} items stored in '{filename}'")


if __name__ == "__main__":
    updates = fetch_tech_trends_last_5_days()
    export_to_structured_xml(updates)
