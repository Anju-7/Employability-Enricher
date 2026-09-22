import json
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime

HISTORICAL_FILE = "historical_market_intelligence.xml"
SUMMARY_JSON = "market_summary_analytics.json"
REPORT_MD = "market_summary_report.md"

def consolidate_and_analyze():
    """Reads historical XML, aggregates trend frequencies, and exports analytics."""
    try:
        tree = ET.parse(HISTORICAL_FILE)
        root = tree.getroot()
    except Exception as e:
        print(f"[!] Error reading {HISTORICAL_FILE}: {e}")
        return

    total_records = 0
    sources_counter = Counter()
    tags_counter = Counter()
    domain_mentions = Counter()

    for item in root.findall("TrendItem"):
        total_records += 1
        
        # Source breakdown
        source = item.findtext("Source", "Unknown").strip()
        sources_counter[source] += 1
        
        # Tag breakdown
        tags_raw = item.findtext("ExtractedTags", "")
        if tags_raw:
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
            tags_counter.update(tags)

        # Basic keyword analysis in summaries
        summary = item.findtext("SummaryPayload", "").lower()
        keywords_to_track = ["ai", "llm", "cloud", "security", "kubernetes", "database", "python"]
        for kw in keywords_to_track:
            if kw in summary:
                domain_mentions[kw] += 1

    # Structured output object
    analytics = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_historical_records": total_records,
        "top_sources": dict(sources_counter.most_common(10)),
        "top_tags": dict(tags_counter.most_common(15)),
        "keyword_occurrences": dict(domain_mentions.most_common())
    }

    # Save JSON metrics
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(analytics, f, indent=2)
    print(f"[+] Exported JSON metrics to {SUMMARY_JSON}")

    # Save human-readable Markdown report
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(f"# Market Intelligence Consolidation Report\n")
        f.write(f"*Generated: {analytics['generated_at']}*\n\n")
        f.write(f"**Total Tracked Trends:** {total_records}\n\n")
        
        f.write("## Top Sources\n")
        for src, count in sources_counter.most_common(10):
            f.write(f"- **{src}**: {count} articles\n")

        f.write("\n## Top Extracted Tags\n")
        for tag, count in tags_counter.most_common(10):
            f.write(f"- `{tag}`: {count}\n")

    print(f"[+] Generated Markdown report at {REPORT_MD}")

if __name__ == "__main__":
    consolidate_and_analyze()