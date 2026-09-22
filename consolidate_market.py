import json
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer

HISTORICAL_FILE = "historical_market_intelligence.xml"
SUMMARY_JSON = "market_summary_analytics.json"
REPORT_MD = "market_summary_report.md"
VECTOR_NPY = "historical_vectors.npy"
HISTORICAL_JSON = "historical_text.json"

def consolidate_and_analyze():
    """Reads historical XML, aggregates trend frequencies, exports analytics, and builds vector index."""
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
    
    historical_documents = []

    for item in root.findall("TrendItem"):
        total_records += 1
        
        title = item.findtext("Title", "").strip()
        summary = item.findtext("SummaryPayload", "").strip()
        tags_raw = item.findtext("ExtractedTags", "").strip()
        source = item.findtext("Source", "Unknown").strip()
        
        sources_counter[source] += 1
        
        if tags_raw:
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
            tags_counter.update(tags)

        # Keyword analysis
        summary_lower = summary.lower()
        keywords_to_track = ["ai", "llm", "cloud", "security", "kubernetes", "database", "python"]
        for kw in keywords_to_track:
            if kw in summary_lower:
                domain_mentions[kw] += 1

        # Store doc for vector encoding
        full_doc = f"{title} {summary} {tags_raw}".strip()
        if full_doc:
            historical_documents.append(full_doc)

    # Structured analytics output
    analytics = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_historical_records": total_records,
        "top_sources": dict(sources_counter.most_common(10)),
        "top_tags": dict(tags_counter.most_common(15)),
        "keyword_occurrences": dict(domain_mentions.most_common())
    }

    # 1. Save JSON metrics
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(analytics, f, indent=2)
    print(f"[+] Exported JSON metrics to {SUMMARY_JSON}")

    # 2. Save Markdown report
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# Market Intelligence Consolidation Report\n")
        f.write(f"*Generated: {analytics['generated_at']}*\n\n")
        f.write(f"**Total Tracked Trends:** {total_records}\n\n")
        
        f.write("## Top Sources\n")
        for src, count in sources_counter.most_common(10):
            f.write(f"- **{src}**: {count} articles\n")

        f.write("\n## Top Extracted Tags\n")
        for tag, count in tags_counter.most_common(10):
            f.write(f"- `{tag}`: {count}\n")

    print(f"[+] Generated Markdown report at {REPORT_MD}")

    # 3. Generate and export Vector Database Embeddings
    if historical_documents:
        print("[+] Generating Transformer Embeddings for Historical Vector Database...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedder.encode(historical_documents, convert_to_numpy=True)
        
        np.save(VECTOR_NPY, embeddings)
        with open(HISTORICAL_JSON, "w", encoding="utf-8") as f:
            json.dump(historical_documents, f, indent=2)
            
        print(f"[+] Saved Vector Ledger to {VECTOR_NPY} and {HISTORICAL_JSON}")

if __name__ == "__main__":
    consolidate_and_analyze()