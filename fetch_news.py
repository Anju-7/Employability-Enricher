import datetime
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import urllib.request
import feedparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutil import parser as date_parser

TARGET_FEEDS = {
    # === AI / LLM / MACHINE LEARNING ===
    "OpenAI Blog": "https://openai.com/feed.xml",
    "DeepMind Blog": "https://deepmind.google/feed.xml",
    "Hugging Face Blog": "https://huggingface.co/feed.xml",
    "Weights & Biases": "https://wandb.ai/blog/feed.xml",
    "Papers with Code": "https://paperswithcode.com/rss/",
    "Anthropic Blog": "https://www.anthropic.com/feed.xml",
    "Stability AI": "https://stability.ai/blog/feed",
    "Scale AI": "https://scale.com/blog/feed.xml",
    
    # === DATA ENGINEERING / ANALYTICS ===
    "dbt Blog": "https://blog.getdbt.com/feed.xml",
    "Databricks Blog": "https://www.databricks.com/blog/feed",
    "Apache Airflow": "https://airflow.apache.org/feed.xml",
    "Great Expectations": "https://greatexpectations.io/blog/feed.xml",
    "Fivetran Blog": "https://fivetran.com/feed.xml",
    "Census Blog": "https://www.getcensus.com/blog/feed.xml",
    "Meltano Blog": "https://meltano.com/blog/feed.xml",
    "Iceberg Blog": "https://iceberg.apache.org/feed.xml",
    "Delta Lake": "https://delta.io/feed.xml",
    "Prefect Blog": "https://www.prefect.io/blog/feed.xml",
    
    # === CLOUD PLATFORMS ===
    "AWS Architecture": "https://aws.amazon.com/blogs/architecture/feed/",
    "AWS Machine Learning": "https://aws.amazon.com/blogs/machine-learning/feed/",
    "GCP Blog": "https://cloud.google.com/blog/feed.xml",
    "Google Cloud AI": "https://cloud.google.com/blog/topics/ai-machine-learning/feed.xml",
    "Azure Blog": "https://azure.microsoft.com/en-us/blog/feed/",
    
    # === INFRASTRUCTURE / DEVOPS ===
    "Docker Blog": "https://www.docker.com/blog/feed.xml",
    "Kubernetes Blog": "https://kubernetes.io/feed.xml",
    "HashiCorp Blog": "https://www.hashicorp.com/feed.xml",
    "Grafana Blog": "https://grafana.com/blog/feed.xml",
    "Prometheus Blog": "https://prometheus.io/feed.xml",
    "Istio Blog": "https://istio.io/feed.xml",
    
    # === TECH BLOGS (Major Companies) ===
    "Netflix Tech Blog": "https://netflixtechblog.com/feed",
    "Meta Engineering": "https://engineering.fb.com/feed/",
    "Google Research": "https://research.google/blog/feed.xml",
    "Apple Machine Learning": "https://machinelearning.apple.com/feed.xml",
    "LinkedIn Engineering": "https://engineering.linkedin.com/feed",
    "Uber Engineering": "https://eng.uber.com/feed.xml",
    "Airbnb Engineering": "https://airbnb.engineering/feed.xml",
    "Stripe Blog": "https://stripe.com/blog/feed.xml",
    "Shopify Engineering": "https://shopify.engineering/feed.xml",
    "Lyft Engineering": "https://eng.lyft.com/feed.xml",
    
    # === SOFTWARE ENGINEERING / ARCHITECTURE ===
    "Stack Overflow Blog": "https://stackoverflow.blog/feed/",
    "InfoQ": "https://feed.infoq.com/",
    "Martin Fowler": "https://martinfowler.com/feed.atom",
    "High Scalability": "http://highscalability.com/feed.rss",
    "ACM Queue": "https://queue.acm.org/feeds/queue.xml",
    "ThoughtWorks Technology Radar": "https://www.thoughtworks.com/feed.xml",
    
    # === RESEARCH / ACADEMIC ===
    "ArXiv AI": "http://arxiv.org/rss/cs.AI",
    "ArXiv ML": "http://arxiv.org/rss/stat.ML",
    "ArXiv LG": "http://arxiv.org/rss/cs.LG",
    "Towards Data Science": "https://towardsdatascience.com/feed",
    
    # === OBSERVABILITY / MONITORING ===
    "DataDog Blog": "https://www.datadoghq.com/blog/feed.xml",
    "New Relic Blog": "https://blog.newrelic.com/feed/",
    "Splunk Blog": "https://www.splunk.com/en_us/feed.xml",
    "Honeycomb Blog": "https://www.honeycomb.io/blog/feed.xml",
}

CUSTOM_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def parse_entry_date(entry):
    """Robust date parser supporting structured feed tuples and raw strings."""
    parsed_struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed_struct:
        return datetime.date(parsed_struct.tm_year, parsed_struct.tm_mon, parsed_struct.tm_mday)
    
    raw_date = entry.get("published") or entry.get("updated")
    if raw_date:
        try:
            return date_parser.parse(raw_date).date()
        except Exception:
            pass
    return None

def fetch_single_feed(source_name, feed_url, five_days_ago, today_date):
    """Fetches and parses a single feed thread-safely."""
    articles = []
    try:
        req = urllib.request.Request(feed_url, headers=CUSTOM_HEADERS)
        with urllib.request.urlopen(req, timeout=8) as response:
            content = response.read()
        
        feed = feedparser.parse(content)
        for entry in feed.entries:
            entry_date = parse_entry_date(entry)
            if entry_date and (five_days_ago <= entry_date <= today_date):
                title = entry.get("title", "No Title").strip()
                link = entry.get("link", "").strip()
                summary = entry.get("summary", "No summary available.").strip()[:300]
                
                # Robust tag extraction
                tags = []
                for tag in entry.get("tags", []):
                    if isinstance(tag, dict) and "term" in tag:
                        tags.append(tag["term"])
                    elif isinstance(tag, str):
                        tags.append(tag)

                articles.append({
                    "source": source_name,
                    "title": title,
                    "link": link,
                    "summary": summary,
                    "tags": ", ".join(tags) if tags else "General Tech",
                })
        return source_name, articles, None
    except Exception as e:
        return source_name, [], str(e)

def fetch_tech_trends_concurrent(max_workers=10):
    """Executes parallel fetches across all target RSS feeds."""
    today = datetime.datetime.utcnow().date()
    five_days_ago = today - datetime.timedelta(days=5)
    
    all_articles = []
    successful = 0
    failed = 0

    print(f"[*] Starting concurrent ingestion across {len(TARGET_FEEDS)} feeds...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_single_feed, name, url, five_days_ago, today): name 
            for name, url in TARGET_FEEDS.items()
        }
        
        for future in as_completed(futures):
            source_name, articles, error = future.result()
            if error:
                failed += 1
                print(f"  [✗] {source_name:35} ERROR: {error}")
            else:
                successful += 1
                all_articles.extend(articles)
                print(f"  [✓] {source_name:35} Fetched {len(articles)} item(s)")

    print(f"\n[+] Ingestion Complete: {successful} successful, {failed} failed. Total Articles: {len(all_articles)}")
    return all_articles

def export_to_xml(data_list, filename="todays_market_drift.xml"):
    """Writes standardized XML format."""
    root = ET.Element("MarketIngestion")
    root.set("date", str(datetime.datetime.utcnow().date()))
    root.set("record_count", str(len(data_list)))
    root.set("status", "SUCCESS" if data_list else "EMPTY")

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

    xml_bytes = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(xml_bytes)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(parsed.toprettyxml(indent="  "))

if __name__ == "__main__":
    results = fetch_tech_trends_concurrent(max_workers=12)
    export_to_xml(results)