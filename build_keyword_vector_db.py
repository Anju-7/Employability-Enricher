import os
import json
import xml.etree.ElementTree as ET
from collections import Counter
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

# Ensure required NLTK corpora & tokenizers are downloaded
NLTK_RESOURCES = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
for resource in NLTK_RESOURCES:
    try:
        nltk.download(resource, quiet=True)
    except Exception as err:
        print(f"[!] NLTK download failed for {resource}: {err}")

HISTORICAL_FILE = "historical_market_intelligence.xml"
KEYWORD_VECTORS_NPY = "keyword_vectors.npy"
KEYWORD_METADATA_JSON = "keyword_metadata.json"

def extract_and_vectorize_keywords():
    print("[+] Extracting NLTK market keywords and generating weighted vector database...")
    
    if not os.path.exists(HISTORICAL_FILE):
        print(f"[!] Historical ledger {HISTORICAL_FILE} not found.")
        return

    try:
        tree = ET.parse(HISTORICAL_FILE)
        root = tree.getroot()
    except Exception as err:
        print(f"[!] Error parsing historical XML ledger: {err}")
        return

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    stop_words.update({'using', 'based', 'system', 'data', 'application', 'framework', 'solution', 'build', 'new'})

    raw_corpus = []
    for item in root.findall("TrendItem"):
        title = item.findtext("Title", "").strip()
        summary = item.findtext("SummaryPayload", "").strip()
        tags = item.findtext("ExtractedTags", "").strip()
        raw_corpus.append(f"{title} {summary} {tags}")

    full_text = " ".join(raw_corpus).lower()
    
    # Tokenization & Lemmatization via NLTK
    tokens = word_tokenize(full_text)
    clean_tokens = [
        lemmatizer.lemmatize(word) for word in tokens 
        if word.isalnum() and word not in stop_words and len(word) > 2
    ]

    # Combine Unigrams & Bigrams
    bigrams = [' '.join(clean_tokens[i:i+2]) for i in range(len(clean_tokens)-1)]
    all_terms = clean_tokens + bigrams

    term_counts = Counter(all_terms)
    top_terms = term_counts.most_common(150)

    if not top_terms:
        print("[!] No extracted terms met frequency thresholds.")
        return

    keywords = [term for term, count in top_terms]
    counts = np.array([count for term, count in top_terms], dtype=np.float32)
    
    # Frequency Density Normalization for Importance Weights
    importance_weights = counts / np.max(counts)

    # Embed using Transformer
    print("[+] Encoding keywords with SentenceTransformer ('all-MiniLM-L6-v2')...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    keyword_vectors = embedder.encode(keywords, convert_to_numpy=True)

    # Store output database files
    np.save(KEYWORD_VECTORS_NPY, keyword_vectors)
    
    metadata = {
        "keywords": keywords,
        "importance_weights": importance_weights.tolist(),
        "total_indexed": len(keywords)
    }
    
    with open(KEYWORD_METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[+] Stored {len(keywords)} vector embeddings and NLTK metadata weights into DB successfully.")

if __name__ == "__main__":
    extract_and_vectorize_keywords()