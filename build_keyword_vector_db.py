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

# Download NLTK data dependencies
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

HISTORICAL_FILE = "historical_market_intelligence.xml"
KEYWORD_VECTORS_NPY = "keyword_vectors.npy"
KEYWORD_METADATA_JSON = "keyword_metadata.json"

def extract_and_vectorize_keywords():
    print("[+] Extracting NLTK keywords and generating weighted vectors...")
    
    if not os.path.exists(HISTORICAL_FILE):
        print(f"[!] Historical file {HISTORICAL_FILE} not found.")
        return

    try:
        tree = ET.parse(HISTORICAL_FILE)
        root = tree.getroot()
    except Exception as e:
        print(f"[!] Error parsing XML: {e}")
        return

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    additional_stops = {'using', 'based', 'system', 'data', 'application', 'framework', 'solution', 'build'}
    stop_words.update(additional_stops)

    raw_corpus = []
    for item in root.findall("TrendItem"):
        title = item.findtext("Title", "").strip()
        summary = item.findtext("SummaryPayload", "").strip()
        tags = item.findtext("ExtractedTags", "").strip()
        raw_corpus.append(f"{title} {summary} {tags}")

    full_text = " ".join(raw_corpus).lower()
    
    # NLTK Tokenization & Cleaning
    tokens = word_tokenize(full_text)
    clean_tokens = [
        lemmatizer.lemmatize(word) for word in tokens 
        if word.isalnum() and word not in stop_words and len(word) > 2
    ]

    # Generate Unigrams & Bigrams
    bigrams = [' '.join(clean_tokens[i:i+2]) for i in range(len(clean_tokens)-1)]
    all_terms = clean_tokens + bigrams

    term_counts = Counter(all_terms)
    top_terms = term_counts.most_common(200)

    if not top_terms:
        print("[!] No valid terms extracted.")
        return

    keywords = [term for term, count in top_terms]
    counts = np.array([count for term, count in top_terms], dtype=np.float32)
    
    # Calculate term importance weights (Normalized Frequency Density)
    importance_weights = counts / np.max(counts)

    # Embed keywords using SentenceTransformer
    print("[+] Encoding keywords with SentenceTransformer ('all-MiniLM-L6-v2')...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    keyword_vectors = embedder.encode(keywords, convert_to_numpy=True)

    # Save outputs to DB files
    np.save(KEYWORD_VECTORS_NPY, keyword_vectors)
    
    metadata = {
        "keywords": keywords,
        "importance_weights": importance_weights.tolist(),
        "total_terms_indexed": len(keywords)
    }
    
    with open(KEYWORD_METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[+] Successfully stored {len(keywords)} keyword vectors & importance weights in database.")

if __name__ == "__main__":
    extract_and_vectorize_keywords()