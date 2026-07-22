import os
import xml.etree.ElementTree as ET
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

# Crucial fix to ensure the script looks at your D: drive cache folder
os.environ["HF_HOME"] = r"D:\hf_cache"

def extract_text_from_pdf(pdf_path):
    """Extracts raw text content from the candidate's resume PDF."""
    if not os.path.exists(pdf_path):
        print(f"[!] Error: Resume file '{pdf_path}' not found.")
        return ""
    
    print(f"[*] Extracting text from resume: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def parse_scraped_market_trends(xml_filename="todays_market_drift.xml"):
    """Extracts and combines text payloads from your completed web scraper output."""
    if not os.path.exists(xml_filename):
        print(f"[!] Error: {xml_filename} not found. Run your scraper first.")
        return ""
    
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    
    combined_market_text = []
    print(f"[*] Extracting targets from live market asset: {xml_filename}")
    
    for item in root.findall('TrendItem'):
        title = item.find('Title').text if item.find('Title') is not None else ""
        summary = item.find('SummaryPayload').text if item.find('SummaryPayload') is not None else ""
        tags = item.find('ExtractedTags').text if item.find('ExtractedTags') is not None else ""
        
        # Consolidate trends into a dense context baseline for comparison
        combined_market_text.append(f"{title} ({tags}): {summary}")
        
    return " ".join(combined_market_text)

def calculate_compatibility(resume_path, xml_filename="todays_market_drift.xml"):
    """Vectorizes the inputs and computes a mathematical alignment score using Cosine Similarity."""
    # 1. Ingest text streams
    resume_text = extract_text_from_pdf(resume_path)
    market_text = parse_scraped_market_trends(xml_filename)
    
    if not resume_text or not market_text:
        print("[!] Evaluation halted: Missing baseline inputs.")
        return

    # 2. Load the locally installed model (uses the D:\hf_cache folder instantly)
    print("[*] Loading cached Vector Transformer Model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Process text into mathematical coordinates (Inference)
    print("[*] Generating high-dimensional semantic vectors...")
    resume_vector = model.encode(resume_text)
    market_vector = model.encode(market_text)
    
    # 4. Matrix Similarity Calculation (Cosine Similarity)
    dot_product = np.dot(resume_vector, market_vector)
    norm_resume = np.linalg.norm(resume_vector)
    norm_market = np.linalg.norm(market_vector)
    
    similarity_score = dot_product / (norm_resume * norm_market)
    
    # Scale from vector similarity bounds to a clean 0 - 100% metric
    percentage_score = round(max(0, similarity_score) * 100, 2)
    
    print("\n" + "="*50)
    print("         EMPLOYABILITY ENRICHER METRICS         ")
    print("="*50)
    print(f"Target Baseline:  Live Scraped Corporate Trends")
    print(f"Candidate Profile: {os.path.basename(resume_path)}")
    print(f"Vector Space Match Alignment: {percentage_score}%")
    print("="*50)
    
    if percentage_score >= 65:
        print("[Status]: STRONG ALIGNMENT. Profile matches active market trends.")
    elif percentage_score >= 40:
        print("[Status]: MODERATE GAP. Skill drift detected. Upskilling recommended.")
    else:
        print("[Status]: CRITICAL MISMATCH. Profile requires targeted training paths.")
    print("="*50)
    
    return percentage_score

if __name__ == "__main__":
    # Put a technical resume PDF in your D:\PBL_MLOPS folder 
    # and change "my_resume.pdf" below to match its name!
    calculate_compatibility(resume_path="D:\Anj cert\RESUME_ANJANA.pdf")
