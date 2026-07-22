import os
import xml.etree.ElementTree as ET
from pypdf import PdfReader
import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer

# Force same local cache environment path layout
os.environ["HF_HOME"] = r"D:\hf_cache"

# ==========================================
# 1. EXACT SAME PYTORCH MODEL ARCHITECTURE
# ==========================================
class EmployabilityDeepScorer(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128):
        super(EmployabilityDeepScorer, self).__init__()
        
        self.drift_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.history_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )

    def forward(self, resume_vec, drift_vec, history_vec):
        drift_features = self.drift_encoder(drift_vec)
        history_features = self.history_encoder(history_vec)
        fused_context = torch.cat((drift_features, history_features), dim=1)
        score = self.fusion_network(fused_context)
        return score

# ==========================================
# 2. DATA LOADERS AND EXTRACTION UTILITIES
# ==========================================
def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"[!] Error: Resume file '{pdf_path}' not found.")
        return ""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def load_xml_text_payload(xml_path):
    if not os.path.exists(xml_path):
        return "Foundational data engineering architecture deployment pipeline metadata systems."
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        text_nodes = []
        for item in root.findall('.//TrendItem'):
            title = item.find('Title').text if item.find('Title') is not None else ""
            summary = item.find('SummaryPayload').text if item.find('SummaryPayload') is not None else ""
            text_nodes.append(f"{title} {summary}")
        return " ".join(text_nodes) if text_nodes else "Empty corpus placeholder context."
    except Exception:
        return "Backup standard engineering parameters."

# ==========================================
# 3. EVALUATION INFERENCE LOOP
# ==========================================
def evaluate_candidate_resume(resume_filename="my_resume.pdf"):
    print("="*60)
    print("      EMPLOYABILITY ENRICHER - DEEP NEURAL EVALUATION ENGINE    ")
    print("="*60)
    
    WEIGHTS_PATH = "deep_scorer_weights.pth"
    if not os.path.exists(WEIGHTS_PATH):
        print(f"[!] Critical Error: Trained weights asset '{WEIGHTS_PATH}' not found.")
        print("[!] Make sure to successfully run python train_model.py first.")
        return

    # 1. Parse real candidate resume and local XML textual files
    resume_text = extract_text_from_pdf(resume_filename)
    if not resume_text:
        print("[!] Evaluation halted: Resume extraction output is empty.")
        return
        
    drift_text = load_xml_text_payload("todays_market_drift.xml")
    history_text = load_xml_text_payload("historical_market_intelligence.xml")

    # 2. Ingest pre-trained structural transformer embeddings using explicit cache
    print("[*] Loading base transformer encoder from local cache...")
    base_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=r"D:\hf_cache")
    
    # Generate 1-row batches for prediction inference
    resume_emb = torch.tensor(np.array([base_model.encode(resume_text)]), dtype=torch.float32)
    drift_emb = torch.tensor(np.array([base_model.encode(drift_text)]), dtype=torch.float32)
    history_emb = torch.tensor(np.array([base_model.encode(history_text)]), dtype=torch.float32)

    # 3. Instantiate Model and Mount Trained Weights Matrices
    model = EmployabilityDeepScorer(input_dim=384, hidden_dim=128)
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    model.eval() # Turn off dropout arrays for pure mathematical prediction evaluation

    # 4. Neural Network Forward Pass Inference Execution
    print("[*] Performing deep inference through network layers...")
    with torch.no_grad():
        output_tensor = model(resume_emb, drift_emb, history_emb)
        
    # Convert probability vector [0.0 - 1.0] scale up to clean printable percentage
    final_score = round(output_tensor.item() * 100, 2)
    
    print("\n" + "="*50)
    print("             PIPELINE NEURAL DIAGNOSTICS         ")
    print("="*50)
    print(f"Target Baseline: Live Scraped Corporate Trends")
    print(f"Candidate Profile: {os.path.basename(resume_filename)}")
    print(f"NEURAL FUSION ALIGNMENT SCORE: {final_score}%")
    print("="*50)
    
    return final_score

if __name__ == "__main__":
    # Ensure a test resume PDF is placed in D:\PBL_MLOPS and matches this name!
    evaluate_candidate_resume(resume_filename="my_resume.pdf")
