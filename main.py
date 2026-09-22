import os
import json
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pypdf
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import mlflow.pytorch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load Local Transformer Model for Semantic Embeddings
print("Loading Transformer embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Dynamic Retrieval from Historical Market Intelligence Database/JSON
MARKET_DATA_PATH = os.path.join(os.path.dirname(__file__), "market_summary_analytics.json")

def load_market_intelligence():
    """Dynamically parses historical market intelligence records."""
    if not os.path.exists(MARKET_DATA_PATH):
        print(f"Warning: {MARKET_DATA_PATH} not found. Using empty market profile.")
        return {}, {}

    with open(MARKET_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract dynamic skill descriptions and market weightings from your json stream
    market_skills = {}
    market_weights = {}

    # Extract skills directly from parsed historical market data structure
    skills_list = data.get("top_demanded_skills", []) or data.get("skills_benchmark", [])
    
    if isinstance(skills_list, list):
        for item in skills_list:
            if isinstance(item, dict):
                name = item.get("skill_name") or item.get("name")
                desc = item.get("description", f"Proficiency in {name} as required by current market demand.")
                weight = item.get("importance_weight", 1.0)
            else:
                name = str(item)
                desc = f"Demonstrated expertise and hands-on production experience in {name}."
                weight = 1.0
            
            if name:
                market_skills[name] = desc
                market_weights[name] = weight
    
    return market_skills, market_weights

DYNAMIC_MARKET_SKILLS, MARKET_WEIGHTS = load_market_intelligence()

# Pre-compute dynamic market embeddings once on server start
MARKET_EMBEDDINGS = {
    skill: embedder.encode(desc, convert_to_tensor=True)
    for skill, desc in DYNAMIC_MARKET_SKILLS.items()
}

# 3. Load MLflow PyTorch Deep Scorer Model
MODEL_URI = "models:/employability-deep-scorer/Production"
mlflow_model = None
try:
    mlflow_model = mlflow.pytorch.load_model(MODEL_URI)
    mlflow_model.eval()
    print("MLflow PyTorch model loaded successfully.")
except Exception as e:
    print(f"MLflow model load warning: {e}")

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

@app.post("/api/analyze-resume")
async def analyze_resume(file: UploadFile = File(...)):
    # 1. Parse PDF payload
    try:
        reader = pypdf.PdfReader(file.file)
        extracted_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        if not extracted_text.strip():
            return {"status": "ERROR", "message": "Could not extract readable text from PDF."}
    except Exception as err:
        return {"status": "ERROR", "message": f"PDF parsing error: {str(err)}"}

    # 2. Extract Dense Vector Embedding from PDF
    resume_embedding = embedder.encode(extracted_text, convert_to_tensor=True)

    # 3. Cross-reference dynamically retrieved market skills against PDF vector
    similarity_scores = {}
    detected_gaps = []
    weighted_scores = []

    for skill_name, market_emb in MARKET_EMBEDDINGS.items():
        # Cosine distance computation
        cos_sim = util.cos_sim(resume_embedding, market_emb).item()
        bounded_score = max(0.0, float(cos_sim))
        
        weight = MARKET_WEIGHTS.get(skill_name, 1.0)
        similarity_scores[skill_name] = round(bounded_score * 100, 1)
        weighted_scores.append(bounded_score * weight)

        # Flag skill gaps dynamically if match falls below historical threshold
        if bounded_score < 0.38:
            detected_gaps.append(skill_name)

    # 4. Neural Model Inference / Historical Weighted Scoring
    if mlflow_model:
        with torch.no_grad():
            input_tensor = torch.tensor([[s for s in similarity_scores.values()]], dtype=torch.float32)
            model_output = mlflow_model(input_tensor)
            calculated_score = round(float(model_output.item()), 1)
    else:
        # Dynamic weighted average against historical intelligence baseline
        if weighted_scores:
            mean_val = float(np.mean(weighted_scores))
            calculated_score = round(min(100.0, max(15.0, mean_val * 130.0)), 1)
        else:
            calculated_score = 75.0

    # 5. Fetch YouTube recommendations for dynamic gaps
    video_recommendations = []
    if YOUTUBE_API_KEY and detected_gaps:
        for skill in detected_gaps[:2]:
            yt_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={skill}+tutorial&type=video&maxResults=1&key={YOUTUBE_API_KEY}"
            try:
                res = requests.get(yt_url).json()
                if "items" in res and len(res["items"]) > 0:
                    item = res["items"][0]
                    video_recommendations.append({
                        "skill": skill,
                        "title": item["snippet"]["title"],
                        "videoId": item["id"]["videoId"],
                        "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"]
                    })
            except Exception as err:
                print(f"YouTube Fetch Error: {err}")

    return {
        "status": "SUCCESS",
        "alignmentScore": calculated_score,
        "featureScores": similarity_scores,
        "gaps": detected_gaps[:3],
        "recommendations": video_recommendations
    }