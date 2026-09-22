import os
import json
import re
import requests
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pypdf
from sentence_transformers import SentenceTransformer, util

# Ensure NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
KEYWORD_VECTORS_PATH = os.path.join(BASE_DIR, "keyword_vectors.npy")
KEYWORD_METADATA_PATH = os.path.join(BASE_DIR, "keyword_metadata.json")

# Transformer Engine
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def load_keyword_db():
    if os.path.exists(KEYWORD_VECTORS_PATH) and os.path.exists(KEYWORD_METADATA_PATH):
        vectors = np.load(KEYWORD_VECTORS_PATH)
        with open(KEYWORD_METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return vectors, meta["keywords"], np.array(meta["importance_weights"])
    return None, [], None

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

@app.post("/api/analyze-resume")
async def analyze_resume(file: UploadFile = File(...)):
    print(f"\n---> Analyzing Resume Stream: {file.filename}")

    # Extract text from PDF
    try:
        reader = pypdf.PdfReader(file.file)
        resume_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        if not resume_text.strip():
            return {"status": "ERROR", "message": "Failed to extract text from PDF."}
    except Exception as err:
        return {"status": "ERROR", "message": f"PDF parse error: {str(err)}"}

    keyword_vectors, keywords, importance_weights = load_keyword_db()

    feature_scores = {}
    verified_skills = []
    detected_gaps = []

    if keyword_vectors is not None and len(keywords) > 0:
        # Encode resume text using Transformer
        resume_vector = embedder.encode(resume_text, convert_to_numpy=True)
        
        # Calculate Cosine Similarities against NLTK Keyword Vectors
        cos_sims = util.cos_sim(resume_vector, keyword_vectors)[0].numpy()

        # Weight similarities by market importance
        weighted_scores = cos_sims * importance_weights
        
        # Weighted Score Normalized (0-100%)
        total_possible_weight = np.sum(importance_weights)
        calculated_score = (np.sum(weighted_scores) / total_possible_weight) * 100.0 * 2.5
        overall_alignment = round(float(min(98.0, max(35.0, calculated_score))), 1)

        # Categorize matches vs gaps
        for idx, kw in enumerate(keywords[:20]):  # Focus on top 20 market keywords
            score = float(cos_sims[idx] * 100)
            feature_scores[kw] = round(score, 1)
            
            if score >= 45.0 or kw.lower() in resume_text.lower():
                verified_skills.append(kw)
            else:
                detected_gaps.append(kw)
    else:
        # Fallback scoring
        overall_alignment = 75.0
        detected_gaps = ["Python", "Kubernetes", "MLOps"]
        verified_skills = ["Data Analysis", "API Design"]

    # Retrieve YouTube Remediation Paths
    video_recommendations = []
    target_gaps = detected_gaps[:2] if detected_gaps else ["MLOps", "Cloud Native"]

    for skill in target_gaps:
        fetched = False
        if YOUTUBE_API_KEY:
            yt_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={skill}+tutorial&type=video&maxResults=1&key={YOUTUBE_API_KEY}"
            try:
                res = requests.get(yt_url).json()
                if "items" in res and len(res["items"]) > 0:
                    item = res["items"][0]
                    video_recommendations.append({
                        "skill": skill,
                        "title": item["snippet"]["title"],
                        "videoId": item["id"]["videoId"],
                        "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
                        "searchUrl": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
                    })
                    fetched = True
            except Exception as err:
                print(f"YouTube Fetch Error: {err}")

        if not fetched:
            video_recommendations.append({
                "skill": skill,
                "title": f"Master {skill} - Production Essentials",
                "videoId": "",
                "searchUrl": f"https://www.youtube.com/results?search_query={skill.replace(' ', '+')}+tutorial",
                "thumbnail": "https://img.youtube.com/vi/s3JldKoA0zw/hqdefault.jpg"
            })

    return {
        "status": "SUCCESS",
        "alignmentScore": overall_alignment,
        "featureScores": feature_scores,
        "exactMatches": verified_skills[:8],
        "gaps": detected_gaps[:4],
        "recommendations": video_recommendations
    }