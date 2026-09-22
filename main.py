import os
import json
import re
import requests
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pypdf
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
ANALYTICS_PATH = os.path.join(BASE_DIR, "market_summary_analytics.json")
VECTOR_NPY_PATH = os.path.join(BASE_DIR, "historical_vectors.npy")
HISTORICAL_JSON_PATH = os.path.join(BASE_DIR, "historical_text.json")

# Initialize SentenceTransformer DL Model globally
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def load_vector_ledger():
    """Loads pre-computed historical vectors and raw documents."""
    if os.path.exists(VECTOR_NPY_PATH) and os.path.exists(HISTORICAL_JSON_PATH):
        vectors = np.load(VECTOR_NPY_PATH)
        with open(HISTORICAL_JSON_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
        return vectors, docs
    return None, []

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

@app.post("/api/analyze-resume")
async def analyze_resume(file: UploadFile = File(...)):
    print(f"\n---> Evaluating Resume Payload: {file.filename}")

    # 1. Extract raw text from PDF
    try:
        reader = pypdf.PdfReader(file.file)
        resume_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        if not resume_text.strip():
            return {"status": "ERROR", "message": "Failed to extract text from uploaded PDF."}
    except Exception as err:
        return {"status": "ERROR", "message": f"PDF parse error: {str(err)}"}

    # 2. Load Vector Ledger & Historical Data
    historical_vectors, historical_docs = load_vector_ledger()

    # Fallback default target skills if vector DB is uninitialized
    target_skills = ["Python", "PyTorch", "FastAPI", "Docker", "Kubernetes", "CI/CD", "MLOps", "LLM", "Cloud"]

    feature_scores = {}
    detected_gaps = []
    exact_matches = []

    # 3. Deep Learning Vector Similarity Comparison
    if historical_vectors is not None and len(historical_docs) > 0:
        resume_vector = embedder.encode(resume_text, convert_to_numpy=True)
        cosine_scores = util.cos_sim(resume_vector, historical_vectors)[0].numpy()
        
        top_indices = np.argsort(cosine_scores)[-10:][::-1]
        semantic_score = float(np.mean(cosine_scores[top_indices])) * 100.0
        
        # 4. Exact Word/Phrase Match Overlap
        resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
        
        for skill in target_skills:
            if skill.lower() in resume_words or skill.lower() in resume_text.lower():
                feature_scores[skill] = 95.0
                exact_matches.append(skill)
            else:
                feature_scores[skill] = round(float(semantic_score * 0.5), 1)
                detected_gaps.append(skill)

        overall_alignment = round(min(98.0, max(30.0, (semantic_score * 0.6) + (len(exact_matches) * 5.0))), 1)
    else:
        # Fallback keyword logic if vectors aren't pre-computed
        resume_lower = resume_text.lower()
        for skill in target_skills:
            if skill.lower() in resume_lower:
                feature_scores[skill] = 90.0
                exact_matches.append(skill)
            else:
                feature_scores[skill] = 20.0
                detected_gaps.append(skill)

        match_ratio = len(exact_matches) / len(target_skills)
        overall_alignment = round(min(98.0, max(30.0, match_ratio * 100.0)), 1)

    print(f"Calculated DL Alignment Score: {overall_alignment}%")
    print(f"Exact Skills Detected: {exact_matches}")
    print(f"Detected Gaps: {detected_gaps}")

    # 5. Retrieve YouTube Recommendations for Detected Skill Gaps
    video_recommendations = []
    target_gaps = detected_gaps[:2] if detected_gaps else target_skills[:2]

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
                "title": f"Master {skill} - Production Guide",
                "videoId": "",
                "searchUrl": f"https://www.youtube.com/results?search_query={skill.replace(' ', '+')}+tutorial",
                "thumbnail": "https://img.youtube.com/vi/s3JldKoA0zw/hqdefault.jpg"
            })

    return {
        "status": "SUCCESS",
        "alignmentScore": overall_alignment,
        "featureScores": feature_scores,
        "exactMatches": exact_matches,
        "gaps": detected_gaps[:3],
        "recommendations": video_recommendations
    }