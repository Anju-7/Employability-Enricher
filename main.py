import os
import json
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pypdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MARKET_DATA_PATH = os.path.join(os.path.dirname(__file__), "market_summary_analytics.json")

def load_historical_market_intelligence():
    """
    Parses market_summary_analytics.json dynamically and extracts
    all skill vectors, market descriptions, and historical demand metrics.
    """
    if not os.path.exists(MARKET_DATA_PATH):
        print(f"Warning: {MARKET_DATA_PATH} not found.")
        return [], []

    with open(MARKET_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten JSON data to extract raw skill descriptions
    skill_names = []
    skill_documents = []

    # Recursively locate list/dictionary records in historical market JSON
    records = []
    if isinstance(data, dict):
        for key in ["skills", "top_demanded_skills", "market_skills", "technologies", "analytics"]:
            if key in data and isinstance(data[key], list):
                records = data[key]
                break
        if not records:
            records = [data]
    elif isinstance(data, list):
        records = data

    for item in records:
        if isinstance(item, dict):
            name = item.get("skill_name") or item.get("name") or item.get("title") or "Technical Requirement"
            desc = item.get("description") or item.get("summary") or f"Hands-on expertise and production usage of {name}"
            if name:
                skill_names.append(name)
                skill_documents.append(f"{name} {desc}")
        elif isinstance(item, str):
            skill_names.append(item)
            skill_documents.append(f"{item} production level proficiency and experience")

    return skill_names, skill_documents

SKILL_NAMES, SKILL_DOCUMENTS = load_historical_market_intelligence()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

@app.post("/api/analyze-resume")
async def analyze_resume(file: UploadFile = File(...)):
    print(f"\n---> Analyzing Payload: {file.filename}")

    # 1. Extract raw text from PDF payload
    try:
        reader = pypdf.PdfReader(file.file)
        resume_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        if not resume_text.strip():
            return {"status": "ERROR", "message": "Failed to extract text from uploaded PDF."}
    except Exception as err:
        return {"status": "ERROR", "message": f"PDF parse error: {str(err)}"}

    # 2. Reload market intelligence dynamically if updated
    skill_names, skill_documents = load_historical_market_intelligence()
    
    if not skill_documents:
        # Fallback if market intelligence file is empty
        skill_names = ["Docker", "Kubernetes", "PyTorch", "FastAPI", "CI/CD"]
        skill_documents = [f"{s} production level experience and skills" for s in skill_names]

    # 3. Apply TF-IDF Retrieval Algorithm over Historical Market Intelligence
    corpus = [resume_text] + skill_documents
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Calculate Cosine Similarity between Resume (Index 0) and Market Skills (Index 1..N)
    resume_vector = tfidf_matrix[0]
    market_vectors = tfidf_matrix[1:]
    
    similarity_scores = cosine_similarity(resume_vector, market_vectors)[0]

    # 4. Map similarity scores back to Market Intelligence Skills
    feature_scores = {}
    detected_gaps = []
    matches = []

    for idx, score in enumerate(similarity_scores):
        skill = skill_names[idx]
        match_percentage = float(score * 100)
        feature_scores[skill] = round(match_percentage, 1)

        # Gap detection threshold based on TF-IDF cosine score
        if score < 0.12:
            detected_gaps.append(skill)
        else:
            matches.append(score)

    # Overall alignment score calculated directly from matched TF-IDF scores
    if matches:
        overall_score = round(min(98.0, max(35.0, (sum(matches) / len(skill_documents)) * 300.0 + 40.0)), 1)
    else:
        overall_score = 32.5

    print(f"Calculated TF-IDF Alignment Score: {overall_score}%")
    print(f"Detected Gaps: {detected_gaps}")

    # 5. Retrieve YouTube Recommendations for Detected Gaps
    video_recommendations = []
    target_gaps = detected_gaps[:2] if detected_gaps else skill_names[:2]

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

        # Fallback YouTube Search Link if API key is absent
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
        "alignmentScore": overall_score,
        "featureScores": feature_scores,
        "gaps": detected_gaps[:3],
        "recommendations": video_recommendations
    }