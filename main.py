import os
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pypdf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Target market benchmark skills to check against
TARGET_MARKET_SKILLS = [
    "Python", "PyTorch", "Docker", "Kubernetes", "FastAPI",
    "MLflow", "React", "CI/CD", "SQL", "Git"
]

@app.post("/api/analyze-resume")
async def analyze_resume(file: UploadFile = File(...)):
    # 1. Parse PDF text payload
    reader = pypdf.PdfReader(file.file)
    extracted_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()]).lower()

    # 2. Extract present skills and detect missing gaps
    found_skills = [skill for skill in TARGET_MARKET_SKILLS if skill.lower() in extracted_text]
    detected_gaps = [skill for skill in TARGET_MARKET_SKILLS if skill.lower() not in extracted_text]

    # 3. Dynamic score calculation based on matching keyword ratio
    total_skills = len(TARGET_MARKET_SKILLS)
    match_ratio = len(found_skills) / total_skills if total_skills > 0 else 0.5
    # Scale score dynamically (e.g., base 50% + up to 50% based on match ratio)
    calculated_score = round(50.0 + (match_ratio * 50.0), 1)

    # 4. Fetch YouTube recommendations for top 2 detected gaps
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
                print(f"YouTube API Error: {err}")

    return {
        "status": "SUCCESS",
        "alignmentScore": calculated_score,
        "gaps": detected_gaps[:3],  # Return top 3 missing gaps
        "recommendations": video_recommendations
    }