import os
import requests
from fastapi import FastAPI, UploadFile, File
import PyPDF2
import mlflow.pytorch

app = FastAPI()

# Load latest production model dynamically
MODEL_URI = "models:/employability-deep-scorer/Production"
model = mlflow.pytorch.load_model(MODEL_URI)

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

@app.post("/api/analyze-resume")
async def analyze_resume(file: UploadFile = File(...)):
    # 1. Parse PDF text payload
    reader = PyPDF2.PdfReader(file.file)
    extracted_text = " ".join([page.extract_text() for page in reader.pages])

    # 2. Run model inference & compute missing skill gaps
    # (Placeholder logic for gap detection)
    detected_gaps = ["Docker Containerization", "Kubernetes Scaling"]

    # 3. Dynamic YouTube Video Fetching
    video_recommendations = []
    for skill in detected_gaps:
        yt_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={skill}+tutorial&type=video&maxResults=1&key={YOUTUBE_API_KEY}"
        res = requests.get(yt_url).json()
        if "items" in res and len(res["items"]) > 0:
            item = res["items"][0]
            video_recommendations.append({
                "skill": skill,
                "title": item["snippet"]["title"],
                "videoId": item["id"]["videoId"],
                "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"]
            })

    return {
        "status": "SUCCESS",
        "alignmentScore": 88.5,
        "gaps": detected_gaps,
        "recommendations": video_recommendations
    }