"""
FastAPI backend for the Employability Enricher.
- Loads latest 'Production' model stage from MLflow Registry or local weights
- Caches static XML market embeddings for low latency
- Logs predictions to SQLite monitoring database
"""

import io
import os
import sqlite3
import time
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from datetime import datetime, timezone

# MUST BE SET BEFORE IMPORTING SENTENCE_TRANSFORMERS
os.environ["HF_HOME"] = os.getenv("HF_HOME", "./hf_cache")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.getenv("SENTENCE_TRANSFORMERS_HOME", "./hf_cache")

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MLFLOW_DB_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
WEIGHTS_PATH = os.environ.get("SCORER_WEIGHTS_PATH", "deep_scorer_weights.pth")
DRIFT_XML_PATH = os.environ.get("DRIFT_XML_PATH", "todays_market_drift.xml")
HISTORY_XML_PATH = os.environ.get("HISTORY_XML_PATH", "historical_market_intelligence.xml")
DB_PATH = os.environ.get("MONITORING_DB_PATH", "monitoring.sqlite3")


class EmployabilityDeepScorer(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128):
        super().__init__()
        self.drift_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2))
        self.history_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2))
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, resume_vec, drift_vec, history_vec):
        drift_features = self.drift_encoder(drift_vec)
        history_features = self.history_encoder(history_vec)
        fused_context = torch.cat((drift_features, history_features), dim=1)
        return self.fusion_network(fused_context)


class ModelState:
    embedder: SentenceTransformer | None = None
    scorer: nn.Module | None = None
    drift_emb: torch.Tensor | None = None
    history_emb: torch.Tensor | None = None


state = ModelState()


def init_monitoring_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            resume_filename TEXT,
            score REAL,
            latency_ms REAL
        )
        """
    )
    conn.commit()
    conn.close()


def log_prediction(filename: str, score: float, latency_ms: float):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (timestamp, resume_filename, score, latency_ms) VALUES (?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), filename, score, latency_ms),
    )
    conn.commit()
    conn.close()


def load_xml_text_payload(xml_path: str) -> str:
    if not os.path.exists(xml_path):
        return "Foundational data engineering architecture deployment pipeline metadata systems."
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        text_nodes = []
        for item in root.findall(".//TrendItem"):
            title = item.find("Title").text if item.find("Title") is not None else ""
            summary = item.find("SummaryPayload").text if item.find("SummaryPayload") is not None else ""
            text_nodes.append(f"{title} {summary}")
        return " ".join(text_nodes) if text_nodes else "Empty corpus placeholder context."
    except Exception:
        return "Backup standard engineering parameters."


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[*] Initializing MLOps FastAPI Inference Engine...")
    init_monitoring_db()

    # Load SentenceTransformer with relative fallback caching
    state.embedder = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=os.environ["HF_HOME"])

    # Cache static XML market embeddings at startup
    drift_text = load_xml_text_payload(DRIFT_XML_PATH)
    history_text = load_xml_text_payload(HISTORY_XML_PATH)
    state.drift_emb = torch.from_numpy(state.embedder.encode(drift_text)).unsqueeze(0).float()
    state.history_emb = torch.from_numpy(state.embedder.encode(history_text)).unsqueeze(0).float()

    # Attempt to load model from MLflow Registry 'Production' stage, fallback to local file
    try:
        mlflow.set_tracking_uri(MLFLOW_DB_URI)
        model_uri = "models:/employability-deep-scorer/Production"
        print(f"[*] Fetching Production model from MLflow Registry: {model_uri}")
        state.scorer = mlflow.pytorch.load_model(model_uri)
        print("[+] Model successfully loaded from MLflow Production Registry!")
    except Exception as e:
        print(f"[!] Could not load from MLflow Production stage ({e}). Falling back to local file.")
        scorer = EmployabilityDeepScorer(input_dim=384, hidden_dim=128)
        if os.path.exists(WEIGHTS_PATH):
            scorer.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
            print(f"[+] Loaded scorer weights from {WEIGHTS_PATH}")
        state.scorer = scorer

    state.scorer.eval()
    yield
    print("[*] Shutting down inference service.")


app = FastAPI(title="Employability Enricher API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Employability Enricher API</h1><p>POST a resume to /evaluate</p>")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": state.scorer is not None,
        "weights_found": os.path.exists(WEIGHTS_PATH),
    }


@app.post("/evaluate")
async def evaluate(resume: UploadFile = File(...)):
    if state.scorer is None or state.embedder is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    start = time.perf_counter()

    pdf_bytes = await resume.read()
    resume_text = extract_text_from_pdf_bytes(pdf_bytes)
    if not resume_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract any text from that PDF.")

    resume_emb = torch.from_numpy(state.embedder.encode(resume_text)).unsqueeze(0).float()

    with torch.no_grad():
        output = state.scorer(resume_emb, state.drift_emb, state.history_emb)

    score = round(output.item() * 100, 2)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    log_prediction(resume.filename, score, latency_ms)

    if score >= 65:
        status = "STRONG ALIGNMENT"
    elif score >= 40:
        status = "MODERATE GAP"
    else:
        status = "CRITICAL MISMATCH"

    return {
        "filename": resume.filename,
        "score": score,
        "status": status,
        "latency_ms": latency_ms,
    }


@app.get("/monitoring/recent")
async def recent_predictions(limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT timestamp, resume_filename, score, latency_ms FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]