"""
api.py – FastAPI backend for CreditLens.

Exposes the pipeline via a REST API.
Instantiates the pipeline once at startup via lifespan context.
"""

import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pipeline import CreditLensPipeline
from models.embedder import ProfileEmbedder

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("creditlens_api")

# Global pipeline instance (loaded on startup)
ml_pipeline: CreditLensPipeline | None = None
embedder: ProfileEmbedder | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_pipeline, embedder
    logger.info("Initializing ML pipeline models (this may take a moment)...")
    ml_pipeline = CreditLensPipeline()
    embedder = ProfileEmbedder()
    # Force single warm-up pass
    _ = embedder.collection_stats()
    logger.info("ML pipeline ready.")
    yield
    logger.info("Shutting down API...")


app = FastAPI(title="CreditLens API", lifespan=lifespan)

# ── middleware ───────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"[{request.method}] {request.url.path} "
        f"- Status: {response.status_code} "
        f"- Process Time: {process_time:.3f}s"
    )
    return response


# ── schemas ──────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    business_name: str
    description: str
    financial_text: str
    fetch_news: bool = True


# ── endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "CreditLens API running"}


@app.get("/similar")
def get_similar(text: str, n: int = 5):
    if not embedder:
        return {"error": "Embedder not initialized"}
    results = embedder.find_similar(text, n=n)
    return {"results": results}


@app.post("/analyze")
def analyze_risk(req: AnalyzeRequest):
    if not ml_pipeline:
        return {"error": "Pipeline not initialized"}
    
    result = ml_pipeline.analyze(
        business_name=req.business_name,
        description=req.description,
        financial_text=req.financial_text,
        fetch_news=req.fetch_news,
    )
    return result

