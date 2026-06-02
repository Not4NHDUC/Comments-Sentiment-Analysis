import io
import json
import os
import pickle
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocessing.pipeline import preprocess

# ── Startup: load artifacts ──────────────────────────────────────────────────
with open("models/best_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/best_config.json") as f:
    config = json.load(f)

with open("models/label_mapping.json") as f:
    label_mapping = json.load(f)  # {"0": "Negative", "1": "Neutral", "2": "Positive"}

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Vietnamese Sentiment API", version="1.0")


# ── Pydantic models ──────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str


class BatchInput(BaseModel):
    texts: List[str]


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict


# ── Helpers ──────────────────────────────────────────────────────────────────
def _predict_one(text: str) -> PredictResponse:
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    proba = model.predict_proba(vec)[0]
    pred_idx = proba.argmax()
    label = label_mapping[str(pred_idx)]
    confidence = round(float(proba.max()), 4)
    probabilities = {
        label_mapping[str(i)]: round(float(p), 4) for i, p in enumerate(proba)
    }
    return PredictResponse(label=label, confidence=confidence, probabilities=probabilities)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "model": config["model"],
        "feature": config["feature"],
        "f1_macro": config["f1_macro_test"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: TextInput):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    return _predict_one(body.text)


@app.post("/predict_batch", response_model=List[PredictResponse])
def predict_batch(body: BatchInput):
    if len(body.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")
    return [_predict_one(t) for t in body.texts]


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if "review_text" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain a 'review_text' column")

    results = []
    for text in df["review_text"].fillna("").astype(str):
        resp = _predict_one(text)
        results.append({
            "text": text,
            "predicted_label": resp.label,
            "confidence": resp.confidence,
        })

    return {"total": len(results), "results": results}
