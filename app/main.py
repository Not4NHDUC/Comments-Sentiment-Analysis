import io
import json
import os
import pickle
import re
import sys

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocessing.pipeline import preprocess

# ── Startup: load TF-IDF artifacts ───────────────────────────────────────────
with open("models/best_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/best_config.json") as f:
    config = json.load(f)

with open("models/label_mapping.json") as f:
    label_mapping = json.load(f)

# ── Startup: load PhoBERT (if available) ─────────────────────────────────────
phobert_tokenizer = None
phobert_model = None
phobert_label_mapping = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists("models/phobert_config.json"):
    phobert_tokenizer = AutoTokenizer.from_pretrained("models/phobert")
    phobert_model = AutoModelForSequenceClassification.from_pretrained("models/phobert")
    phobert_model.eval()
    phobert_model = phobert_model.to(device)
    with open("models/phobert_label_mapping.json") as f:
        phobert_label_mapping = json.load(f)

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Vietnamese Sentiment API", version="1.0")


# ── Pydantic models ──────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str
    model_type: str = "tfidf"


class BatchInput(BaseModel):
    texts: List[str]


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict


# ── Helpers ──────────────────────────────────────────────────────────────────
def check_rating_pattern(text: str):
    text_lower = text.strip().lower()
    one_star = r'^(1\s*sao|1\s*star|1/5|1\s*điểm|một\s*sao)[\s\.,!]*$'
    two_star = r'^(2\s*sao|2\s*star|2/5|hai\s*sao)[\s\.,!]*$'
    five_star = r'^(5\s*sao|5\s*star|5/5|năm\s*sao)[\s\.,!]*$'
    four_star = r'^(4\s*sao|4\s*star|4/5|bốn\s*sao)[\s\.,!]*$'
    if re.match(one_star, text_lower) or re.match(two_star, text_lower):
        return "Negative"
    if re.match(five_star, text_lower) or re.match(four_star, text_lower):
        return "Positive"
    return None


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


def _predict_one_phobert(text: str) -> PredictResponse:
    inputs = phobert_tokenizer(
        text, return_tensors="pt", max_length=128, truncation=True, padding=True
    )
    with torch.no_grad():
        logits = phobert_model(**{k: v.to(device) for k, v in inputs.items()}).logits
    proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(proba.argmax())
    label = phobert_label_mapping[str(pred_idx)]
    confidence = round(float(proba.max()), 4)
    probabilities = {
        phobert_label_mapping[str(i)]: round(float(p), 4) for i, p in enumerate(proba)
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
        "phobert_available": phobert_model is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: TextInput):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    early = check_rating_pattern(body.text)
    if early is not None:
        return PredictResponse(label=early, confidence=1.0, probabilities={early: 1.0})
    if body.model_type == "phobert":
        if phobert_model is None:
            raise HTTPException(status_code=503, detail="PhoBERT model not loaded")
        return _predict_one_phobert(body.text)
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
