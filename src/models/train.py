import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    print("[train] xgboost not installed — XGBoost model will be skipped")

warnings.filterwarnings("ignore")

DATA_PATH = "data/processed/clean_data.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"
RANDOM_STATE = 42

# ── Step 1: Load data ────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df = df.dropna(subset=["text_clean"])
df = df[df["text_clean"].str.strip() != ""]

X = df["text_clean"].astype(str)
y = df["label"]

le = LabelEncoder()
y_enc = le.fit_transform(y)  # alphabetical: Negative=0, Neutral=1, Positive=2

os.makedirs(MODEL_DIR, exist_ok=True)
label_mapping = {int(i): cls for i, cls in enumerate(le.classes_)}
with open(os.path.join(MODEL_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, ensure_ascii=False, indent=2)

print(f"[train] Loaded {len(df)} samples | classes: {label_mapping}")

# ── Step 2: Split ────────────────────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_enc, test_size=0.30, random_state=RANDOM_STATE, stratify=y_enc
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)
print(
    f"[train] Split: train={len(X_train)} | val={len(X_val)} | test={len(X_test)}"
)

# ── Step 3: Feature extractors ───────────────────────────────────────────────
feature_extractors = {
    "BoW": CountVectorizer(max_features=10000, ngram_range=(1, 1)),
    "TF-IDF": TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2),
}

# ── Step 4: Models ───────────────────────────────────────────────────────────
def _build_models():
    models = {
        "NaiveBayes": MultinomialNB(),
        "SVM": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=2000)
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if _XGB_AVAILABLE:
        xgb_kwargs = dict(
            n_estimators=200,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            scale_pos_weight=1,
        )
        # use_label_encoder removed in XGBoost ≥ 1.6
        try:
            models["XGBoost"] = XGBClassifier(
                use_label_encoder=False, **xgb_kwargs
            )
        except TypeError:
            models["XGBoost"] = XGBClassifier(**xgb_kwargs)
    return models

# ── Step 5: Train & evaluate ─────────────────────────────────────────────────
results = []

for feat_name, vectorizer in feature_extractors.items():
    vec = vectorizer  # fresh per feature extractor (already new objects)
    X_train_vec = vec.fit_transform(X_train)
    X_val_vec = vec.transform(X_val)
    X_test_vec = vec.transform(X_test)

    for model_name, model in _build_models().items():
        if model_name == "NaiveBayes" and feat_name == "Word2Vec":
            continue

        model.fit(X_train_vec, y_train)

        y_val_pred = model.predict(X_val_vec)
        f1_val = f1_score(y_val, y_val_pred, average="macro")

        y_test_pred = model.predict(X_test_vec)
        f1_test = f1_score(y_test, y_test_pred, average="macro")

        # Per-class F1 in alphabetical label order (Negative, Neutral, Positive)
        f1_per = f1_score(y_test, y_test_pred, average=None, labels=[0, 1, 2])

        print(
            f"[{feat_name}+{model_name}] "
            f"val_f1={f1_val:.4f} | test_f1={f1_test:.4f}"
        )

        results.append(
            {
                "feature": feat_name,
                "model": model_name,
                "f1_macro_val": round(f1_val, 4),
                "f1_macro_test": round(f1_test, 4),
                "f1_negative": round(f1_per[0], 4),
                "f1_neutral": round(f1_per[1], 4),
                "f1_positive": round(f1_per[2], 4),
            }
        )

# ── Step 6: Save results ─────────────────────────────────────────────────────
os.makedirs(REPORT_DIR, exist_ok=True)
results_df = pd.DataFrame(results).sort_values("f1_macro_test", ascending=False)
results_df.to_csv("reports/results_summary.csv", index=False)

print("\n[train] Results summary:")
print(results_df.to_string(index=False))

# ── Step 7: Save best model ──────────────────────────────────────────────────
best = results_df.iloc[0].to_dict()
print(
    f"\n[train] Best model: {best['feature']}+{best['model']} "
    f"| F1={best['f1_macro_test']:.4f}"
)

# Re-fit on train + val combined
X_trainval = pd.concat([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

best_vec = {
    "BoW": CountVectorizer(max_features=10000, ngram_range=(1, 1)),
    "TF-IDF": TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2),
}[best["feature"]]

best_models = _build_models()
best_clf = best_models[best["model"]]

X_trainval_vec = best_vec.fit_transform(X_trainval)
best_clf.fit(X_trainval_vec, y_trainval)

with open(os.path.join(MODEL_DIR, "best_vectorizer.pkl"), "wb") as f:
    pickle.dump(best_vec, f)
with open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump(best_clf, f)

best_config = {
    "feature": best["feature"],
    "model": best["model"],
    "f1_macro_test": best["f1_macro_test"],
}
with open(os.path.join(MODEL_DIR, "best_config.json"), "w") as f:
    json.dump(best_config, f, indent=2)

print(f"[train] Saved vectorizer, model, and config to {MODEL_DIR}/")

# ── Step 8: Confusion matrix ─────────────────────────────────────────────────
cm_dir = "reports/confusion_matrix"
os.makedirs(cm_dir, exist_ok=True)

# Predict on test with the final re-fitted model
X_test_vec_final = best_vec.transform(X_test)
y_test_pred_final = best_clf.predict(X_test_vec_final)

cm = confusion_matrix(y_test, y_test_pred_final, labels=[0, 1, 2])
tick_labels = ["Negative", "Neutral", "Positive"]

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=tick_labels, yticklabels=tick_labels, ax=ax,
)
ax.set_title(f"Confusion Matrix — {best['feature']}+{best['model']}", fontsize=13)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(cm_dir, "best_model_cm.png"), dpi=150)
plt.close()

print(f"[train] Confusion matrix saved to {cm_dir}/best_model_cm.png")
