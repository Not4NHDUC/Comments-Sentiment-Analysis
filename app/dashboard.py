import os
import re
import sys
import json
import pickle
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocessing.pipeline import preprocess

st.set_page_config(
    page_title="Vietnamese Sentiment Analysis",
    page_icon="🎯",
    layout="wide",
)

PHOBERT_AVAILABLE = os.path.exists("models/phobert_config.json")


@st.cache_resource
def load_model():
    vectorizer = pickle.load(open("models/best_vectorizer.pkl", "rb"))
    model = pickle.load(open("models/best_model.pkl", "rb"))
    label_mapping = json.load(open("models/label_mapping.json"))
    config = json.load(open("models/best_config.json"))
    return vectorizer, model, label_mapping, config


@st.cache_resource
def load_phobert_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("models/phobert")
    model = AutoModelForSequenceClassification.from_pretrained("models/phobert")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    label_mapping = json.load(open("models/phobert_label_mapping.json"))
    return tokenizer, model, label_mapping, device


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


def predict_one(text: str, vectorizer, model, label_mapping) -> dict:
    early = check_rating_pattern(text)
    if early is not None:
        return {"label": early, "confidence": 1.0, "probabilities": {early: 1.0}}
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    proba = model.predict_proba(vec)[0]
    idx = proba.argmax()
    return {
        "label": label_mapping[str(idx)],
        "confidence": round(float(proba.max()), 4),
        "probabilities": {
            label_mapping[str(i)]: round(float(p), 4) for i, p in enumerate(proba)
        },
    }


def predict_one_phobert(text: str, tokenizer, model, label_mapping, device) -> dict:
    early = check_rating_pattern(text)
    if early is not None:
        return {"label": early, "confidence": 1.0, "probabilities": {early: 1.0}}
    import torch
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits
    proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(proba.argmax())
    return {
        "label": label_mapping[str(idx)],
        "confidence": round(float(proba.max()), 4),
        "probabilities": {
            label_mapping[str(i)]: round(float(p), 4) for i, p in enumerate(proba)
        },
    }


# ── Load artifacts ────────────────────────────────────────────────────────────
vectorizer, model, label_mapping, config = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Thông tin mô hình")
st.sidebar.write(f"**Model:** {config['model']}")
st.sidebar.write(f"**Feature:** {config['feature']}")
st.sidebar.write(f"**F1-macro:** {config['f1_macro_test']:.4f}")
if PHOBERT_AVAILABLE:
    phobert_config = json.load(open("models/phobert_config.json"))
    st.sidebar.markdown("---")
    st.sidebar.write(f"**PhoBERT F1-macro:** {phobert_config['f1_macro_test']:.4f}")
st.sidebar.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["🔍 Phân tích đơn", "📁 Phân tích hàng loạt", "📊 Kết quả mô hình"]
)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single prediction
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Phân tích cảm xúc một đánh giá")

    if PHOBERT_AVAILABLE:
        model_choice = st.radio(
            "Chọn model:",
            ["TF-IDF + SVM", "PhoBERT"],
            horizontal=True,
        )
    else:
        model_choice = "TF-IDF + SVM"

    text_input = st.text_area(
        "Nhập đánh giá sản phẩm tiếng Việt:",
        height=150,
        placeholder="VD: Sản phẩm rất tốt, giao hàng nhanh, đóng gói cẩn thận...",
    )

    if st.button("🎯 Phân tích", type="primary"):
        if not text_input.strip():
            st.warning("Vui lòng nhập nội dung đánh giá.")
        else:
            if model_choice == "PhoBERT":
                with st.spinner("Đang tải PhoBERT..."):
                    pb_tokenizer, pb_model, pb_label_mapping, pb_device = load_phobert_model()
                result = predict_one_phobert(
                    text_input, pb_tokenizer, pb_model, pb_label_mapping, pb_device
                )
            else:
                result = predict_one(text_input, vectorizer, model, label_mapping)

            col1, col2 = st.columns(2)
            with col1:
                label = result["label"]
                emoji = "😊" if label == "Positive" else "😐" if label == "Neutral" else "😞"
                color = "green" if label == "Positive" else "gray" if label == "Neutral" else "red"
                st.markdown(f"### Kết quả: :{color}[{emoji} {label}]")
                st.metric("Độ tin cậy", f"{result['confidence'] * 100:.1f}%")
                st.caption(f"Model: {model_choice}")

            with col2:
                st.markdown("**Xác suất từng nhãn:**")
                for lbl, prob in result["probabilities"].items():
                    bar_color = (
                        "🟢" if lbl == "Positive" else "⚪" if lbl == "Neutral" else "🔴"
                    )
                    st.write(f"{bar_color} {lbl}: {prob * 100:.1f}%")
                    st.progress(prob)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch prediction
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Phân tích hàng loạt từ file CSV")
    st.info("File CSV cần có cột 'review_text'")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        if "review_text" not in df.columns:
            st.error("Không tìm thấy cột 'review_text'")
        else:
            st.write(f"Đã tải {len(df)} đánh giá")
            if st.button("🚀 Chạy phân tích"):
                with st.spinner("Đang phân tích..."):
                    results = [
                        predict_one(str(t), vectorizer, model, label_mapping)
                        for t in df["review_text"]
                    ]
                    df["predicted_label"] = [r["label"] for r in results]
                    df["confidence"] = [r["confidence"] for r in results]

                st.success(f"Hoàn thành! {len(df)} đánh giá đã được phân loại.")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Phân bố nhãn dự đoán")
                    counts = Counter(df["predicted_label"])
                    fig, ax = plt.subplots()
                    colors = [
                        "green" if l == "Positive" else "gray" if l == "Neutral" else "red"
                        for l in counts.keys()
                    ]
                    ax.bar(counts.keys(), counts.values(), color=colors)
                    ax.set_ylabel("Số lượng")
                    for i, (k, v) in enumerate(counts.items()):
                        ax.text(i, v + 1, str(v), ha="center")
                    st.pyplot(fig)

                with col2:
                    st.subheader("Thống kê")
                    total = len(df)
                    for lbl in ["Positive", "Neutral", "Negative"]:
                        n = counts.get(lbl, 0)
                        st.metric(lbl, f"{n} ({n / total * 100:.1f}%)")

                st.subheader("Kết quả chi tiết")
                st.dataframe(
                    df[["review_text", "predicted_label", "confidence"]].head(100)
                )

                csv_out = df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "⬇️ Tải kết quả CSV",
                    csv_out,
                    "predictions.csv",
                    "text/csv",
                )

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model results
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Kết quả huấn luyện mô hình")

    if os.path.exists("reports/results_summary.csv"):
        results_df = pd.read_csv("reports/results_summary.csv")
        results_df = results_df.sort_values("f1_macro_test", ascending=False)

        st.subheader("Bảng so sánh tất cả model")
        st.dataframe(
            results_df.style.highlight_max(
                subset=["f1_macro_test"], color="lightgreen"
            )
        )

        st.subheader("Biểu đồ F1-macro")
        fig, ax = plt.subplots(figsize=(10, 5))
        labels_chart = [
            f"{r.feature}+{r.model}" for r in results_df.itertuples()
        ]
        bars = ax.barh(labels_chart, results_df["f1_macro_test"])
        ax.set_xlabel("F1-macro (test set)")
        ax.set_xlim(0, 1)
        bars[0].set_color("green")
        for bar, val in zip(bars, results_df["f1_macro_test"]):
            ax.text(
                val + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
            )
        st.pyplot(fig)

    if os.path.exists("reports/confusion_matrix/best_model_cm.png"):
        st.subheader("Confusion Matrix — Best Model")
        st.image("reports/confusion_matrix/best_model_cm.png")

    if os.path.exists("reports/eda/label_distribution.png"):
        st.subheader("EDA — Phân bố dữ liệu")
        col1, col2 = st.columns(2)
        with col1:
            st.image("reports/eda/label_distribution.png")
        with col2:
            st.image("reports/eda/review_length.png")
        st.image("reports/eda/top_words.png", use_column_width=True)
