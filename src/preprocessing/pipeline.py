import re
import json
import os

import pandas as pd

try:
    import underthesea
    _UNDERTHESEA = True
except ImportError:
    print("[pipeline] underthesea not installed — using str.split() fallback for tokenization")
    _UNDERTHESEA = False

_DIR = os.path.dirname(__file__)

with open(os.path.join(_DIR, "teen_code_dict.json"), encoding="utf-8") as _f:
    _TEEN_CODE: dict[str, str] = json.load(_f)

with open(os.path.join(_DIR, "vietnamese_stopwords.txt"), encoding="utf-8") as _f:
    _STOPWORDS: set[str] = {line.strip() for line in _f if line.strip()}

_VI_CHARS = (
    r"àáâãèéêìíòóôõùúýăđơư"
    r"ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
)
_KEEP_PATTERN = re.compile(
    rf"[^\w\s{_VI_CHARS}]", re.UNICODE
)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _KEEP_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_teen_code(text: str) -> str:
    words = text.split()
    return " ".join(_TEEN_CODE.get(w, w) for w in words)


def tokenize(text: str) -> str:
    if _UNDERTHESEA:
        return underthesea.word_tokenize(text, format="text")
    return text


def remove_stopwords(text: str) -> str:
    tokens = text.split()
    return " ".join(t for t in tokens if t not in _STOPWORDS and len(t) >= 2)


def preprocess(text: str) -> str:
    text = clean_text(text)
    text = apply_teen_code(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    return text


def run_pipeline(input_csv: str, output_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv, encoding="utf-8-sig")

    n_input = len(df)
    df = df.dropna(subset=["review_text"])
    df = df[df["review_text"].str.strip() != ""]

    df["text_clean"] = df["review_text"].apply(preprocess)
    df = df.rename(columns={"review_text": "text_raw"})
    df = df[["text_raw", "text_clean", "label", "platform"]]
    df = df[df["text_clean"].str.strip() != ""]

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    label_dist = df["label"].value_counts().to_dict()
    print(f"[pipeline] Input: {n_input} rows -> Output: {len(df)} rows")
    print(f"[pipeline] Label distribution: {label_dist}")

    return df
