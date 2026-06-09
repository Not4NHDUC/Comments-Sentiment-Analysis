import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from src.preprocessing.pipeline import run_pipeline
from src.preprocessing.eda import run_eda

input_csv = "data/raw/combined.csv"
output_csv = "data/processed/clean_data.csv"
os.makedirs("data/processed", exist_ok=True)

extra_path = "data/extra/short_sentences.csv"
if os.path.exists(extra_path):
    extra = pd.read_csv(extra_path)
    combined = pd.read_csv(input_csv, encoding="utf-8-sig")
    merged = pd.concat([combined, extra], ignore_index=True)
    merged.to_csv(input_csv, index=False, encoding="utf-8-sig")
    print(f"[extra] Added {len(extra)} short sentences to combined.csv")

df = run_pipeline(input_csv, output_csv)
run_eda(df)
print("Phase 2 complete.")
