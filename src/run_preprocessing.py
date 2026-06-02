import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing.pipeline import run_pipeline
from src.preprocessing.eda import run_eda

input_csv = "data/raw/combined.csv"
output_csv = "data/processed/clean_data.csv"
os.makedirs("data/processed", exist_ok=True)

df = run_pipeline(input_csv, output_csv)
run_eda(df)
print("Phase 2 complete.")
