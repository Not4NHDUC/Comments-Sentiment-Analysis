import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import xgboost
except ImportError:
    os.system("pip install xgboost")

from src.models.train import *

print("Phase 3 complete.")
