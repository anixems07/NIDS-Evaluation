from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DATA_PATH = BASE_DIR / "data" / "processed" / "cicids2017_clean.csv"

if not CLEAN_DATA_PATH.exists():
    raise FileNotFoundError(f"Cleaned dataset not found at: {CLEAN_DATA_PATH}")

print(f"Loading {CLEAN_DATA_PATH} for validation...")
df = pd.read_csv(CLEAN_DATA_PATH)

print("=== LIVE VERIFICATION OF CLEANED DATASET ===")
print(f"Shape:              {df.shape}  (was 2,830,743 x 79 raw)")
print(f"Missing values:     {df.isna().sum().sum()}")          # expect 0
print(f"Infinite values:    {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")  # expect 0
print(f"Duplicate rows:     {df.duplicated().sum()}")           # expect 0
text_cols = [c for c in df.columns if df[c].dtype == object and c != "Label"]
print(f"Text feature cols:  {text_cols}")                       # expect []
leaks = [c for c in ["Source IP","Destination IP","Source Port",
                     "Destination Port","Timestamp","Flow ID"] if c in df.columns]
print(f"Leakage columns:    {leaks}")                           # expect []
print(f"Classes:            {df['Label'].nunique()} (encoded in 'Label_encoded')")