import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLEANED = "BloodPressure_kaggle_cleaned.csv"

if not os.path.exists(CLEANED):
    raise SystemExit(f"Missing file: {CLEANED}. Please run the cleaning script first.")

# Load cleaned data
df = pd.read_csv(CLEANED)

# Ensure numeric dtype for hemoglobin
if 'Level_of_Hemoglobin' not in df.columns:
    raise SystemExit("Column 'Level_of_Hemoglobin' not found in cleaned dataset.")

df['Level_of_Hemoglobin'] = pd.to_numeric(df['Level_of_Hemoglobin'], errors='coerce')

# Ensure outlier flag exists; if not, compute a fallback IQR-based flag
flag_col = 'is_outlier_Level_of_Hemoglobin'
if flag_col not in df.columns:
    s = df['Level_of_Hemoglobin']
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[flag_col] = ((s < lower) | (s > upper)).astype(int)

# 1) Count of outliers
count_outliers = int(pd.to_numeric(df[flag_col], errors='coerce').fillna(0).sum())
print(f"is_outlier_Level_of_Hemoglobin sum: {count_outliers}")

# 2) Preview flagged rows (top 5)
flagged = df.loc[df[flag_col] == 1]
print("\nFlagged examples (top 5):")
# Show only relevant columns if present
cols_to_show = [c for c in ['Patient_Number', 'Sex', 'Age', 'BMI', 'Level_of_Hemoglobin', flag_col] if c in df.columns]
print(flagged[cols_to_show].head(5).to_string(index=False))

# 3) Histogram of hemoglobin
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(8, 4))
df['Level_of_Hemoglobin'].hist(bins=60, edgecolor='black')
plt.title("Level_of_Hemoglobin Distribution")
plt.xlabel("Hemoglobin")
plt.ylabel("Count")
plt.tight_layout()
hist_path = os.path.join("plots", "hemoglobin_hist.png")
plt.savefig(hist_path)
print(f"\nSaved histogram to: {hist_path}")

# 4) Sex-wise describe
if 'Sex' in df.columns:
    # Normalize to string to avoid category issues
    df['Sex'] = df['Sex'].astype(str)
    sex_desc = df.groupby('Sex')['Level_of_Hemoglobin'].describe()
    print("\nSex-wise Level_of_Hemoglobin summary:")
    print(sex_desc.to_string())
else:
    print("\nColumn 'Sex' not found for grouping.")
