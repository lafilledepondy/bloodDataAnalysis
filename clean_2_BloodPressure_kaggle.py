import pandas as pd
import numpy as np

url = "https://gayathiri-ravendirane.emi.u-bordeaux.fr/DATA/BloodPressure_kaggle.csv"
print("Loading data from:", url)
# Preserve Patient_Number as string when present to avoid accidental coercion
df = pd.read_csv(url, dtype=str)
# keep original patient id column safe copy (handle possible case variations)
if 'Patient_Number' in df.columns:
    df['_orig_Patient_Number'] = df['Patient_Number']
elif 'patient_number' in df.columns:
    df['_orig_Patient_Number'] = df['patient_number']
print('Loaded shape (raw):', df.shape)
print("Initial shape:", df.shape)

# Remove exact duplicates
dup_count = df.duplicated().sum()
print("Exact duplicates:", dup_count)
if dup_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Dropped duplicates. New shape:", df.shape)

# Sanitize physiologic ranges
def sanitize_range(series, low=None, high=None):
    s = pd.to_numeric(series, errors='coerce')
    if low is not None:
        s = s.where(s >= low, np.nan)
    if high is not None:
        s = s.where(s <= high, np.nan)
    return s

if 'Age' in df.columns:
    df['Age'] = sanitize_range(df['Age'], low=18, high=100)
if 'BMI' in df.columns:
    df['BMI'] = sanitize_range(df['BMI'], low=10, high=60)
if 'Level_of_Hemoglobin' in df.columns:
    df['Level_of_Hemoglobin'] = sanitize_range(df['Level_of_Hemoglobin'], low=4, high=20)

# Numeric columns

# Convert numeric-like columns to numeric but exclude patient id column from numeric processing
possible_id_cols = [c for c in ['Patient_Number', 'patient_number', '_orig_Patient_Number'] if c in df.columns]

# First attempt to coerce numeric columns but keep id columns untouched
for col in df.columns:
    if col in possible_id_cols:
        continue
    # try to coerce numeric where appropriate
    # if coercion produces mostly numbers, keep numeric dtype
    coerced = pd.to_numeric(df[col], errors='coerce')
    # consider as numeric if at least half values parse as numbers
    if coerced.notna().sum() >= (len(df) / 2):
        df[col] = coerced

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric columns (excluding ID):", numeric_cols)

if numeric_cols:
    # Median imputation for simplicity
    for col in numeric_cols:
        median = df[col].median()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(median)

    # Create IQR-based outlier flags and winsorize
    outlier_summary = {}
    for col in numeric_cols:
        col_series = pd.to_numeric(df[col], errors='coerce')
        Q1 = col_series.quantile(0.25)
        Q3 = col_series.quantile(0.75)
        IQR = Q3 - Q1
        flag_col = f'is_outlier_{col}'
        df[flag_col] = 0
        if pd.isna(IQR) or IQR == 0:
            outlier_summary[col] = 0
            continue
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        is_out = ((col_series < lower) | (col_series > upper)).astype(int)
        df[flag_col] = is_out
        outlier_summary[col] = int(is_out.sum())

    # Winsorize
    for col in numeric_cols:
        col_series = pd.to_numeric(df[col], errors='coerce')
        if col_series.notna().any():
            low_q = col_series.quantile(0.01)
            high_q = col_series.quantile(0.99)
            df[col] = col_series.clip(lower=low_q, upper=high_q)

    print("Outliers per numeric column:")
    for k, v in outlier_summary.items():
        print(f" - {k}: {v}")

# Categorical handling
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print("Categorical columns:", cat_cols)
for col in cat_cols:
    df[col] = df[col].where(df[col].notnull(), 'Unknown').astype(str)

# Normalize Sex
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].astype(str).str.strip().str.lower()
    df['Sex'] = df['Sex'].replace({
        'm': 'male', 'male.': 'male', 'male': 'male',
        'f': 'female', 'female.': 'female', 'female': 'female',
        'nan': 'unknown', 'none': 'unknown', 'unknown': 'unknown'
    }).fillna('unknown')

# Pregnancy to 0/1
if 'Pregnancy' in df.columns:
    df['Pregnancy'] = df['Pregnancy'].replace({
        'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'Y':1, 'N':0, True:1, False:0
    })
    df['Pregnancy'] = pd.to_numeric(df['Pregnancy'], errors='coerce').fillna(0).astype(int)
    if 'Sex' in df.columns:
        male_mask = df['Sex'].str.startswith('m', na=False)
        df.loc[male_mask, 'Pregnancy'] = 0

# Convert some categorical flags to binary
binary_candidates = ['Pregnancy', 'Chronic_kidney_disease', 'Adrenal_and_thyroid_disorders', 'Smoking']
for c in binary_candidates:
    if c in df.columns:
        s = df[c].astype(str).str.strip().str.lower()
        s = s.replace({'yes':1, 'y':1, 'true':1, '1':1, 'no':0, 'n':0, 'false':0, '0':0, 'unknown':0, 'nan':0})
        df[c] = pd.to_numeric(s, errors='coerce').fillna(0).astype(int)

# Drop non-informative is_outlier_ flags
outlier_cols = [c for c in df.columns if c.startswith('is_outlier_')]
non_informative = []
for c in outlier_cols:
    try:
        ssum = int(df[c].sum())
    except Exception:
        ssum = 0
    if ssum == 0 or 'patient_number' in c.lower():
        non_informative.append(c)

if non_informative:
    print('Dropping non-informative outlier flags:', non_informative)
    df = df.drop(columns=non_informative)

# Restore original patient id column and remove temporary copy
if '_orig_Patient_Number' in df.columns:
    if 'Patient_Number' in df.columns:
        df['Patient_Number'] = df['_orig_Patient_Number']
    elif 'patient_number' in df.columns:
        df['patient_number'] = df['_orig_Patient_Number']
    df = df.drop(columns=['_orig_Patient_Number'])


