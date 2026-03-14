import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

print("LOAN ACCEPTANCE PREDICTION MODEL")

# 1. LOAD DATA
print("\n[1/6] Loading data...")
accepted_df = pd.read_csv("sme1.csv")
print(f"  Accepted (full): {len(accepted_df):,} records")

# Sample rejected to match accepted count — keeps classes balanced
# and avoids training on 27M+ rows unnecessarily
n_accepted = len(accepted_df)
rejected_df = pd.read_csv("sme2.csv").sample(n=n_accepted, random_state=42)
print(f"  Rejected (sampled to match): {len(rejected_df):,} records")

# 2. PREPARE AND COMBINE DATASETS
print("\n[2/6] Preparing datasets...")

# columns_to_use defines the features we want from the accepted dataset
columns_to_use = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "purpose",
    "annual_inc",
    "dti",
    "home_ownership",
    "emp_length",
    "verification_status",
    "inq_last_6mths",
    "delinq_2yrs",
    "open_acc",
    "total_acc",
    "revol_bal",
    "revol_util",
    "pub_rec",
    "pub_rec_bankruptcies",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "issue_d",
    "delinq_amnt",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
]

# Map accepted dataset — use columns_to_use directly
accepted_mapped = accepted_df[columns_to_use].copy()
accepted_mapped['loan_decision'] = 'Accepted'

# Map rejected dataset — different column names, fill unavailable fields with NaN
# sme2.csv has: Amount Requested, Debt-To-Income Ratio, Employment Length, Risk_Score, State
rejected_mapped = pd.DataFrame()
rejected_mapped['loan_amnt'] = pd.to_numeric(rejected_df['Amount Requested'], errors='coerce')
rejected_mapped['dti'] = (
    rejected_df['Debt-To-Income Ratio']
    .astype(str).str.rstrip('%')
    .pipe(pd.to_numeric, errors='coerce')
)
rejected_mapped['emp_length'] = rejected_df['Employment Length'].astype(str).str.strip()
# Use Risk_Score as a proxy for fico_range_low and fico_range_high
rejected_mapped['fico_range_low'] = pd.to_numeric(rejected_df['Risk_Score'], errors='coerce')
rejected_mapped['fico_range_high'] = pd.to_numeric(rejected_df['Risk_Score'], errors='coerce')

# Remaining columns not present in rejected dataset — fill with NaN (handled in cleaning)
for col in columns_to_use:
    if col not in rejected_mapped.columns:
        rejected_mapped[col] = np.nan

rejected_mapped['loan_decision'] = 'Rejected'

# Combine
df = pd.concat([accepted_mapped, rejected_mapped], ignore_index=True)
print(f"  Combined: {len(df):,} records")

# 3. DATA CLEANING
print("\n[3/6] Cleaning data...")

null_counts = df.isnull().sum()
print("  Null counts (non-zero):\n", null_counts[null_counts > 0].to_string())

# Numeric columns → fill with median
numeric_null_cols = df.columns[df.isnull().any()].intersection(
    df.select_dtypes(include='number').columns
)
for col in numeric_null_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical/text columns → fill with mode
cat_null_cols = df.columns[df.isnull().any()].intersection(
    df.select_dtypes(include='object').columns
)
for col in cat_null_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print(f"  Remaining nulls: {df.isnull().sum().sum()}")

# 4. FEATURE ENGINEERING
print("\n[4/6] Engineering features...")

def engineer_features(df):
    # 1. Credit history length (in months) — only meaningful for accepted records
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format='%b-%Y', errors='coerce')
    df["issue_d"] = pd.to_datetime(df["issue_d"], format='%b-%Y', errors='coerce')
    df["credit_history_months"] = (
        (df["issue_d"] - df["earliest_cr_line"]).dt.days / 30.44
    ).round(0).fillna(0)

    # 2. Income to loan ratio
    df["income_to_loan_ratio"] = (df["annual_inc"] / (df["loan_amnt"] + 1)).fillna(0)

    # 3. Revolving utilization bucket
    df["revol_util_bucket"] = pd.cut(
        df["revol_util"].fillna(0),
        bins=[-1, 20, 40, 60, 80, 100],
        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    )

    # 4. Inquiry intensity score (0-100)
    max_inq = df["inq_last_6mths"].max()
    df["inquiry_intensity_score"] = (
        (df["inq_last_6mths"] / max_inq * 100) if max_inq > 0 else 0
    ).fillna(0).clip(0, 100)

    # 5. Delinquency severity score (0-100)
    df['delinq_amnt'] = df['delinq_amnt'].fillna(0)
    df['num_tl_30dpd'] = df['num_tl_30dpd'].fillna(0)
    df['num_tl_90g_dpd_24m'] = df['num_tl_90g_dpd_24m'].fillna(0)
    raw_score = (
        df['delinq_2yrs'] * 10 +
        df['num_tl_30dpd'] * 5 +
        df['num_tl_90g_dpd_24m'] * 15 +
        (df['delinq_amnt'] / 1000)
    )
    max_score = raw_score.max()
    df['delinquency_severity_score'] = (
        (raw_score / max_score * 100) if max_score > 0 else 0
    ).fillna(0).clip(0, 100)

    # 6. Employment stability score
    emp_mapping = {
        '< 1 year': 1, '1 year': 2, '2 years': 3, '3 years': 4,
        '4 years': 5, '5 years': 6, '6 years': 7, '7 years': 8,
        '8 years': 9, '9 years': 10, '10+ years': 11, 'Unknown': 0, 'nan': 0
    }
    df['employment_stability_score'] = df['emp_length'].map(emp_mapping).fillna(0)

    return df

df = engineer_features(df)
print("  Feature engineering complete.")

# 5. PREPARE FEATURES AND TRAIN MODEL
print("\n[5/6] Training model...")

# Drop date columns and target; keep engineered + raw features
drop_cols = ["issue_d", "earliest_cr_line", "loan_decision"]
X = df.drop(columns=drop_cols)
y = df["loan_decision"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# Encode target (Accepted=0, Rejected=1 or vice versa — LabelEncoder decides)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
print(f"  Classes: {le.classes_}")

# One-hot encode categoricals
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"  Categorical cols: {categorical_cols}")

X_train_processed = pd.get_dummies(X_train.copy(), columns=categorical_cols, drop_first=True)
X_test_processed = pd.get_dummies(X_test.copy(), columns=categorical_cols, drop_first=True)

# Align columns
X_train_processed, X_test_processed = X_train_processed.align(
    X_test_processed, join='left', axis=1, fill_value=0
)

# Sanitize column names for XGBoost
for frame in [X_train_processed, X_test_processed]:
    frame.columns = (
        frame.columns
        .str.replace('[', '_', regex=False)
        .str.replace(']', '_', regex=False)
        .str.replace('<', '_', regex=False)
        .str.replace('>', '_', regex=False)
    )

# Train XGBoost
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(
    X_train_processed, y_train_encoded,
    eval_set=[(X_test_processed, y_test_encoded)],
    verbose=False
)

# Evaluate
y_pred = model.predict(X_test_processed)
y_pred_prob = model.predict_proba(X_test_processed)[:, 1]
accuracy = accuracy_score(y_test_encoded, y_pred)
roc_auc = roc_auc_score(y_test_encoded, y_pred_prob)
print(f"  Accuracy : {accuracy * 100:.2f}%")
print(f"  ROC-AUC  : {roc_auc:.4f}")

# Sample comparison
comparison = pd.DataFrame({
    'actual': y_test.values,
    'predicted': le.inverse_transform(y_pred),
    'probability': y_pred_prob,
    'match': y_test.values == le.inverse_transform(y_pred)
})
comparison.index = y_test.index
print("\nSample predictions (first 10):")
print(comparison.head(10).to_string())

# 6. SHAP EXPLAINER + SAVE .pkl FILES
print("\n[6/6] Training SHAP explainer and saving .pkl files...")

sample_size = min(1000, len(X_train_processed))
X_shap_sample = X_train_processed.sample(n=sample_size, random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap_sample)

feature_importance = pd.DataFrame({
    'feature': X_shap_sample.columns,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

print("\nTop 10 features by SHAP importance:")
print(feature_importance.head(10).to_string(index=False))

# Package everything needed for inference
model_package = {
    'model': model,
    'label_encoder': le,
    'feature_columns': X_train_processed.columns.tolist(),
    'categorical_columns': categorical_cols,
    'explainer': explainer,
    'feature_importance': feature_importance,
}

joblib.dump(model_package, 'loan_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("\n  ✓ loan_model.pkl saved")
print("  ✓ label_encoder.pkl saved")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"  Accuracy : {accuracy * 100:.2f}%")
print(f"  ROC-AUC  : {roc_auc:.4f}")
print(f"  Classes  : {le.classes_}")
