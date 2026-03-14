from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import shap
import os

app = Flask(__name__)
CORS(app)

# ── Load model package once at startup ──────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml', 'loan_model.pkl')
pkg = joblib.load(MODEL_PATH)

model           = pkg['model']
le              = pkg['label_encoder']
feature_columns = pkg['feature_columns']
explainer       = pkg['explainer']

# ── Constants ────────────────────────────────────────────────
EMP_MAPPING = {
    '< 1 year': 1, '1 year': 2, '2 years': 3, '3 years': 4,
    '4 years': 5, '5 years': 6, '6 years': 7, '7 years': 8,
    '8 years': 9, '9 years': 10, '10+ years': 11, 'Unknown': 0
}

# ── Helpers ──────────────────────────────────────────────────
def map_to_credit_score(rejection_prob: float) -> int:
    """Map rejection probability to a 300-850 credit score."""
    score = 850 - (rejection_prob * 550)
    return int(np.clip(score, 300, 850))

def get_risk_band(credit_score: int) -> str:
    if credit_score >= 750:
        return 'Low Risk'
    elif credit_score >= 650:
        return 'Medium Risk'
    return 'High Risk'

def plain_english(feature_name: str, feature_value, shap_value: float) -> str:
    """Convert a feature + SHAP value into a human-readable explanation."""
    direction = "increases rejection risk" if shap_value > 0 else "decreases rejection risk"
    templates = {
        'revol_util':                 f"Revolving credit utilization of {feature_value:.1f}% {direction}",
        'dti':                        f"Debt-to-income ratio of {feature_value:.2f} {direction}",
        'annual_inc':                 f"Annual income of ${feature_value:,.0f} {direction}",
        'int_rate':                   f"Interest rate of {feature_value:.2f}% {direction}",
        'loan_amnt':                  f"Loan amount of ${feature_value:,.0f} {direction}",
        'delinq_2yrs':                f"{int(feature_value)} delinquencies in last 2 years {direction}",
        'inq_last_6mths':             f"{int(feature_value)} credit inquiries in last 6 months {direction}",
        'open_acc':                   f"{int(feature_value)} open credit accounts {direction}",
        'total_acc':                  f"{int(feature_value)} total credit accounts {direction}",
        'pub_rec':                    f"{int(feature_value)} public records {direction}",
        'fico_range_low':             f"FICO score (low) of {int(feature_value)} {direction}",
        'fico_range_high':            f"FICO score (high) of {int(feature_value)} {direction}",
        'credit_history_months':      f"Credit history of {int(feature_value)} months {direction}",
        'income_to_loan_ratio':       f"Income-to-loan ratio of {feature_value:.2f} {direction}",
        'inquiry_intensity_score':    f"Inquiry intensity score of {feature_value:.1f} {direction}",
        'delinquency_severity_score': f"Delinquency severity score of {feature_value:.1f} {direction}",
        'employment_stability_score': f"Employment stability score of {int(feature_value)} {direction}",
        'installment':                f"Monthly installment of ${feature_value:,.0f} {direction}",
        'revol_bal':                  f"Revolving balance of ${feature_value:,.0f} {direction}",
        'pub_rec_bankruptcies':       f"{int(feature_value)} bankruptcy records {direction}",
    }
    fn = feature_name.lower()
    for key, text in templates.items():
        if key in fn:
            return text
    return f"{feature_name.replace('_', ' ').title()} ({feature_value}) {direction}"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the same feature engineering used during training."""
    df = df.copy()

    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format='%b-%Y', errors='coerce')
    df["issue_d"]          = pd.to_datetime(df["issue_d"],          format='%b-%Y', errors='coerce')
    df["credit_history_months"] = (
        (df["issue_d"] - df["earliest_cr_line"]).dt.days / 30.44
    ).round(0).fillna(0)

    df["income_to_loan_ratio"] = (df["annual_inc"] / (df["loan_amnt"] + 1)).fillna(0)

    df["revol_util_bucket"] = pd.cut(
        df["revol_util"].fillna(0),
        bins=[-1, 20, 40, 60, 80, 100],
        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    )

    max_inq = df["inq_last_6mths"].max()
    df["inquiry_intensity_score"] = (
        (df["inq_last_6mths"] / max_inq * 100).clip(0, 100) if max_inq > 0
        else pd.Series(0.0, index=df.index)
    ).fillna(0)

    raw_score = (
        df['delinq_2yrs'] * 10 +
        df['num_tl_30dpd'] * 5 +
        df['num_tl_90g_dpd_24m'] * 15 +
        (df['delinq_amnt'] / 1000)
    )
    max_score = raw_score.max()
    df['delinquency_severity_score'] = (
        (raw_score / max_score * 100).clip(0, 100) if max_score > 0
        else pd.Series(0.0, index=df.index)
    ).fillna(0)

    df['employment_stability_score'] = df['emp_length'].map(EMP_MAPPING).fillna(0)

    return df

def preprocess(data: dict) -> pd.DataFrame:
    """Turn raw frontend payload into a model-ready DataFrame."""
    df = pd.DataFrame([data])

    df = engineer_features(df)
    df = df.drop(columns=["issue_d", "earliest_cr_line"], errors='ignore')

    # One-hot encode categoricals
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Sanitize column names
    df.columns = (
        df.columns
        .str.replace('[', '_', regex=False)
        .str.replace(']', '_', regex=False)
        .str.replace('<', '_', regex=False)
        .str.replace('>', '_', regex=False)
    )

    # Align to training feature columns
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

# ── Routes ───────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/predict', methods=['POST'])
def predict():
    body = request.get_json(force=True)
    if not body:
        return jsonify({"error": "Request body is required"}), 400

    try:
        X = preprocess(body)
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 422

    # Predict
    pred_encoded   = model.predict(X)[0]
    pred_proba     = model.predict_proba(X)[0]  # [prob_class_0, prob_class_1]
    loan_status    = le.inverse_transform([pred_encoded])[0]  # 'Accepted' or 'Rejected'

    # Rejection probability drives credit score (higher rejection prob = lower score)
    rejected_idx       = list(le.classes_).index('Rejected')
    rejection_prob     = float(pred_proba[rejected_idx])
    default_probability = round(rejection_prob, 4)

    credit_score = map_to_credit_score(rejection_prob)
    risk_band    = get_risk_band(credit_score)

    # SHAP explanations
    shap_vals = explainer.shap_values(X)[0]  # shape: (n_features,)
    feature_vals = X.iloc[0]

    # Pair each feature with its SHAP value, sort by absolute impact
    shap_pairs = sorted(
        zip(feature_columns, feature_vals, shap_vals),
        key=lambda x: abs(x[2]),
        reverse=True
    )

    positive_explanations = []  # features that help acceptance (negative SHAP = less rejection)
    negative_explanations = []  # features that hurt acceptance (positive SHAP = more rejection)

    for feat, val, sv in shap_pairs[:10]:  # top 10 drivers
        if abs(sv) < 1e-4:
            continue
        text = plain_english(feat, val, sv)
        if sv < 0:
            positive_explanations.append(text.replace("decreases rejection risk", "").strip().rstrip(","))
        else:
            negative_explanations.append(text.replace("increases rejection risk", "").strip().rstrip(","))

    # Clean up explanation text for frontend
    def clean(texts):
        cleaned = []
        for t in texts:
            t = t.replace(" decreases rejection risk", "").replace(" increases rejection risk", "")
            cleaned.append(t.strip())
        return cleaned

    return jsonify({
        "default_probability": default_probability,
        "credit_score":        credit_score,
        "risk_band":           risk_band,
        "loan_status":         loan_status,
        "explanations": {
            "positive": clean(positive_explanations),
            "negative": clean(negative_explanations)
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
