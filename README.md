# NexScore 💳

> Know your loan eligibility before you walk into a bank. Knowledge is leverage.

NexScore is an AI-powered loan eligibility prediction web app. Enter your financial profile, and an XGBoost model trained on 160,000 real loan records tells you — in under 3 seconds — whether you'd be accepted or rejected, why, and what your credit score looks like.

---

## ✨ What it does

- Predicts loan acceptance / rejection with **98.67% accuracy**
- Returns a **credit score (300–850)**, default probability, and risk band
- Explains every factor — strengths and areas of concern — using SHAP values
- Beautiful multi-step form with field-level tooltips, validation, and smooth animations
- Full result page with glassmorphism metrics, animated score arc, and factor breakdown

---

## 🗂 Project Structure

```
NexScore/
├── frontend/          # React + Vite UI
│   └── src/
│       ├── pages/     # LandingPage, ApplyPage
│       ├── components/# ResultPanel, MonthYearPicker
│       └── styles/    # CSS per page/component
├── backend/
│   └── app.py         # Flask API (port 8000)
├── ml/
│   ├── main.py        # Model training script
│   └── loan_model.pkl # Trained XGBoost model (gitignored)
├── sme1.csv           # Accepted loans dataset
├── sme2.csv           # Rejected loans dataset
└── requirements.txt
```

---

## 🧠 The Model

| Metric | Value |
|---|---|
| Algorithm | XGBoost (binary:logistic) |
| Training samples | ~160,000 (80k accepted + 80k rejected) |
| Accuracy | **98.67%** |
| ROC-AUC | **0.9988** |
| Features | 10 (FICO scores, DTI, loan amount, employment stability, DTI buckets) |
| Explainability | SHAP TreeExplainer |

The model was trained exclusively on features present in **both** datasets to avoid data leakage. SHAP values drive the positive/negative factor explanations shown on the result page.

---

## 🚀 Running Locally

### Prerequisites

- Python 3.9+
- Node.js 18+
- `uv` or `pip` for Python deps

### 1. Clone & set up Python env

```bash
git clone https://github.com/your-username/NexScore.git
cd NexScore

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model (first time only)

```bash
python ml/main.py
```

This reads `sme1.csv` and `sme2.csv`, trains the XGBoost model, and saves `ml/loan_model.pkl`.

### 3. Start the backend

```bash
venv/bin/python backend/app.py
# → Running on http://127.0.0.1:8000
```

### 4. Start the frontend

```bash
cd frontend
npm install
npm run dev
# → Running on http://localhost:5173
```

Open [http://localhost:5173](http://localhost:5173) and you're live.

---

## 🔌 API

### `POST /predict`

Send a JSON payload with the user's loan details:

```json
{
  "loan_amnt": 15000,
  "term": "36 months",
  "int_rate": 12.5,
  "installment": 502.3,
  "purpose": "debt_consolidation",
  "annual_inc": 75000,
  "dti": 18.5,
  "home_ownership": "RENT",
  "emp_length": "5 years",
  "verification_status": "Verified",
  "inq_last_6mths": 1,
  "delinq_2yrs": 0,
  "open_acc": 8,
  "total_acc": 20,
  "revol_bal": 5000,
  "revol_util": 35.0,
  "pub_rec": 0,
  "pub_rec_bankruptcies": 0,
  "earliest_cr_line": "Jan-2010",
  "fico_range_low": 700,
  "fico_range_high": 704,
  "issue_d": "Mar-2026",
  "delinq_amnt": 0,
  "num_tl_30dpd": 0,
  "num_tl_90g_dpd_24m": 0
}
```

**Response:**

```json
{
  "credit_score": 749,
  "default_probability": 0.1832,
  "risk_band": "Medium Risk",
  "loan_status": "Accepted",
  "explanations": {
    "positive": ["✓ FICO score (high) of 704", "✓ No recent credit inquiries"],
    "negative": ["✗ DTI in range 10-20%", "✗ High interest rate of 12.50%"]
  }
}
```

### `GET /health`

Returns `{ "status": "ok" }` — useful for deployment health checks.

---

## 🎨 Tech Stack

| Layer | Tech |
|---|---|
| Frontend | React 19, Vite 8, Framer Motion, React Router |
| Styling | Pure CSS — no UI library |
| Backend | Flask, Flask-CORS |
| ML | XGBoost, scikit-learn, SHAP, pandas, numpy |
| Fonts | Playfair Display, DM Sans (Google Fonts) |

---

## 🖼 Pages

**Landing** — Finance-themed hero with parallax, animated score card, feature grid, and stats pulled directly from real model metrics.

**Apply** — 5-step guided form with sidebar navigation, field tooltips, inline validation, and a custom month-year date picker. No field is left unexplained.

**Result** — Full-bleed background, glassmorphism metrics strip, animated credit score arc, and a two-column breakdown of what helped and what hurt.

---

## 📄 License

MIT — do whatever you want with it.

---

<p align="center">Crafted with ❤️ by <a href="https://portfolio-website-umber-pi-71.vercel.app/">Sakshi Paygude</a>™</p>
