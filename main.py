import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import numpy as np


df1 = pd.read_csv("sme1.csv").sample(n=100000, random_state=42)
df2 = pd.read_csv("sme2.csv").sample(n=100000, random_state=42)
df = pd.concat([df1, df2], ignore_index=True)
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
    "loan_status"
]

#Select only specified columns
df = df[columns_to_use]

#data cleaning
#1. Handle missing values
df = df[columns_to_use]
null_counts = df.isnull().sum()
print("Null counts:\n", null_counts[null_counts > 0])

null_cols = df.columns[df.isnull().any()]
print("\nNull column types:\n", df[null_cols].dtypes)

# Numeric columns → fill with median
numeric_null_cols = df[null_cols].select_dtypes(include='number').columns
for col in numeric_null_cols:
    df[col] = df[col].fillna(df[col].median())
    print(f"Filled {col} with median: {df[col].median()}")

# Categorical/text columns → fill with mode
cat_null_cols = df[null_cols].select_dtypes(include='object').columns
for col in cat_null_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    print(f"Filled {col} with mode: {df[col].mode()[0]}")

print("\nRemaining nulls:", df.isnull().sum().sum())

#Feature engineering
def engineer_features(df):
    #1. Credit history length(in months)
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format='%b-%Y', errors='coerce')
    df["issue_d"] = pd.to_datetime(df["issue_d"], format='%b-%Y', errors='coerce')                                        
    df["credit_history_months"] = ((df["issue_d"] - df["earliest_cr_line"]).dt.days / 30.44).round(0)
    df["credit_history_months"] = df["credit_history_months"].fillna(df["credit_history_months"].median())

    #2.income to loan ratio
    df["income_to_loan_ratio"] = df["annual_inc"] / (df["loan_amnt"] + 1)  # +1 to avoid division by zero
    df["income_to_loan_ratio"] = df["income_to_loan_ratio"].fillna(df["income_to_loan_ratio"].median())

    #3. revol_unitilizatiin bucket
    df["revol_util_bucket"] = pd.cut(df["revol_util"], bins=[-1, 20, 40, 60, 80, 100], labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])

    #4. Inquiry intensity score
    df["Inquiry_intensity_score"] = (df["inq_last_6mths"] / df["inq_last_6mths"].max()*100).fillna(0)
    df["Inquiry_intensity_score"] =  df["Inquiry_intensity_score"].clip(0,100)

    #5. Deliquency severity score
    df['delinq_amnt'] = df['delinq_amnt'].fillna(0)
    df['num_tl_30dpd'] = df['num_tl_30dpd'].fillna(0)
    df['num_tl_90g_dpd_24m'] = df['num_tl_90g_dpd_24m'].fillna(0)
    # Weighted severity score (0-100 scale)
    delinq_score = (
        df['delinq_2yrs'] * 10 +  # Recent delinquencies
        df['num_tl_30dpd'] * 5 +   # 30-day past due
        df['num_tl_90g_dpd_24m'] * 15 +  # 90+ days past due (more severe)
        (df['delinq_amnt'] / 1000)  # Delinquency amount normalized
    )
    df['delinquency_severity_score'] = (delinq_score / delinq_score.max() * 100).fillna(0)
    df['delinquency_severity_score'] = df['delinquency_severity_score'].clip(0, 100)

    return df

df = engineer_features(df)

print("Feature Engineering Complete!")
print("\nNew Features Summary:")
print(f"1. Credit History Length (months): Mean = {df['credit_history_months'].mean():.2f}")
print(f"2. Income-to-Loan Ratio: Mean = {df['income_to_loan_ratio'].mean():.2f}")
print(f"3. Revolving Utilization Buckets:\n{df['revol_util_bucket'].value_counts()}")
print(f"4. Inquiry Intensity Score: Mean = {df['Inquiry_intensity_score'].mean():.2f}")
print(f"5. Delinquency Severity Score: Mean = {df['delinquency_severity_score'].mean():.2f}")

#Splitting the dataset
X = df.drop(["loan_status", "issue_d", "earliest_cr_line"], axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

#model training
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

categorical_cols = X_train.select_dtypes(include = ['object', 'category']).columns.tolist()
print("categorical_cols",categorical_cols)

X_train_processed = X_train.copy()
X_test_processed = X_test.copy()

for col in categorical_cols:
    X_train_processed[col] = X_train_processed[col].astype(str)
    X_test_processed[col] = X_test_processed[col].astype(str)

X_train_processed = pd.get_dummies(X_train_processed,columns= categorical_cols, drop_first=True)
X_test_processed = pd.get_dummies(X_test_processed,columns= categorical_cols, drop_first=True)    

X_train_processed, X_test_processed = X_train_processed.align(X_test_processed,join='left',axis = 1,fill_value=0)

X_train_processed.columns = X_train_processed.columns.str.replace('[', '_', regex=False)
X_train_processed.columns = X_train_processed.columns.str.replace(']', '_', regex=False)
X_train_processed.columns = X_train_processed.columns.str.replace('<', '_', regex=False)
X_train_processed.columns = X_train_processed.columns.str.replace('>', '_', regex=False)

X_test_processed.columns = X_test_processed.columns.str.replace('[', '_', regex=False)
X_test_processed.columns = X_test_processed.columns.str.replace(']', '_', regex=False)
X_test_processed.columns = X_test_processed.columns.str.replace('<', '_', regex=False)
X_test_processed.columns = X_test_processed.columns.str.replace('>', '_', regex=False)

model = xgb.XGBClassifier(objective = 'binary:logistic', eval_metric = 'auc', max_depth = 6, learningrate = 0.1, n_estimators= 100,subsample = 0.8, random_state = 42, n_jobs = 1)
model.fit (X_train_processed, y_train_encoded, eval_set= [(X_test_processed, y_test_encoded)], verbose = False)

# Save the trained model and label encoder
joblib.dump(model, 'trained_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

#make predictions
y_pred_prob = model.predict_proba(X_test_processed)[:,1]
y_pred = model.predict(X_test_processed)

comparison = pd.DataFrame({
    'real': y_test.values,
    'predicted' : le.inverse_transform(y_pred),
    'probability' : y_pred_prob, 'match': y_test.values == le.inverse_transform(y_pred)
})

comparison.index = y_test.index
print('COMPARISON')
print(comparison.head(30).to_string())

#credit score mapping
def map_to_credit_score(prob):
    credit_score = 850 - (prob * 550)
    credit_score = np.clip(credit_score, 300, 850)
    return credit_score

def get_risk_band(credit_score):
    if credit_score >= 750:
        return 'Low Risk'
    elif credit_score >= 650:
        return 'Medium Risk'
    else:
        return 'High Risk' 

