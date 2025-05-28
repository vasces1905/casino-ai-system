# models/train_rf.py

import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load labeled dataset
df = pd.read_csv("labeled_customer_dataset.csv")

# Features / Target
X = df.drop(columns=["promo_response"])
y = df["promo_response"]

# Encode categorical vars (if needed)
X = pd.get_dummies(X)

# Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

# Save model & report
joblib.dump(rf, "models/rf_model.pkl")

with open("models/rf_metrics.json", "w") as f:
    json.dump(report, f, indent=4)

print("✅ RF model trained and saved to 'models/'")
print("📊 Metrics saved as 'rf_metrics.json'")
