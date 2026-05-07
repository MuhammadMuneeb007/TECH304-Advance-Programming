from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


# Load the email spam dataset
DATA_PATH = Path(__file__).parent / "emails.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

data = pd.read_csv(DATA_PATH)

if "Prediction" not in data.columns:
    raise ValueError("Expected a 'Prediction' column in emails.csv")

# Separate features and labels
X = data.drop(columns=["Prediction"])
if "Email No." in X.columns:
    X = X.drop(columns=["Email No."])

y = data["Prediction"]

# Ensure numeric features only
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (helps SVM performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM classifier
svm = SVC(kernel="linear", C=1.0, class_weight="balanced")
svm.fit(X_train_scaled, y_train)

# Evaluate model
pred = svm.predict(X_test_scaled)
accuracy = accuracy_score(y_test, pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, pred))

print("Suggested Enhancements:")
print("- Try TF-IDF or n-gram features instead of raw word counts")
print("- Tune hyperparameters (C, kernel) with cross-validation")
print("- Compare with LinearSVC for faster training on large datasets")
print("- Address class imbalance with resampling or class weights")
print("- Perform feature selection to reduce dimensionality")
