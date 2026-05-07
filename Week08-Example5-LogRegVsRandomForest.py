from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression with scaling
log_reg = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

# Random Forest (no scaling needed)
rf = RandomForestClassifier(n_estimators=200, random_state=42)

# Train models
log_reg.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predict on test data
log_reg_pred = log_reg.predict(X_test)
rf_pred = rf.predict(X_test)

# Evaluate performance
log_reg_acc = accuracy_score(y_test, log_reg_pred)
rf_acc = accuracy_score(y_test, rf_pred)

log_reg_f1 = f1_score(y_test, log_reg_pred, average="macro")
rf_f1 = f1_score(y_test, rf_pred, average="macro")

print("Logistic Regression Performance")
print(f"Accuracy: {log_reg_acc:.4f}")
print(f"F1-score (macro): {log_reg_f1:.4f}")

print("\nRandom Forest Performance")
print(f"Accuracy: {rf_acc:.4f}")
print(f"F1-score (macro): {rf_f1:.4f}")

print("\nQuick Comparison Summary")
print("- Logistic Regression: linear decision boundary, needs scaling, fast and interpretable")
print("- Random Forest: non-linear, no scaling needed, robust but slower, offers feature importance")

# Feature importance from Random Forest
feature_importances = rf.feature_importances_
ranked_features = sorted(
    zip(iris.feature_names, feature_importances), key=lambda item: item[1], reverse=True
)

print("\nRandom Forest Feature Importance")
for name, importance in ranked_features:
    print(f"{name}: {importance:.3f}")
