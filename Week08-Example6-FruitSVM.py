from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report


# Load the fruit dataset
DATA_PATH = Path(__file__).parent / "fruit_data_with_colors.txt"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

fruit_data = pd.read_table(DATA_PATH)

# Count plot of fruit categories
sns.countplot(x="fruit_name", data=fruit_data, label="Count", hue="fruit_name")
plt.show()

# Box plots of numeric features
features = ["mass", "width", "height", "color_score"]
fruit_data[features].plot(kind="box", subplots=True, layout=(1, 4), figsize=(10, 4))
plt.tight_layout()
plt.savefig("fruits_box.png")
plt.show()

# SVM classification
X = fruit_data[features]
y = fruit_data["fruit_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC()
svm.fit(X_train_scaled, y_train)

print(
    "Accuracy of SVM classifier on training set: {:.2f}".format(
        svm.score(X_train_scaled, y_train)
    )
)
print(
    "Accuracy of SVM classifier on test set: {:.2f}".format(
        svm.score(X_test_scaled, y_test)
    )
)

pred = svm.predict(X_test_scaled)
print(classification_report(y_test, pred))
