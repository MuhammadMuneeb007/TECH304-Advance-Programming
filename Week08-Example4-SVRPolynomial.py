from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR


# Load the Position Salaries dataset
DATA_PATH = Path(__file__).parent / "Position_Salaries.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

salary_data = pd.read_csv(DATA_PATH)

features = salary_data.iloc[:, 1:-1].values
labels = salary_data.iloc[:, 2].values

# Train an SVR model with a polynomial kernel
svr = SVR(kernel="poly", C=20)
svr.fit(features, labels)

# Evaluate the model
predictions = svr.predict(features)
mse = mean_squared_error(labels, predictions)
rmse = mse ** 0.5
print(f"RMSE: {rmse:.2f}")

# Plot actual data and model predictions
plt.scatter(features, labels)
plt.plot(features, predictions, c="r")
plt.xlabel("Level in firm.")
plt.ylabel("Salary for that position.")
plt.show()
