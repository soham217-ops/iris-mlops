
import pandas as pd
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/iris.csv")

X = df.drop(columns=["target", "species"])
y = df["target"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model training completed!")
print("Accuracy:", accuracy)
print(classification_report(
    y_test,
    predictions,
    target_names=["Setosa", "Versicolor", "Virginica"]
))

# Save model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/iris_model.pkl")

# Save metrics
Path("metrics").mkdir(exist_ok=True)
with open("metrics/results.json", "w") as file:
    json.dump({"accuracy": accuracy}, file, indent=4)

print("Model saved successfully!")
