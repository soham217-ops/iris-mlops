
import pandas as pd
import joblib
import json
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# MLflow experiment setup
mlflow.set_experiment("Iris_Flower_Classification")

# Load dataset
df = pd.read_csv("data/iris.csv")

X = df.drop(columns=["target", "species"])
y = df["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Start MLflow experiment run
with mlflow.start_run():

    # Model parameters
    n_estimators = 100
    random_state = 42

    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        target_names=["Setosa", "Versicolor", "Virginica"]
    )

    print("Model training completed!")
    print("Accuracy:", accuracy)
    print(report)

    # Log parameters and metrics in MLflow
    mlflow.log_param("algorithm", "RandomForest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("test_size", 0.2)

    mlflow.log_metric("accuracy", accuracy)

    # Log trained model in MLflow
    mlflow.sklearn.log_model(
    sk_model=model,
    name="iris_random_forest",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    # Save model locally
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/iris_model.pkl")

    # Save metrics locally
    Path("metrics").mkdir(exist_ok=True)
    with open("metrics/results.json", "w") as file:
        json.dump({"accuracy": accuracy}, file, indent=4)

    print("Model saved successfully!")
    print("MLflow Run ID:", mlflow.active_run().info.run_id)

print("MLflow tracking completed!")
