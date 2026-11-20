
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score


def test_dataset_exists():
    assert Path("data/iris.csv").exists()


def test_model_exists():
    assert Path("models/iris_model.pkl").exists()


def test_model_prediction():
    model = joblib.load("models/iris_model.pkl")

    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    prediction = model.predict(sample)
    assert prediction[0] == 0


def test_model_accuracy():
    df = pd.read_csv("data/iris.csv")

    X = df.drop(columns=["target", "species"])
    y = df["target"]

    model = joblib.load("models/iris_model.pkl")
    predictions = model.predict(X)

    assert accuracy_score(y, predictions) >= 0.85
