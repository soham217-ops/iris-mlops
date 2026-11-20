
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Iris Flower Prediction API")

model = joblib.load("models/iris_model.pkl")


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return {"message": "Iris ML Prediction API is running!"}


@app.post("/predict")
def predict(data: IrisInput):

    sample = pd.DataFrame(
        [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    prediction = int(model.predict(sample)[0])

    species = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    return {
        "prediction": prediction,
        "species": species[prediction]
    }
