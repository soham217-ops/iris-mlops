
import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/iris_model.pkl")

# Flower measurements
print("Enter Iris flower measurements in cm:")

sepal_length = float(input("Sepal length: "))
sepal_width = float(input("Sepal width: "))
petal_length = float(input("Petal length: "))
petal_width = float(input("Petal width: "))

# Prepare input
sample = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
)

# Predict species
prediction = model.predict(sample)[0]

species = ["Setosa", "Versicolor", "Virginica"]

print("Predicted Iris species:", species[prediction])
