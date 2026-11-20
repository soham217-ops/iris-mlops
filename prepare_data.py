
import pandas as pd
from sklearn.datasets import load_iris

# Load built-in Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add target species
df["target"] = iris.target
df["species"] = df["target"].map(
    dict(enumerate(iris.target_names))
)

# Save dataset
df.to_csv("data/iris.csv", index=False)

print("Iris dataset created successfully!")
print("Total records:", len(df))
print(df.head())
