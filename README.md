# Iris Flower Classification – MLOps Project

## 1. Project Overview

This project develops a Machine Learning model to classify Iris flowers into three species: Setosa, Versicolor, and Virginica.

It demonstrates an end-to-end MLOps workflow, including data management, model training, experiment tracking, testing, API development, containerization, and continuous integration.

## 2. Technologies Used

- **Python** – Programming language
- **Scikit-learn** – Machine Learning model
- **Pandas & NumPy** – Data processing
- **DVC & DagsHub** – Dataset version control
- **MLflow** – Experiment tracking
- **Pytest** – Model testing
- **FastAPI** – Prediction API
- **Docker** – Application containerization
- **Git & GitHub** – Version control
- **GitHub Actions** – CI automation

## 3. Project Architecture

The project follows this workflow:

1. Dataset preparation
2. Dataset version control using DVC
3. Model training using Random Forest
4. Experiment tracking using MLflow
5. Model testing using Pytest
6. API development using FastAPI
7. Docker image creation
8. Automated testing using GitHub Actions

## 4. Project Structure

```text
iris-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
├── metrics/
├── models/
├── src/
├── tests/
│   └── test_model.py
├── prepare_data.py
├── train.py
├── predict.py
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## 5. Installation and Execution

Clone the repository:

```bash
git clone https://github.com/soham217-ops/iris-mlops.git
cd iris-mlops
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare the dataset:

```bash
python prepare_data.py
```

Train the model:

```bash
python train.py
```

Run the API:

```bash
python -m uvicorn app:app --reload
```

Open API documentation:

http://127.0.0.1:8000/docs

## 6. API Testing

Example prediction input:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Expected prediction:

```json
{
  "prediction": 0,
  "species": "Setosa"
}
```

## 7. Docker Deployment

Build the Docker image:

```bash
docker build -t iris-mlops .
```

Run the container:

```bash
docker run -d -p 8000:8000 --name iris-container iris-mlops
```

Check running containers:

```bash
docker ps
```

## 8. Automated Testing and CI/CD

GitHub Actions automatically runs the project tests when code is pushed to the main branch or a pull request is created.

Run tests locally:

```bash
pytest -v
```

The current CI workflow successfully executes all four tests.

## 9. Experiment Tracking

MLflow is used to record model training parameters, accuracy metrics, and the trained model.

## 10. Conclusion

This project demonstrates how Machine Learning can be integrated with MLOps tools to build, test, package, and deploy a model through a reproducible workflow.
