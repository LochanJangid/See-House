import numpy as np
import mlflow
import skops.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Setup mlflow experiment
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("see_house")

housing_dataset = fetch_california_housing()
X = housing_dataset['data']
y = housing_dataset['target']

Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)

preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        random_state=42,
        n_jobs=1
    ))
])

pipeline.fit(Xtr, ytr)

sio.dump(pipeline, "model/model.skops")