import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import skops.io as sio

housing = fetch_california_housing()
X, y = housing["data"], housing["target"]
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)

pipeline = Pipeline([
    ("preprocessor", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])),
    ("model", RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,  
        random_state=42,
        n_jobs=1,
    )),
])

pipeline.fit(Xtr, ytr)
print("test R2:", pipeline.score(Xte, yte))

sio.dump(pipeline, "model/model.skops")