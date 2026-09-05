from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import skops.io as sio


class HouseData(BaseModel):
    inputs: list[float]


app = FastAPI(
    title="Golden State Housing Estimator",
    description="California housing price prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

untrusted_types = sio.get_untrusted_types(
    file="model/model.skops"
)

print("Trusted model types:", untrusted_types)

model = sio.load(
    "model/model.skops",
    trusted=untrusted_types
)


@app.get("/")
def read_root():
    return {
        "message": "House Price API is running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/predict")
def predict(data: HouseData):

    X = np.array(data.inputs).reshape(1, -1)

    prediction = model.predict(X)

    return {
        "prediction": float(prediction[0])
    }