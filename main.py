from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import skops.io as sio
from pathlib import Path


class HouseData(BaseModel):
    inputs: list[float]


app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.skops"

untrusted_types = sio.get_untrusted_types(
    file=str(MODEL_PATH)
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


@app.post("/predict")
def predict(data: HouseData):

    X = np.array(data.inputs).reshape(1, -1)

    prediction = model.predict(X)

    return {
        "prediction": float(prediction[0])
    }