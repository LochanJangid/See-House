from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import numpy as np
import skops.io as sio

MODEL_PATH = Path(__file__).parent / "model" / "model.skops"
N_FEATURES = 8  # MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Lat, Long


app = FastAPI(
    title="Golden State Housing Estimator",
    description="California housing price prediction API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (good for testing). In production, replace with ["http://localhost:3000", "https://your-nextjs-domain.com"]
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)
print(f"Loading model from {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.2f} MB)")
untrusted_types = sio.get_untrusted_types(file=str(MODEL_PATH))
model = sio.load(str(MODEL_PATH), trusted=untrusted_types)
print("Model loaded.")


class HouseData(BaseModel):
    inputs: list[float]

    @field_validator("inputs")
    @classmethod
    def check_length(cls, v):
        if len(v) != N_FEATURES:
            raise ValueError(f"expected {N_FEATURES} features, got {len(v)}")
        return v


@app.get("/")
def read_root():
    return {"message": "House Price API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: HouseData):
    X = np.array(data.inputs).reshape(1, -1)
    try:
        prediction = model.predict(X)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"prediction": float(prediction[0])}