# See-House 🏠

An end-to-end machine learning project for predicting California house prices using a **Random Forest Regressor**, with the trained model exposed through a **FastAPI inference API** and a Streamlit interface for experimentation.


### Live Demo
[BankMarketer](https://lochan.vercel.app/work/see-house#demo)

### Case Study 
[work/see-house](https://lochan.vercel.app/work/see-house)

## Overview

See-House predicts the median house value of a California block group from demographic, housing, and geographic features.

The project covers the complete workflow:

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Preprocessing
   ↓
Random Forest
   ↓
Model Serialization
   ↓
FastAPI
   ↓
Web Interface
```

---

## Model

| Property     | Value                   |
| ------------ | ----------------------- |
| Task         | Regression              |
| Model        | Random Forest Regressor |
| Dataset      | California Housing      |
| Samples      | 20,640                  |
| Features     | 8                       |
| API          | FastAPI                 |
| Model format | `.skops`                |

The model uses a Random Forest configuration with:

* `n_estimators = 100`
* `max_depth = 12`
* `min_samples_leaf = 5`

---

## Features

The model receives eight numerical features:

| Feature      | Description                              |
| ------------ | ---------------------------------------- |
| `MedInc`     | Median income in the block group         |
| `HouseAge`   | Median house age                         |
| `AveRooms`   | Average number of rooms per household    |
| `AveBedrms`  | Average number of bedrooms per household |
| `Population` | Block group population                   |
| `AveOccup`   | Average household occupancy              |
| `Latitude`   | Block group latitude                     |
| `Longitude`  | Block group longitude                    |

---

## Model Evaluation

The documented holdout evaluation reports:

| Metric |      Result |
| ------ | ----------: |
| R²     |   **0.795** |
| MAE    | **$34,285** |
| Split  | **80 / 20** |

The model is intended to estimate values within the patterns represented by the training data. Tree-based models do not reliably extrapolate beyond their learned feature space.

---

## Why Random Forest?

Housing prices contain nonlinear relationships between economic, geographic, and housing characteristics.

Random Forest provides a practical way to model these interactions using an ensemble of decision trees without requiring explicit polynomial feature construction.

Conceptually:

```text
Input Features
      ↓
Decision Tree 1 ─┐
Decision Tree 2 ─┤
Decision Tree 3 ─┤
       ...       ├──→ Average Predictions
Decision Tree B ─┘
      ↓
House Value
```

---

## API

The production API is built with **FastAPI**.

### Endpoint

```text
POST /predict
```

### Request

The API accepts a list containing the eight model inputs in the expected order.

```json
{
  "inputs": [
    8.3252,
    41.0,
    6.9841,
    1.0238,
    322.0,
    2.5556,
    37.88,
    -122.23
  ]
}
```

### Response

```json
{
  "prediction": 4.526
}
```

The returned value is the model's normalized California Housing target prediction.

---

## Model Serialization

The trained Random Forest model is serialized using **skops**.

The API loads the serialized model and uses it directly for inference.

```text
model/model.skops
```

The API also uses `get_untrusted_types()` before loading the model and explicitly passes the resulting types to `skops.io.load()`.

---

## Project Structure

```text
See-House/
│
├── model/
│   └── model.skops
│
├── notebooks/
│   └── model development & experiments
│
├── src/
│   └── project source code
│
├── main.py
│   └── FastAPI application
│
├── streamlit_app.py
│   └── Streamlit interface
│
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

The repository currently separates the model artifact, notebooks, source code, API entry point, and Streamlit application.

---

## Tech Stack

### Machine Learning

* Python
* NumPy
* Scikit-learn
* Random Forest
* MLflow
* skops

### API

* FastAPI
* Pydantic
* Uvicorn

### Interface

* Streamlit

### Deployment

* FastAPI Cloud

The API project specifies FastAPI, NumPy, scikit-learn, and skops as its core dependencies and targets Python 3.12.

---

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/LochanJangid/See-House.git
cd See-House
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI server

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

## Streamlit

The repository also contains a Streamlit interface for interacting with the model.

Run:

```bash
streamlit run streamlit_app.py
```

---

## API Architecture

```text
                User
                  │
                  ▼
          Streamlit / Client
                  │
                  │ POST /predict
                  ▼
             FastAPI
                  │
                  ▼
          Pydantic Validation
                  │
                  ▼
        Serialized Random Forest
                  │
                  ▼
             Prediction
```

---

## Important Limitation

The model is trained on historical California Housing data.

Predictions should therefore be interpreted within the distribution represented by the training data. A Random Forest can model complex nonlinear relationships, but it is not designed to extrapolate reliably outside the patterns it has learned.

---


## Author

[**Lochan Jangid**](https://lochan.vercel.app/)
