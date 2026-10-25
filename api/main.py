from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("model/model.pkl")

class Input(BaseModel):
    recency_days: float
    frequency: float
    monetary_avg: float
    freight_avg: float
    items_avg: float
    lifespan_days: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: Input):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0].tolist()
    return {"churn": int(pred), "prob": prob}
