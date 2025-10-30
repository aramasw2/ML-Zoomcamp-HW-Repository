import pickle
from typing import Literal
from pydantic import BaseModel, Field


from fastapi import FastAPI
import uvicorn



class Lead(BaseModel):
    lead_source: Literal["organic_search", "paid_ads", "referral", "social_media","NA"]  # include all possible categories
    number_of_courses_viewed: int
    annual_income: float

class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool

app = FastAPI(title="customer-churn-prediction")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(lead: dict) -> float:
    """Return probability of conversion"""
    return float(pipeline.predict_proba([lead])[0, 1])


@app.post("/predict")
def predict(lead: Lead) -> PredictResponse:
    prob = predict_single(lead.model_dump())

    return PredictResponse(
        churn_probability=prob,
        churn=prob >= 0.5
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)