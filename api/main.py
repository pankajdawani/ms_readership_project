from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import numpy as np
import pandas as pd
import os

# Fix paths for Docker
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "model", "model.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "model", "feature_names.pkl"))
with open(os.path.join(BASE_DIR, "model", "mappings.json"), "r") as f:
    mappings = json.load(f)

app = FastAPI(
    title="MS Publication Readership Predictor",
    description="Predicts expected reads for a Morgan Stanley publication before it is sent",
    version="1.0.0"
)

class PublicationInput(BaseModel):
    analyst_seniority: str
    word_count: int
    num_charts: int
    report_page_count: int
    topic_sensitivity: int
    subject_line_length: int
    send_hour: int
    is_market_open: int
    days_to_earnings: int
    subscriber_count: int
    num_explicit_recipients: int
    analyst_30d_avg_reads: int
    analyst_publish_frequency: str
    days_since_last_pub: int
    channel: str

class PredictionOutput(BaseModel):
    predicted_reads: int
    confidence_range_low: int
    confidence_range_high: int
    model_version: str

@app.get("/")
def root():
    return {"message": "MS Publication Readership Predictor API is live"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input: PublicationInput):
    seniority_encoded = mappings["seniority_map"][input.analyst_seniority]
    freq_encoded = mappings["freq_map"][input.analyst_publish_frequency]
    channel_encoded = mappings["channel_map"][input.channel]

    features = {
        "analyst_seniority": seniority_encoded,
        "word_count": input.word_count,
        "num_charts": input.num_charts,
        "report_page_count": input.report_page_count,
        "topic_sensitivity": input.topic_sensitivity,
        "subject_line_length": input.subject_line_length,
        "send_hour": input.send_hour,
        "is_market_open": input.is_market_open,
        "days_to_earnings": input.days_to_earnings,
        "subscriber_count": input.subscriber_count,
        "num_explicit_recipients": input.num_explicit_recipients,
        "analyst_30d_avg_reads": input.analyst_30d_avg_reads,
        "analyst_publish_frequency": freq_encoded,
        "days_since_last_pub": input.days_since_last_pub,
        "channel_Portal": channel_encoded["channel_Portal"],
        "channel_Web": channel_encoded["channel_Web"]
    }

    df_input = pd.DataFrame([features])[feature_names]
    pred_log = model.predict(df_input)[0]
    predicted_reads = int(np.expm1(pred_log))
    mae = 89

    return PredictionOutput(
        predicted_reads=predicted_reads,
        confidence_range_low=max(0, predicted_reads - mae),
        confidence_range_high=predicted_reads + mae,
        model_version="xgb_tuned_v1"
    )