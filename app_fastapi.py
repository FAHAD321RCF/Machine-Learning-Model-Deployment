from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

class Features(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running!"}

@app.post("/predict")
def predict(data: Features):
    features = np.array(data.features).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    return {"prediction": round(prediction, 2)}
