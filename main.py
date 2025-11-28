from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib

app = FastAPI()

# Load model
model = joblib.load("house_model.pkl")

# Input schema
class Input(BaseModel):
    data: List[float]

# Home route (fix for Not Found on Render)
@app.get("/")
def home():
    return {"message": "House Price Prediction API is running!"}

# Prediction route
@app.post("/predict")
def predict(input: Input):
    pred = model.predict([input.data])
    return {"prediction": float(pred[0])}

# Uvicorn server for Render
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
