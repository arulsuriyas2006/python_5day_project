from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib

app = FastAPI()

model = joblib.load("house_model.pkl")

class Input(BaseModel):
    data: List[float]

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running!"}

@app.post("/predict")
def predict(input: Input):
    pred = model.predict([input.data])
    return {"prediction": float(pred[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
