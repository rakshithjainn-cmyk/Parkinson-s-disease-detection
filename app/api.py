from fastapi import FastAPI
from pydantic import BaseModel
from src.eval import predict_single


app = FastAPI(title="Parkinson's Detection API")


class Sample(BaseModel):
features: list


@app.post('/predict')
async def predict(sample: Sample):
result = predict_single(sample.features)
return {"prediction": result, "label": "Parkinson's" if result == 1 else "Healthy"}


# Run with: uvicorn app.api:app --reload --port 8000
