from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, HTTPException
import pandas as pd
import io
from src.config import MODEL
from src.preprocessing_pipeline import preprocessing_engine_data

class PredictionResult(BaseModel):
    unit: float
    cycle: float
    predicted_RUL: float

app = FastAPI(title="AirCraft Engine RUL Prediction API")

@app.post("/predict", response_model=list[PredictionResult])
async def predict(file: UploadFile):
    contents = await file.read()
    raw_df = pd.read_csv(io.BytesIO(contents))
    
    try:
        processed = preprocessing_engine_data(raw_df)
        model_input = processed.drop(columns=["cycle", "unit"])
        predicted_RUL = MODEL.predict(model_input)
        
        result_df = processed[["cycle", "unit"]].copy()
        result_df["predicted_RUL"] = predicted_RUL
        
        return result_df.to_dict(orient="records")
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        