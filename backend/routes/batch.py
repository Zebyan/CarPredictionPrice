from fastapi import APIRouter, HTTPException
from typing import List

from backend.models.schemas import CarPredictionRequest
from backend.services.feature_engineer import engineer_features
from backend.services.predictor import predict_price, price_confidence_interval
from backend.config import settings

router = APIRouter(prefix="/predict-batch", tags=["batch"])


@router.post("/")
async def predict_batch(cars: List[CarPredictionRequest]):
    """Predict prices for multiple cars."""
    try:
        predictions = []

        for car_data in cars:
            # 1) Feature engineering for this car
            features = engineer_features(car_data)

            # 2) Point prediction
            predicted_price = predict_price(features)

            # 3) Interval based on percentage (MAPE) + small absolute floor
            interval = price_confidence_interval(predicted_price, features)

            predictions.append({
                "car": {
                    "marca": car_data.marca,
                    "model": car_data.model,
                    "an_fabricatie": car_data.an_fabricatie,
                    "rulaj": car_data.rulaj,
                },
                "prediction": {
                    "predicted": predicted_price,
                    "min_price": interval["min_price"],
                    "max_price": interval["max_price"],
                    "margin": interval["margin"],
                    "confidence": interval.get("confidence"),
                    "residual_std": settings.model_mae,
                },
            })

        return {
            "count": len(predictions),
            "predictions": predictions,
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Batch prediction error: {str(e)}",
        )
