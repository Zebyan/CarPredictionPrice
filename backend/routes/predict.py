from fastapi import APIRouter, HTTPException
from backend.models.schemas import CarPredictionRequest, PricePrediction
from backend.services.feature_engineer import engineer_features
from backend.services.predictor import predict_price, price_confidence_interval
from backend.config import settings

router = APIRouter(prefix="/predict", tags=["predictions"])

@router.post("/", response_model=PricePrediction)
async def predict_price_endpoint(car_data: CarPredictionRequest):
    """Predict car price based on features"""
    try:
        features = engineer_features(car_data)
        predicted_price = predict_price(features)
        interval = price_confidence_interval(predicted_price, features)

        return PricePrediction(
            predicted=predicted_price,
            min_price=interval["min_price"],
            max_price=interval["max_price"],
            margin=interval["margin"],
            confidence=interval["confidence"],
            residual_std=settings.model_mae,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")