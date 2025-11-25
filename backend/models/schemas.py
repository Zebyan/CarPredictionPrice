from pydantic import BaseModel, Field

class CarPredictionRequest(BaseModel):
    """Request schema for single car prediction"""
    marca: str = Field(..., description="Car brand")
    model: str = Field(..., description="Car model")
    an_fabricatie: int = Field(..., ge=2000, le=2025, description="Year of manufacture")
    rulaj: int = Field(..., ge=0, description="Mileage in km")
    putere: float = Field(..., gt=0, description="Power in HP")
    capacitate_motor: float = Field(..., description="Engine capacity in cc")
    combustibil: str = Field(..., description="Fuel type")
    caroserie: str = Field(..., description="Body type")
    culoare: str = Field(..., description="Car color")
    cutie_viteza: str = Field(..., description="Transmission type")

class PricePrediction(BaseModel):
    """Response schema for price prediction"""
    predicted: float = Field(..., description="Best estimated price in EUR")
    min_price: float = Field(..., description="Minimum price estimate")
    max_price: float = Field(..., description="Maximum price estimate")
    margin: float = Field(..., description="±EUR margin")
    confidence: float = Field(..., description="Model confidence percentage")
    residual_std: float = Field(..., description="Standard deviation of residuals")

class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    car: dict
    prediction: dict
