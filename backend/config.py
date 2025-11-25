from pydantic_settings import BaseSettings
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Application settings"""

    # App info
    app_name: str = "Car Price Predictor API"
    app_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    model_confidence: float = 77.13  
    model_mae: float = 1788.02
    model_mape: float = 22.87

    price_margin_percent: float = 0.15

    model_path: str = str(
        BASE_DIR
        / "models_storage"
        / "random_forest_light.pkl"
    )

    preprocessor_path: str = str(
        BASE_DIR
        / "models_storage"
        / "preprocessor.pkl"
    )

    metadata_path: str = str(
        BASE_DIR
        / "models_storage"
        / "metadata"
        / "random_forest_light_metadata.json"
    )

    class Config:
        env_file = ".env"


settings = Settings()
