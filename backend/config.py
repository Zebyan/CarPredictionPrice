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

    # Model performance summary (folosit doar pentru UI / mesaje;
    # valorile exacte vin din fișierul de metadata al modelului)
    model_confidence: float = 77.13   # aproximat ca "% accuracy"
    model_mae: float = 1788.02
    model_mape: float = 22.87

    # Interval implicit pentru confidence interval
    price_margin_percent: float = 0.15

    # Căi către artefactele modelului Random Forest (pipeline complet)
    model_path: str = str(
        BASE_DIR
        / "models_storage"
        / "random_forest_best.pkl"
    )

    # Păstrat pentru compatibilitate (nu e folosit dacă pipeline-ul conține preprocessor-ul)
    preprocessor_path: str = str(
        BASE_DIR
        / "models_storage"
        / "preprocessor.pkl"
    )

    metadata_path: str = str(
        BASE_DIR
        / "models_storage"
        / "metadata"
        / "random_forest_best_metadata.json"
    )

    class Config:
        env_file = ".env"


settings = Settings()
