from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.routes import predict, batch, health
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="ML-powered car price prediction API",
    version=settings.app_version,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(batch.router)

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "endpoints": {
            "health": "/health/",
            "predict": "/predict/",
            "batch": "/predict-batch/",
        }
    }

@app.get("/model-status/")
def model_status():
    """Check if trained model is loaded"""
    import os
    from pathlib import Path
    
    model_path = Path("backend/models_storage/random_forest_light.pkl")
    preprocessor_path = Path("backend/models_storage/preprocessor.pkl")
    metadata_path = Path("backend/models_storage/metadata/random_forest_light_metadata.json")
    
    status = {
        "model_file": {
            "path": str(model_path),
            "exists": model_path.exists(),
            "size_mb": model_path.stat().st_size / 1024 / 1024 if model_path.exists() else 0,
        },
        "preprocessor_file": {
            "path": str(preprocessor_path),
            "exists": preprocessor_path.exists(),
            "size_kb": preprocessor_path.stat().st_size / 1024 if preprocessor_path.exists() else 0,
        },
        "metadata_file": {
            "path": str(metadata_path),
            "exists": metadata_path.exists(),
        },
    }
    
    # Try to load model
    try:
        import joblib
        if model_path.exists():
            model = joblib.load(str(model_path))
            status["model_loaded"] = True
            status["model_type"] = str(type(model))
        else:
            status["model_loaded"] = False
            status["message"] = "Model file not found - using fallback calculation"
    except Exception as e:
        status["model_loaded"] = False
        status["error"] = str(e)
    
    return status

logger.info("✓ FastAPI app initialized")

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )