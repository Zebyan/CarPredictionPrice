from fastapi import APIRouter
from backend.model_loader import ModelLoader

router = APIRouter(tags=["health"])

@router.get("/health/")
def health_check():
    """Health check endpoint"""
    try:
        model_info = ModelLoader.get_model_info()

        status = model_info.get("status")

        if status == "metadata_not_found":
            model_status = "not_ready"
        elif "error" in model_info:
            model_status = "error"
        else:
            # Acceptă și vechiul 'accuracy', și noul 'accuracy_percent'
            accuracy = (
                model_info.get("accuracy")
                or model_info.get("accuracy_percent")
            )
            model_status = "loaded" if accuracy is not None else "not_ready"

    except Exception as e:
        model_info = {"error": str(e)}
        model_status = "error"

    return {
        "status": "ok",
        "message": "API is running",
        "version": "1.0.0",
        "model_status": model_status,
        "model_info": model_info,
    }
