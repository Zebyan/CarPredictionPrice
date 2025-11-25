import joblib
import json
import logging
import os
from typing import Optional, Dict, Any

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage the trained model (Pipeline + metadata)."""

    _instance = None

    _model: Optional[Pipeline] = None
    _preprocessor = None
    _metadata: Optional[Dict[str, Any]] = None
    _feature_names = None

    def __new__(cls):
        """Singleton pattern - only one model instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ========================================================================
    # MODEL (PIPELINE)
    # ========================================================================
    @classmethod
    def load_model(cls, model_path: Optional[str] = None):
        """
        Load the trained model (RandomForest Pipeline).

        - Ia path-ul din settings dacă nu este dat explicit.
        - Extrage preprocessor-ul din pipeline.
        - PATCH: pentru fiecare OneHotEncoder, forțează categories_ de tip object
                 să fie toate stringuri (fără amestec str/int).
        """
        if model_path is None:
            from backend.config import settings
            model_path = settings.model_path

        if cls._model is None:
            try:
                if not os.path.exists(model_path):
                    logger.warning(f"Model file not found: {model_path}")
                    return None

                cls._model = joblib.load(model_path)
                logger.info(f"✓ Model (Pipeline) loaded from {model_path}")

                if isinstance(cls._model, Pipeline):
                    if "preprocessor" in cls._model.named_steps:
                        cls._preprocessor = cls._model.named_steps["preprocessor"]
                        logger.info("✓ Preprocessor extracted from Pipeline")

                        # === PATCH: OneHotEncoder.categories_ → string uniform ===
                        try:
                            transformers = getattr(
                                cls._preprocessor, "transformers_", []
                            )
                            for name, transformer, cols in transformers:
                                if isinstance(transformer, OneHotEncoder):
                                    new_cats = []
                                    for arr in transformer.categories_:
                                        # convertim TOATE valorile la string, ca să nu existe mix str/int
                                        if arr.dtype == object:
                                            arr_str = arr.astype(str)
                                            new_cats.append(arr_str)
                                        else:
                                            new_cats.append(arr)
                                    transformer.categories_ = new_cats
                            logger.info("✓ Normalized OneHotEncoder.categories_ to string for object arrays")
                        except Exception as patch_err:
                            logger.warning(
                                f"Could not normalize OneHotEncoder categories_ to string: {patch_err}"
                            )

                        # Feature names (optional)
                        try:
                            cls._feature_names = (
                                cls._preprocessor.get_feature_names_out()
                            )
                            logger.info("✓ Feature names extracted from preprocessor")
                        except Exception as fe:
                            logger.warning(f"Could not extract feature names: {fe}")
                    else:
                        logger.warning("Loaded Pipeline has no 'preprocessor' step.")
                else:
                    logger.warning("Loaded model is not a sklearn Pipeline.")

            except Exception as e:
                logger.error(f"Error loading model: {str(e)}", exc_info=True)
                cls._model = None
                return None

        return cls._model


    # ========================================================================
    # PREPROCESSOR
    # ========================================================================
    @classmethod
    def load_preprocessor(cls, preprocessor_path: Optional[str] = None):
        """
        Load the preprocessor.

        Nou comportament:
          - Preferă 'preprocessor' extras din pipeline.
        Comportament vechi (fallback):
          - Dacă nu există în pipeline, încearcă să încarce separat din fișier.
        """
        if cls._preprocessor is not None:
            return cls._preprocessor

        # 1) Încearcă din pipeline
        model = cls.load_model()
        if model is not None and isinstance(model, Pipeline):
            if "preprocessor" in model.named_steps:
                cls._preprocessor = model.named_steps["preprocessor"]
                logger.info("✓ Preprocessor obtained from loaded Pipeline")

                try:
                    cls._feature_names = cls._preprocessor.get_feature_names_out()
                    logger.info("✓ Feature names extracted from preprocessor")
                except Exception as fe:
                    logger.warning(f"Could not extract feature names: {fe}")

                return cls._preprocessor

        # 2) Fallback: preprocessor salvat separat (dacă există)
        if preprocessor_path is None:
            try:
                from backend.config import settings
                preprocessor_path = getattr(settings, "preprocessor_path", None)
            except Exception:
                preprocessor_path = None

        if not preprocessor_path:
            logger.warning(
                "No preprocessor_path provided and could not obtain from Pipeline."
            )
            return None

        try:
            if not os.path.exists(preprocessor_path):
                logger.warning(f"Preprocessor file not found: {preprocessor_path}")
                return None

            cls._preprocessor = joblib.load(preprocessor_path)
            logger.info(f"✓ Preprocessor loaded from {preprocessor_path}")

            try:
                cls._feature_names = cls._preprocessor.get_feature_names_out()
                logger.info("✓ Feature names extracted from preprocessor")
            except Exception as fe:
                logger.warning(f"Could not extract feature names: {fe}")

        except Exception as e:
            logger.error(f"Error loading preprocessor: {str(e)}")
            cls._preprocessor = None
            return None

        return cls._preprocessor

    # ========================================================================
    # METADATA
    # ========================================================================
    @classmethod
    def load_metadata(cls, metadata_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load model metadata from JSON."""
        if metadata_path is None:
            from backend.config import settings
            metadata_path = settings.metadata_path

        if cls._metadata is None:
            try:
                if not os.path.exists(metadata_path):
                    logger.warning(f"Metadata file not found: {metadata_path}")
                    return None

                with open(metadata_path, "r", encoding="utf-8") as f:
                    cls._metadata = json.load(f)

                logger.info(f"✓ Metadata loaded from {metadata_path}")

            except Exception as e:
                logger.error(f"Error loading metadata: {str(e)}")
                cls._metadata = None
                return None

        return cls._metadata

    # ========================================================================
    # FEATURE NAMES
    # ========================================================================
    @classmethod
    def get_feature_names(cls):
        """Return feature names after preprocessing, if available."""
        if cls._feature_names is not None:
            return cls._feature_names

        cls.load_preprocessor()
        return cls._feature_names

    # ========================================================================
    # MODEL INFO
    # ========================================================================
    @classmethod
    def get_model_info(cls) -> Dict[str, Any]:
        """
        Get model information based on metadata for the current model.

        Compatibil cu:
          - noul format de metadata (RandomForest + log/price metrics)
          - eventual format vechi (HistGBR / alte chei)
        """
        metadata = cls.load_metadata()
        if metadata is None:
            return {"status": "metadata_not_found"}

        perf = metadata.get("performance_metrics", {}) or {}

        accuracy = (
            perf.get("accuracy_percent")
            or perf.get("accuracy")
        )

        mae = (
            perf.get("mae_price_eur")
            or perf.get("mae_euros")
            or perf.get("mae")
        )

        mape = (
            perf.get("mape_percent")
            or perf.get("mape")
        )

        r2 = (
            perf.get("r2_price")
            or perf.get("r2_log")
            or perf.get("r2_score")
            or perf.get("r2")
        )

        rmse_price = (
            perf.get("rmse_price_eur")
            or perf.get("rmse_euros")
            or perf.get("rmse")
        )

        model_type = metadata.get("model_type", "RandomForestRegressor")
        tuned_with = metadata.get("tuning_method")

        return {
            "status": "ok",
            "trained_at": metadata.get("saved_at") or metadata.get("trained_at"),
            "model_type": model_type,
            "tuning_method": tuned_with,
            "accuracy_percent": accuracy,
            "mae_price_eur": mae,
            "rmse_price_eur": rmse_price,
            "mape_percent": mape,
            "r2": r2,
            "hyperparameters": metadata.get("best_parameters")
                               or metadata.get("hyperparameters"),
            "training_info": metadata.get("training_info"),
        }
