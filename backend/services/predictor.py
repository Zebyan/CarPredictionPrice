# backend/services/predictor.py

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from backend.model_loader import ModelLoader
from backend.config import settings

logger = logging.getLogger(__name__)


# =====================================================================
# HELPER: citesc metricile din metadata (fallback dacă lipsesc în settings)
# =====================================================================

def _get_error_stats() -> tuple[float, float]:
    """
    Returnează (mae_price_eur, mape_percent) din:
      1) settings (dacă sunt setate acolo),
      2) metadata.performance_metrics,
      3) sau valori default hardcodate.
    """
    # 1) încerc din settings
    mae = getattr(settings, "model_mae", None)
    mape = getattr(settings, "model_mape", None)

    # 2) dacă lipsesc, încercăm din metadata
    if mae is None or mape is None:
        meta = ModelLoader.load_metadata() or {}
        perf = meta.get("performance_metrics", {}) or {}

        if mae is None:
            mae = (
                perf.get("mae_price_eur")
                or perf.get("mae_euros")
                or perf.get("mae")
            )
        if mape is None:
            mape = (
                perf.get("mape_percent")
                or perf.get("mape")
            )

    # 3) fallback final
    if mae is None:
        mae = 1800.0
    if mape is None:
        mape = 23.0

    return float(mae), float(mape)


# =====================================================================
# PREDICȚIE
# =====================================================================

def predict_price(features_df: pd.DataFrame) -> float:
    """
    Predict car price from engineered features.

    - features_df: DataFrame cu o singură linie, returnat de engineer_features().
    - Modelul este un Pipeline (preprocessor + RandomForest) încărcat via ModelLoader.
    - Modelul prezice log(pret + 1); noi convertim înapoi în EUR cu expm1.
    """
    model = ModelLoader.load_model()
    if model is None:
        raise RuntimeError("Model not available (ModelLoader.load_model() returned None)")

    if features_df is None or features_df.empty:
        raise RuntimeError("Empty features DataFrame passed to predict_price()")

    try:
        logger.info(
            "Predict: input shape=%s, columns=%s",
            features_df.shape,
            list(features_df.columns),
        )

        # model.predict -> log(pret + 1)
        y_pred_log = model.predict(features_df)[0]
        y_pred_price = np.expm1(y_pred_log)

        price = int(y_pred_price)
        logger.info("Model prediction: log=%0.4f, price=%0.2f EUR", y_pred_log, price)
        return price

    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise RuntimeError(f"Prediction failed: {str(e)}")


# =====================================================================
# INTERVAL DE ÎNCREDERE (PERCENTAGE-BASED)
# =====================================================================

def price_confidence_interval(
    predicted_price: float,
    features_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Calculează interval de încredere în jurul predicției.

    Logică:
      - Folosim MAPE global (~eroare procentuală medie) => margină proporțională cu prețul.
      - margin = predicted_price * (MAPE / 100)
      - min/max = predicted_price ± margin
      - Pentru stabilitate, aplicăm un mic floor absolut (10% din MAE),
        ca să nu avem intervale ridicol de mici la prețuri foarte mici.

    Parametri:
      predicted_price: prețul prezis (EUR)
      features_df: DataFrame cu features pentru mașina curentă (deocamdată nefolosit aici,
                   dar păstrat pentru extensii viitoare per-segment).

    Returnează:
      dict cu cheile: min_price, max_price, margin, confidence
    """
    mae, mape = _get_error_stats()  # mae în EUR, mape în %

    # margină procentuală
    pct = mape / 100.0
    margin_pct = predicted_price * pct

    # floor absolut mic: 10% din MAE, ca să nu fie intervalul prea mic pe mașini foarte ieftine
    abs_floor = mae * 0.10

    margin = max(margin_pct, abs_floor)

    min_price = max(0.0, predicted_price - margin)
    max_price = predicted_price + margin

    confidence = float(getattr(settings, "model_confidence", 77.13))

    return {
        "min_price": float(round(min_price)),
        "max_price": float(round(max_price)),
        "margin": float(round(margin)),
        "confidence": confidence,
    }


# =====================================================================
# METADATE PENTRU UI / HEALTH
# =====================================================================

def get_prediction_metadata() -> Dict[str, Any]:
    """
    Info despre model pentru UI / health endpoint.

    Încearcă să citească cât mai mult din ModelLoader.get_model_info()
    și completează cu fallback-uri.
    """
    info = {}
    try:
        info = ModelLoader.get_model_info()
    except Exception as e:
        logger.warning("Could not get model info from ModelLoader: %s", e)

    mae, mape = _get_error_stats()

    return {
        "model_type": info.get("model_type", "RandomForestRegressor"),
        "tuning_method": info.get("tuning_method"),
        "trained_at": info.get("trained_at"),
        "mae_price_eur": info.get("mae_price_eur", mae),
        "rmse_price_eur": info.get("rmse_price_eur"),
        "r2_score": info.get("r2"),
        "mape_percent": info.get("mape_percent", mape),
        "accuracy_percent": info.get("accuracy_percent"),
        "hyperparameters": info.get("hyperparameters"),
        "training_info": info.get("training_info"),
        "note": "Predictions are point estimates; use confidence interval for decision-making",
    }
