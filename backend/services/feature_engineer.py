import numpy as np
import pandas as pd
from typing import Dict, Any

from backend.models.schemas import CarPredictionRequest

# ============================================================
# CONSTANTS
# ============================================================

PREMIUM_BRANDS = [
    "audi", "bmw", "mercedes-benz", "volvo", "tesla",
    "porche", "lamborghini", "jaguar"
]
BUDGET_BRANDS = [
    "dacia", "skoda", "fiat", "renault", "opel",
    "suzuki", "kia", "hyundai", "chevrolet", "seat"
]
STANDARD_BRANDS = [
    "citroen", "ford", "honda", "mazda", "mitsubishi",
    "nissan", "peugeot", "toyota", "volkswagen"
]

POPULAR_COLORS = [
    "Alb",
    "Verde",
    "Gri",
    "Argintiu",
    "Maro / Bej",
    "Alta culoare",
    "Negru",
    "Albastru",
    "Rosu",
]

MODEL_COUNTS: Dict[str, int] = {}  # Populate from training notebook
RARE_MODEL_THRESHOLD = 50
RARE_MODEL_MEAN_FREQ = 50.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _assign_era(year: int) -> str:
    """Assign car era based on manufacturing year."""
    if year < 1995:
        return "vintage"
    elif year < 2005:
        return "older_standard"
    elif year < 2010:
        return "mid_standard"
    elif year < 2015:
        return "modern_early"
    else:
        return "modern_recent"


def _rulaj_cat_from_km(km: float) -> str:
    """Categorize mileage into bins."""
    if km < 70_000:
        return "low"
    if km < 150_000:
        return "medium"
    if km < 250_000:
        return "high"
    return "very_high"


def _age_category_from_age(age: float) -> str:
    """Categorize car age into bins."""
    if age <= 3:
        return "new"
    if age <= 10:
        return "medium"
    if age <= 20:
        return "old"
    return "very_old"


def _engine_type_from_fuel(fuel: str) -> str:
    """Determine engine type from fuel."""
    f = (fuel or "").lower()
    if "electric" in f:
        return "electric"
    if "hibrid" in f or "hybrid" in f:
        return "hybrid"
    return "ice"


# ============================================================
# MAIN FEATURE ENGINEERING FUNCTION
# ============================================================

def engineer_features(data: CarPredictionRequest) -> pd.DataFrame:
    """
    Reproduce all feature engineering from training notebook.
    Returns a DataFrame with categorical and numeric features.
    The model's ColumnTransformer will handle OneHotEncoding automatically.
    """

    # --------------------------------------------------------
    # 0. PARSE & FILL NA
    # --------------------------------------------------------
    marca_raw = (data.marca or "").strip()
    model_raw = (data.model or "").strip()
    combustibil = (data.combustibil or "Unknown").strip()
    caroserie = (data.caroserie or "").strip()
    culoare = (data.culoare or "").strip()
    cutie_viteza = (data.cutie_viteza or "").strip()
    
    rulaj = float(data.rulaj)
    an_fabricatie = int(data.an_fabricatie)
    putere = float(data.putere)
    capacitate_motor = float(data.capacitate_motor)

    # Fill missing with defaults
    if not cutie_viteza:
        cutie_viteza = "Manuala"
    if not culoare:
        culoare = "Negru"

    # --------------------------------------------------------
    # 1. BASE NUMERIC FEATURES
    # --------------------------------------------------------
    age_years = 2024 - an_fabricatie
    current_year = 2024
    age = max(0, current_year - an_fabricatie)
    
    disp_liters = capacitate_motor / 1000.0 if capacitate_motor > 0 else 1.0
    mileage_per_year = rulaj / (age_years + 1.0)

    # --------------------------------------------------------
    # 2. BINNING (converts to string for categorical treatment)
    # --------------------------------------------------------
    engine_size_bin = str(pd.cut(
        [capacitate_motor],
        bins=[0, 1500, 1800, 2000, 3000, 6000],
        labels=["<1500", "1500–1800", "1800–2000", "2000–3000", ">3000"],
        include_lowest=True,
    )[0])

    hp_bin = str(pd.cut(
        [putere],
        bins=[0, 100, 150, 200, 300, 1000],
        labels=["<100", "100–150", "150–200", "200–300", ">300"],
        include_lowest=True,
    )[0])

    mileage_bin = str(pd.cut(
        [rulaj],
        bins=[0, 100000, 200000, 300000, 800000],
        labels=["<100k", "100–200k", "200–300k", ">300k"],
        include_lowest=True,
    )[0])

    age_bin = str(pd.cut(
        [an_fabricatie],
        bins=[1950, 2000, 2005, 2010, 2015, 2018, 2020, 2025],
        labels=[
            "1950–2000", "2000–2005", "2005–2010",
            "2010–2015", "2015–2018", "2018–2020", "2020–2025"
        ],
        include_lowest=True,
    )[0])

    # --------------------------------------------------------
    # 3. OUTLIER FLAGS
    # --------------------------------------------------------
    large_engine_flag = int(capacitate_motor > 3000)
    small_engine_flag = int(capacitate_motor < 1000)
    high_hp_flag = int(putere > 220)
    low_hp_flag = int(putere < 50)
    high_mileage_flag = int(rulaj > 350000)
    low_mileage_flag = int(rulaj < 5000)
    new_car_flag = int(an_fabricatie >= 2018)
    vintage_flag = int(an_fabricatie < 1995)

    # --------------------------------------------------------
    # 4. ERA & ERA DUMMIES
    # --------------------------------------------------------
    car_era = _assign_era(an_fabricatie)
    era_vintage = int(car_era == "vintage")
    era_older_standard = int(car_era == "older_standard")
    era_mid_standard = int(car_era == "mid_standard")
    era_modern_early = int(car_era == "modern_early")
    era_modern_recent = int(car_era == "modern_recent")

    # --------------------------------------------------------
    # 5. BRAND CATEGORY & DUMMIES
    # --------------------------------------------------------
    marca_lower = marca_raw.lower()
    if marca_lower in PREMIUM_BRANDS:
        brand_category = "premium"
    elif marca_lower in BUDGET_BRANDS:
        brand_category = "budget"
    else:
        brand_category = "standard"

    brand_premium = int(brand_category == "premium")
    brand_budget = int(brand_category == "budget")
    brand_standard = int(brand_category == "standard")
    is_premium_brand = brand_premium
    is_budget_brand = brand_budget

    # Numeric encoding for brand_category (for model training consistency)
    brand_category_num = {"standard": 0, "budget": 1, "premium": 2}[brand_category]

    # --------------------------------------------------------
    # 6. MODEL SIMPLIFICATION & FREQUENCY
    # --------------------------------------------------------
    model_lower = model_raw.lower()
    count = MODEL_COUNTS.get(model_lower, None)
    if count is None:
        model_simplified = "UNKNOWN"
        model_frequency = RARE_MODEL_MEAN_FREQ
    else:
        model_simplified = model_lower if count >= RARE_MODEL_THRESHOLD else "UNKNOWN"
        model_frequency = float(count)

    # --------------------------------------------------------
    # 7. CAROSERIE, FUEL, TRANSMISSION FLAGS
    # --------------------------------------------------------
    is_suv = int(caroserie.lower() == "suv")
    is_sport_body = int(caroserie.lower() in ["coupe", "cabrio"])
    is_large_body = int(caroserie.lower() in ["suv", "pickup", "minibus", "monovolum"])
    is_sedan = int(caroserie.lower() in ["sedan", "berlina"])

    is_electric = int("electric" in combustibil.lower())
    is_hybrid = int(any(h in combustibil.lower() for h in ["hybrid", "hibrid"]))
    is_diesel = int("diesel" in combustibil.lower())
    is_petrol = int("benzina" in combustibil.lower())

    is_automatic = int(cutie_viteza.lower() in ["automata", "automatic"])
    is_manual = int(cutie_viteza.lower() == "manuala")

    # --------------------------------------------------------
    # 8. COLOR ENCODING (as numeric one-hot, not categorical)
    # --------------------------------------------------------
    # Colors were already one-hot encoded in training as numeric features
    # Build color dummies using exact naming from training metadata
    color_features = {}
    
    # Map POPULAR_COLORS to their exact feature names from metadata
    color_name_map = {
        "Negru": "color_negru",
        "Gri": "color_gri",
        "Alb": "color_alb",
        "Albastru": "color_albastru",
        "Rosu": "color_rosu",
        "Argintiu": "color_argintiu",
        "Maro / Bej": "color_maro_/_bej",  # Exact naming from metadata
        "Alta culoare": "color_alta_culoare",
        "Verde": "color_verde",
    }
    
    for original_color, feature_name in color_name_map.items():
        color_features[feature_name] = int(culoare == original_color)

    # --------------------------------------------------------
    # 9. RATIO & DERIVED FEATURES
    # --------------------------------------------------------
    power_to_displacement = putere / disp_liters
    power_per_liter = putere / disp_liters
    engine_efficiency_flag = int((capacitate_motor < 1500) and (putere >= 100))

    # --------------------------------------------------------
    # 10. ECO & INTERACTION FEATURES
    # --------------------------------------------------------
    eco_recent = int((an_fabricatie >= 2015) and (putere < 50))
    new_and_powerful = int((an_fabricatie >= 2018) and (putere > 200))
    old_collectible = int((an_fabricatie < 1995) and (putere > 150))
    high_mileage_old = int((rulaj > 300000) and (an_fabricatie < 2010))
    premium_new = int((is_premium_brand == 1) and (an_fabricatie >= 2018))
    budget_new = int((is_budget_brand == 1) and (an_fabricatie >= 2018))
    modern_suv_auto = int((is_suv == 1) and (an_fabricatie >= 2015) and (is_automatic == 1))

    # --------------------------------------------------------
    # 11. LOG FEATURES
    # --------------------------------------------------------
    log_mileage = float(np.log1p(rulaj))
    log_engine_size = float(np.log1p(capacitate_motor))

    # --------------------------------------------------------
    # 12. CATEGORIZATION FEATURES
    # --------------------------------------------------------
    rulaj_cat = _rulaj_cat_from_km(rulaj)
    age_category = _age_category_from_age(age)
    engine_type = _engine_type_from_fuel(combustibil)
    segment = "unknown"

    # --------------------------------------------------------
    # 13. BUILD FEATURE DICTIONARY
    # --------------------------------------------------------
    features_dict = {
        # CATEGORICAL
        "marca": marca_raw,
        "brand_category": brand_category,
        "model_simplified": model_simplified,
        "caroserie": caroserie,
        "combustibil": combustibil,
        "cutie viteza": cutie_viteza,
        "car_era": car_era,
        "engine_size_bin": engine_size_bin,
        "hp_bin": hp_bin,
        "mileage_bin": mileage_bin,
        "age_bin": age_bin,
        "brand_category.1": str({"standard": 0, "budget": 1, "premium": 2}[brand_category]),

        # NUMERICAL
        "capacitate motor": capacitate_motor,
        "putere": putere,
        "rulaj": rulaj,
        "an fabricatie": an_fabricatie,
        "age_years": age_years,
        "large_engine_flag": large_engine_flag,
        "small_engine_flag": small_engine_flag,
        "high_hp_flag": high_hp_flag,
        "low_hp_flag": low_hp_flag,
        "high_mileage_flag": high_mileage_flag,
        "low_mileage_flag": low_mileage_flag,
        "new_car_flag": new_car_flag,
        "vintage_flag": vintage_flag,
        "is_premium_brand": is_premium_brand,
        "is_budget_brand": is_budget_brand,
        "is_suv": is_suv,
        "is_sport_body": is_sport_body,
        "is_large_body": is_large_body,
        "is_sedan": is_sedan,
        "is_electric": is_electric,
        "is_hybrid": is_hybrid,
        "is_diesel": is_diesel,
        "is_petrol": is_petrol,
        "is_automatic": is_automatic,
        "is_manual": is_manual,
        "engine_efficiency_flag": engine_efficiency_flag,
        "new_and_powerful": new_and_powerful,
        "old_collectible": old_collectible,
        "high_mileage_old": high_mileage_old,
        "premium_new": premium_new,
        "budget_new": budget_new,
        "modern_suv_auto": modern_suv_auto,
        "eco_recent": eco_recent,
        "era_mid_standard": era_mid_standard,
        "era_modern_early": era_modern_early,
        "era_modern_recent": era_modern_recent,
        "era_older_standard": era_older_standard,
        "era_vintage": era_vintage,
        "brand_budget": brand_budget,
        "brand_premium": brand_premium,
        "brand_standard": brand_standard,
        "power_to_displacement": power_to_displacement,
        "power_per_liter": power_per_liter,
        "mileage_per_year": mileage_per_year,
        "model_frequency": model_frequency,
        "log_mileage": log_mileage,
        "log_engine_size": log_engine_size,
    }

    # Add color features (0/1 numerice)
    features_dict.update(color_features)

    # --------------------------------------------------------
    # 14. RETURN AS DATAFRAME (single row)
    # --------------------------------------------------------
    df = pd.DataFrame([features_dict])

    categorical_cols = [
        "marca",
        "brand_category",
        "model_simplified",
        "caroserie",
        "combustibil",
        "cutie viteza",
        "car_era",
        "engine_size_bin",
        "hp_bin",
        "mileage_bin",
        "age_bin",
        "brand_category.1",
    ]
    for col in categorical_cols:
        df[col] = df[col].astype("string")

    return df
