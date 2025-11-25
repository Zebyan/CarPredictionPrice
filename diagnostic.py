# ============================================================================
# diagnostic_model.py - Check what's in the saved model file
# ============================================================================
# Run this in the notebook or as a script
#

import joblib
import os

print("="*70)
print("MODEL FILE DIAGNOSTIC")
print("="*70)

model_path = "backend/models_storage/best_model.pkl"

print(f"\n1. File info:")
print(f"   Path: {model_path}")
print(f"   Exists: {os.path.exists(model_path)}")
print(f"   Size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

# Load and inspect
print(f"\n2. Loading model...")
try:
    loaded = joblib.load(model_path)
    print(f"   ✓ Loaded: {type(loaded)}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Check structure
print(f"\n3. Pipeline structure:")
print(f"   Steps: {list(loaded.named_steps.keys())}")

# Check preprocessor
print(f"\n4. Checking preprocessor...")
preprocessor = loaded.named_steps["preprocessor"]
print(f"   Type: {type(preprocessor)}")
print(f"   Has n_features_in_: {hasattr(preprocessor, 'n_features_in_')}")

if hasattr(preprocessor, 'n_features_in_'):
    print(f"   ✓ n_features_in_: {preprocessor.n_features_in_}")
else:
    print(f"   ✗ NOT FITTED - This is the problem!")

# Check transformers
print(f"\n5. Checking transformers inside preprocessor...")
try:
    transformers = preprocessor.named_transformers_
    print(f"   Transformers: {list(transformers.keys())}")
    
    for name, transformer in transformers.items():
        print(f"\n   {name}:")
        print(f"     Type: {type(transformer)}")
        print(f"     Fitted: {hasattr(transformer, 'n_features_in_')}")
        if hasattr(transformer, 'n_features_in_'):
            print(f"     n_features_in_: {transformer.n_features_in_}")
except Exception as e:
    print(f"   Error: {e}")

# Check model
print(f"\n6. Checking RandomForest model...")
model = loaded.named_steps["model"]
print(f"   Type: {type(model)}")
print(f"   Fitted: {hasattr(model, 'n_features_in_')}")
if hasattr(model, 'n_features_in_'):
    print(f"   ✓ n_features_in_: {model.n_features_in_}")

# Try a prediction
print(f"\n7. Testing prediction...")
try:
    import pandas as pd
    import numpy as np
    
    # Create sample input
    X_sample = pd.DataFrame({
        'capacitate motor': [1600],
        'putere': [110],
        'rulaj': [120000],
        'an fabricatie': [2018],
        'age': [7],
        'km_per_year': [17142],
        'hp_per_liter': [68.75],
        'brand_popularity': [100],
        'model_popularity': [80],
        'is_premium': [0],
        'is_automatic': [0],
        'age2': [49],
        'rulaj2': [14400000000],
        'marca': ['volkswagen'],
        'model': ['golf'],
        'combustibil': ['diesel'],
        'caroserie': ['hatchback'],
        'culoare': ['negru'],
        'cutie viteza': ['manuala'],
        'rulaj_cat': ['medium'],
        'age_category': ['medium'],
        'engine_category': ['small'],
        'power_category': ['normal'],
        'caroserie_grouped': ['Hatch'],
        'fuel_grouped': ['diesel'],
        'segment': ['C'],
    })
    
    print(f"   Input shape: {X_sample.shape}")
    
    pred = loaded.predict(X_sample)
    print(f"   ✓ Prediction successful: {pred}")
    print(f"   ✓ Price (EUR): €{np.expm1(pred[0]):,.0f}")
    
except Exception as e:
    print(f"   ✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)