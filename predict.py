# Updated predict.py for LightGBM 3-class model
import joblib
import pandas as pd
import os

BASE_DIR = r"D:\crime_project2"
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load LightGBM model + label encoder
model = joblib.load(os.path.join(MODEL_DIR, "lgbm_3groups_model.joblib"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))


# -----------------------------
# Feature engineering (MUST MATCH preprocess.py)
# -----------------------------
def build_features(year, month, day, hour, beat, district, ward, community_area):

    ts = pd.Timestamp(year=year, month=month, day=day)

    day_of_week = ts.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0

    # Season mapping
    if month in [3, 4, 5]:
        season = 1
    elif month in [6, 7, 8]:
        season = 2
    elif month in [9, 10, 11]:
        season = 3
    else:
        season = 4

    # Hour category
    if hour < 6:
        hour_group = 0
    elif hour < 12:
        hour_group = 1
    elif hour < 18:
        hour_group = 2
    else:
        hour_group = 3

    # HOTSPOT flag (same as preprocess.py)
    TOP_BEATS = [332, 2523, 1933, 224, 1022, 1113, 414, 2023, 1221, 925]
    hotspot_area = 1 if beat in TOP_BEATS else 0

    row = pd.DataFrame([{
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "season": season,
        "hour_group": hour_group,
        "beat": beat,
        "district": district,
        "ward": ward,
        "community_area": community_area,
        "hotspot_area": hotspot_area
    }])

    return row


# -----------------------------
# Prediction function
# -----------------------------
def predict_crime(year, month, day, hour, beat, district, ward, community_area):
    # Build feature row
    X = build_features(year, month, day, hour, beat, district, ward, community_area)

    # Predict (LightGBM returns probabilities)
    pred_class = model.predict(X).argmax(axis=1)

    # Convert encoded label to crime group
    output = label_encoder.inverse_transform(pred_class)[0]

    return output


# -----------------------------
# Manual test (optional)
# -----------------------------
if __name__ == "__main__":
    print("\nTesting predict.py...\n")
    result = predict_crime(
        year=2025,
        month=11,
        day=15,
        hour=13,
        beat=111,
        district=11,
        ward=29,
        community_area=35
    )
    print("Predicted Crime Group =", result)
