import os
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = r"D:\crime_project2"
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Map crime types into 3 groups
# -----------------------------
def map_crime(c):
    c = str(c).upper()

    violent = [
        "ASSAULT", "BATTERY", "HOMICIDE", "KIDNAPPING",
        "CRIMINAL SEXUAL ASSAULT", "SEX OFFENSE",
        "OFFENSE INVOLVING CHILDREN", "WEAPONS VIOLATION",
        "DOMESTIC", "PUBLIC PEACE"
    ]

    property_c = [
        "THEFT", "BURGLARY", "MOTOR VEHICLE THEFT",
        "CRIMINAL DAMAGE", "CRIMINAL TRESPASS",
        "ARSON", "DECEPTIVE PRACTICE", "ROBBERY",
        "SHOPLIFTING", "VANDALISM"
    ]

    if any(x in c for x in violent):
        return "VIOLENT_CRIME"
    if any(x in c for x in property_c):
        return "PROPERTY_CRIME"
    
    return "OTHER_CRIME"


# -----------------------------
# Feature Engineering
# -----------------------------
def prepare_features(df):
    df = df.copy()

    features = [
        "year","month","day","hour",
        "day_of_week","is_weekend","season","hour_group",
        "beat","district","ward","community_area",
        "hotspot_area"
    ]

    X = df[features].fillna(0)
    y = df["crime_group"].astype(str)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    return X, y_enc, le


# -----------------------------
# Training
# -----------------------------
def train():
    print("Loading cleaned dataset...")
    df = pd.read_csv(os.path.join(BASE_DIR, "cleaned_crimes.csv"))
    df.columns = [c.lower() for c in df.columns]

    print("Mapping crimes into 3 super-groups...")
    df["crime_group"] = df["primary_type"].apply(map_crime)

    print("Sampling 700k rows for training...")
    df = df.sample(700000, random_state=42)

    print("Preparing features...")
    X, y, le = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training LightGBM model...")

    params = {
        "objective": "multiclass",
        "num_class": len(set(y)),
        "learning_rate": 0.05,
        "num_leaves": 50,
        "max_depth": -1,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "boosting": "gbdt",
        "metric": "multi_logloss",
        "verbosity": -1
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)

    model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=300)

    joblib.dump(model, os.path.join(MODEL_DIR, "lgbm_3groups_model.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))

    print("\n✔ Model Training Complete!")
    print("Classes:", list(le.classes_))

    train_pred = model.predict(X_train).argmax(axis=1)
    test_pred = model.predict(X_test).argmax(axis=1)

    print("Train Accuracy:", (train_pred == y_train).mean())
    print("Test Accuracy :", (test_pred == y_test).mean())


if __name__ == "__main__":
    train()
