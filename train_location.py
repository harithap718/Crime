import pandas as pd
import os, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = r"D:\crime_project2"
DATAFILE = os.path.join(BASE_DIR, "cleaned_crimes.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_location_model():
    df = pd.read_csv(DATAFILE)

    if "spatial_cluster" not in df.columns:
        raise ValueError("spatial_cluster missing. Run preprocess.py again with updated version.")

    features = ["latitude", "longitude", "beat", "district", "ward", "community_area"]
    X = df[features]
    y = df["spatial_cluster"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print("Training RandomForest for spatial cluster prediction...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Spatial Cluster Accuracy:", acc)
    print(classification_report(y_test, preds))

    joblib.dump(model, os.path.join(MODEL_DIR, "rf_spatial_location.joblib"))
    print("Saved model to", MODEL_DIR)

if __name__ == "__main__":
    train_location_model()
