import pandas as pd
import numpy as np
import os
import joblib
from sklearn.cluster import KMeans

# ---------------------------
# PATHS
# ---------------------------
BASE_DIR = r"D:\crime_project2"
RAW_FILE = os.path.join(BASE_DIR, "ijzp-q8t2 (4).csv")   # your main dataset
OUT_FILE = os.path.join(BASE_DIR, "cleaned_crimes.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------
# LOCATION GROUP FUNCTION
# ---------------------------
def location_group_func(loc):
    if pd.isna(loc):
        return "UNKNOWN"

    loc = str(loc).upper()

    if any(k in loc for k in ["STREET", "SIDEWALK", "PARKING", "DRIVE"]):
        return "STREET"

    if any(k in loc for k in ["RESIDENCE", "APARTMENT", "HOME", "HOUSE"]):
        return "RESIDENCE"

    if any(k in loc for k in ["BUSINESS", "STORE", "SHOP", "RESTAURANT", "COMMERCIAL"]):
        return "BUSINESS"

    if any(k in loc for k in ["PUBLIC", "SCHOOL", "GOVERNMENT", "HOSPITAL"]):
        return "PUBLIC"

    return "OTHER"


# ---------------------------
# ADD SPATIAL CLUSTERS USING KMEANS
# ---------------------------
def add_spatial_clusters(df, n_clusters=30):
    print(f"Clustering {len(df)} rows into {n_clusters} spatial clusters...")

    coords = df[["latitude", "longitude"]].values

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    df["spatial_cluster"] = kmeans.fit_predict(coords)

    # Save KMeans model for later use
    joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_spatial.joblib"))
    print("Saved KMeans spatial model.")

    return df


# ---------------------------
# MAIN PREPROCESS FUNCTION
# ---------------------------
def preprocess():
    print("Loading raw dataset...")
    df = pd.read_csv(RAW_FILE)
    print("Rows loaded:", len(df))

    # Standardize column names
    df.columns = [c.lower() for c in df.columns]

    # Fix date
    print("Fixing date column...")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Remove rows with no date
    df = df.dropna(subset=["date"])

    # Extract datetime components
    print("Extracting features...")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["hour"] = df["date"].dt.hour

    # Location group
    print("Adding location_group...")
    df["location_group"] = df["location_description"].apply(location_group_func)

    # Remove rows without coordinates
    print("Removing rows without latitude/longitude...")
    df = df.dropna(subset=["latitude", "longitude"])

    # Add spatial clusters
    print("Adding spatial KMeans clusters...")
    df = add_spatial_clusters(df, n_clusters=30)

    # Save output
    print("Saving cleaned file:", OUT_FILE)
    df.to_csv(OUT_FILE, index=False)
    print("Preprocessing complete!")


# ---------------------------
# RUN SCRIPT
# ---------------------------
if __name__ == "__main__":
    preprocess()
