import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os

BASE_DIR = r"D:\crime_project2"
DATA_PATH = os.path.join(BASE_DIR, "cleaned_crimes.csv")
HOT_DIR = os.path.join(BASE_DIR, "hotspots")
os.makedirs(HOT_DIR, exist_ok=True)


def find_hotspots(n_clusters=30):
    print("📌 Loading cleaned_crimes.csv ...")
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lower() for c in df.columns]

    # Latitude & longitude required
    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError("❌ Missing latitude/longitude in cleaned_crimes.csv")

    print("📌 Sampling 300k rows to speed up clustering...")
    df_sample = df.sample(min(300000, len(df)), random_state=42)

    coords = df_sample[["latitude", "longitude"]].values

    print(f"📌 Running K-Means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_sample["cluster"] = kmeans.fit_predict(coords)

    print("📌 Saving cluster labels back to full dataset ...")
    # Predict for entire dataset
    full_coords = df[["latitude", "longitude"]].values
    df["cluster"] = kmeans.predict(full_coords)

    # Save updated dataset
    updated_path = os.path.join(BASE_DIR, "cleaned_crimes_with_clusters.csv")
    df.to_csv(updated_path, index=False)
    print(f"✔ Saved updated dataset with clusters → {updated_path}")

    # Save cluster centers for REGION MAPPING
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=["latitude", "longitude"])
    centers["cluster_id"] = centers.index
    centers_path = os.path.join(HOT_DIR, "cluster_centers.csv")
    centers.to_csv(centers_path, index=False)
    print(f"✔ Saved cluster centers → {centers_path}")

    # Hotspot summary
    print("📌 Generating hotspot summary...")
    report = df["cluster"].value_counts().reset_index()
    report.columns = ["cluster_id", "count"]
    report = report.sort_values("count", ascending=False)
    report_path = os.path.join(HOT_DIR, "hotspot_report.csv")
    report.to_csv(report_path, index=False)

    print(f"✔ Saved hotspot_report.csv → {report_path}")
    print("🎯 Hotspot Clustering Completed Successfully!")

    return report


if __name__ == "__main__":
    find_hotspots(n_clusters=30)
