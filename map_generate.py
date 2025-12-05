# map_generate.py (Color-coded Clusters + Heatmap + Region Names)
import os
import pandas as pd
import folium
from folium.plugins import HeatMap

BASE_DIR = r"D:\crime_project2"
OUT_DIR = os.path.join(BASE_DIR, "maps")
os.makedirs(OUT_DIR, exist_ok=True)

# Your uploaded cluster centers file
CLUSTER_CENTERS_FILE = "/mnt/data/cluster_centers.csv"

# Region name map (same as Streamlit)
REGION_MAP = {
    0: "Garfield Park / Kedzie",
    1: "Englewood / Bronzeville",
    2: "North Side – Albany Park",
    3: "South Deering / Far South",
    4: "Downtown – Loop / River North",
    5: "South Shore",
    6: "Outside Chicago",
    7: "Cicero Border / West Side",
    8: "O'Hare – East Suburbs",
    9: "Auburn Gresham",
    10: "South Chicago",
    11: "Logan Square",
    12: "Belmont Cragin / Northwest",
    13: "Bridgeport / Armour Square",
    14: "Hyde Park / Woodlawn",
    15: "McKinley Park",
    16: "O'Hare Airport",
    17: "Little Village North",
    18: "Uptown / Edgewater",
    19: "West Englewood",
    20: "North Lawndale",
    21: "Chatham",
    22: "Pilsen / Lower West",
    23: "Hermosa / Cragin",
    24: "Rogers Park",
    25: "Grand Crossing",
    26: "East Side",
    27: "South Chicago – West",
    28: "Wicker Park / Bucktown",
    29: "Humboldt Park West"
}

# -----------------------
# Beautiful Color Palette
# -----------------------
CLUSTER_COLORS = [
    "#FF0000", "#00C8FF", "#2BAE66", "#9B59B6", "#F39C12",
    "#E74C3C", "#1ABC9C", "#3498DB", "#8E44AD", "#2ECC71",
    "#D35400", "#C0392B", "#16A085", "#2980B9", "#F1C40F",
    "#E67E22", "#7D3C98", "#117864", "#1B4F72", "#B7950B",
    "#DC7633", "#AF7AC5", "#48C9B0", "#2471A3", "#52BE80",
    "#D68910", "#884EA0", "#5DADE2", "#58D68D", "#F4D03F"
]


def load_cluster_centers():
    """Load cluster centers file from /mnt/data"""
    if not os.path.exists(CLUSTER_CENTERS_FILE):
        return None

    df = pd.read_csv(CLUSTER_CENTERS_FILE)
    df.columns = [c.lower() for c in df.columns]

    if "lat" not in df.columns and "latitude" in df.columns:
        df["lat"] = df["latitude"]

    if "lon" not in df.columns and "longitude" in df.columns:
        df["lon"] = df["longitude"]

    return df


def create_heatmap(n=50000, out="heatmap_color_clusters.html"):
    """Generate heatmap + color-coded cluster markers"""

    # Load crime data for heatmap
    df = pd.read_csv(os.path.join(BASE_DIR, "cleaned_crimes.csv"))
    df = df.dropna(subset=["latitude", "longitude"])

    coords = df[["latitude", "longitude"]].head(n).values.tolist()

    # Base Map
    m = folium.Map(
        location=[41.8781, -87.6298],
        zoom_start=11,
        tiles="CartoDB positron"
    )

    # Heatmap Layer
    HeatMap(coords, radius=8, blur=4, min_opacity=0.3).add_to(m)

    # Load cluster centers
    centers = load_cluster_centers()
    if centers is not None:

        for _, row in centers.iterrows():
            cid = int(row["cluster_id"])
            lat = float(row["lat"])
            lon = float(row["lon"])

            color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
            region = REGION_MAP.get(cid, "Unknown Region")

            popup_html = f"""
            <b>Cluster ID:</b> {cid}<br>
            <b>Region:</b> {region}<br>
            <b>Latitude:</b> {lat:.6f}<br>
            <b>Longitude:</b> {lon:.6f}<br>
            """

            # Color-coded marker
            folium.CircleMarker(
                location=[lat, lon],
                radius=10,
                color=color,
                weight=3,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=folium.Popup(popup_html, max_width=350),
            ).add_to(m)

    # Save output
    out_path = os.path.join(OUT_DIR, out)
    m.save(out_path)
    print(f"✔ Saved color-coded heatmap to: {out_path}")


if __name__ == "__main__":
    create_heatmap()
