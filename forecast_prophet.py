import os
import joblib
import pandas as pd

BASE_DIR = r"D:\crime_project2"
PROPHET_DIR = os.path.join(BASE_DIR, "prophet_models")
os.makedirs(PROPHET_DIR, exist_ok=True)

# Updated frequencies (pandans v2+)
FREQ_MAP = {
    "M": "ME",   # Month-End
    "Q": "QE",   # Quarter-End
    "A": "YE"    # Year-End
}

def train_prophet(period="M"):
    df = pd.read_csv(os.path.join(BASE_DIR, "cleaned_crimes.csv"))
    df.columns = [c.lower() for c in df.columns]

    if "date" not in df.columns:
        raise ValueError("date column required")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    freq = FREQ_MAP[period]

    print(f"Training Prophet for {period} ({freq})...")

    # Resample time series
    df2 = (
        df.set_index("date")
          .resample(freq)
          .size()
          .rename("y")
          .reset_index()
    )

    df2.columns = ["ds", "y"]

    if len(df2) < 3:
        print(f"⚠ Skipping {period} (Not enough time points: {len(df2)})")
        return None

    # Load Prophet
    try:
        from prophet import Prophet
    except Exception as e:
        raise ImportError("Install prophet package: conda install -c conda-forge prophet") from e

    m = Prophet()
    m.fit(df2)

    joblib.dump(m, os.path.join(PROPHET_DIR, f"prophet_{period}.joblib"))
    print(f"✔ Saved Prophet model for period {period}")

    return m


if __name__ == "__main__":
    for p in ["M", "Q", "A"]:
        train_prophet(period=p)
