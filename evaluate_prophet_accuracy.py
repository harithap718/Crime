# evaluate_prophet_accuracy.py
import os, joblib, pandas as pd, numpy as np, math
from prophet import Prophet

BASE_DIR = r"D:\crime_project2"
INFILE = os.path.join(BASE_DIR, "cleaned_crimes.csv")

# ---------- Metrics ----------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return None
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def rmse(y_true, y_pred):
    return math.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))


# ---------- Evaluation Function ----------
def evaluate(freq, horizon, label):
    print(f"\n===== {label} ({freq}) =====")

    df = pd.read_csv(INFILE, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    # Aggregate
    ts = df.set_index("date").resample(freq).size().rename("y").reset_index()
    ts.columns = ["ds", "y"]

    print("Total time points:", len(ts))

    if len(ts) < horizon * 2:
        print(f"⚠ Not enough data points for evaluation (need ≥ {horizon*2})")
        return

    # Train/test split
    train = ts[:-horizon].copy()
    test = ts[-horizon:].copy()

    # Prophet model
    m = Prophet()
    m.fit(train)

    future = m.make_future_dataframe(periods=horizon, freq=freq)
    fc = m.predict(future)

    pred = (
        fc[["ds", "yhat"]]
        .set_index("ds")
        .reindex(test["ds"])
        .reset_index()
    )

    y_true = test["y"].values
    y_pred = pred["yhat"].values

    # Results
    print("MAPE:", round(mape(y_true, y_pred), 3))
    print("MAE: ", round(mae(y_true, y_pred), 3))
    print("RMSE:", round(rmse(y_true, y_pred), 3))


# ---------- Run All ----------
if __name__ == "__main__":

    # Monthly (ME) — Month End
    evaluate("ME", 12, "Monthly Forecast Accuracy")

    # Quarterly (QE) — Quarter End
    evaluate("QE", 8, "Quarterly Forecast Accuracy")

    # Yearly (YE) — Year End
    evaluate("YE", 3, "Yearly Forecast Accuracy")
