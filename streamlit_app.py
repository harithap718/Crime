# full streamlit_app.py replacement (includes previous features + EDA + hotspots)
import streamlit as st, pandas as pd, os, joblib
# ======= CLUSTER REGION NAME & DESCRIPTION =======

REGION_MAP = {
    0: ("Garfield Park / Kedzie", "High risk zone with theft and battery incidents."),
    1: ("Englewood / Bronzeville", "South side area with higher violent crime rate."),
    2: ("Albany Park (North Side)", "Residential area with mixed crime patterns."),
    3: ("South Deering / Far South", "Industrial belt with lower population density."),
    4: ("Downtown – Loop / River North", "Highest theft density and tourist crime."),
    5: ("South Shore", "High residential crime especially assaults."),
    6: ("Outside Chicago", "Noise cluster, very low crime relevance."),
    7: ("Cicero Border / West Side", "Narcotics and burglary hotspots."),
    8: ("O’Hare East Suburbs", "Vehicle theft + airport-related activity."),
    9: ("Auburn Gresham", "High domestic violence + property crime."),
    10: ("South Chicago", "Frequent assault + theft combination."),
    11: ("Logan Square", "Burglary-dominant zone."),
    12: ("Belmont Cragin / Northwest", "Family area with moderate burglary rate."),
    13: ("Bridgeport / Armour Square", "Stable crime rate, mostly property crimes."),
    14: ("Hyde Park / Woodlawn", "University area, mixed theft patterns."),
    15: ("McKinley Park", "Medium crime zone, old residential blocks."),
    16: ("O’Hare Airport", "Travel-related theft."),
    17: ("Little Village North", "High gang-related activity."),
    18: ("Uptown / Edgewater", "Retail theft + street crimes."),
    19: ("West Englewood", "High assault + battery density."),
    20: ("North Lawndale", "High-risk narcotics region."),
    21: ("Chatham", "Property crime–heavy."),
    22: ("Pilsen / Lower West", "Art district, pickpocket zone."),
    23: ("Hermosa / Cragin", "Mixed residential theft."),
    24: ("Rogers Park", "Low to medium risk area."),
    25: ("Grand Crossing", "High violent-crime density."),
    26: ("East Side", "Lower activity, industrial area."),
    27: ("South Chicago West", "Medium risk, street-level crimes."),
    28: ("Wicker Park / Bucktown", "High nightlife theft."),
    29: ("Humboldt Park West", "High narcotics concentration.")
}

# ======= RISK LEVEL PER CLUSTER =======
RISK_LEVEL = {
    0: "HIGH", 1: "HIGH", 2: "LOW", 3: "LOW", 4: "HIGH", 5: "HIGH",
    6: "LOW", 7: "HIGH", 8: "LOW", 9: "MEDIUM", 10: "MEDIUM", 
    11: "MEDIUM", 12: "LOW", 13: "LOW", 14: "MEDIUM", 15: "LOW",
    16: "LOW", 17: "HIGH", 18: "MEDIUM", 19: "HIGH", 20: "HIGH",
    21: "MEDIUM", 22: "LOW", 23: "LOW", 24: "LOW", 25: "HIGH",
    26: "LOW", 27: "MEDIUM", 28: "HIGH", 29: "HIGH"
}

st.set_page_config(layout="wide", page_title="Crime Project (Paper Match)")

BASE_DIR = r"D:\crime_project2"
MODEL_DIR = os.path.join(BASE_DIR, "models")
PROPHET_DIR = os.path.join(BASE_DIR, "prophet_models")
MAP_DIR = os.path.join(BASE_DIR, "maps")
HOT_DIR = os.path.join(BASE_DIR, "hotspots")

st.title("Machine Learning-Based Crime Prediction and Hotspot Analysis")

tabs = st.tabs(["Predict Crime Type","Predict Location","Forecasting","Hotspots & Map","EDA","Data"])

 # Tab 0: Predict Crime Category (3 groups, LightGBM)
with tabs[0]:
    st.header("Predict Crime Category (3 Groups)")

    Year = st.number_input("Year", value=2025, min_value=1990, max_value=2035, key="pct_year")
    Month = st.number_input("Month", value=11, min_value=1, max_value=12, key="pct_month")
    Day = st.number_input("Day", value=15, min_value=1, max_value=31, key="pct_day")
    Hour = st.number_input("Hour", value=12, min_value=0, max_value=23, key="pct_hour")

    Beat = st.number_input("Beat", value=111, key="pct_beat")
    District = st.number_input("District", value=11, key="pct_district")
    Ward = st.number_input("Ward", value=29, key="pct_ward")
    CommArea = st.number_input("Community Area", value=35, key="pct_commarea")

    if st.button("Predict Crime Category", key="pct_button"):
        try:
            model = joblib.load(os.path.join(MODEL_DIR, "lgbm_3groups_model.joblib"))
            le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
        except Exception as e:
            st.error("LightGBM model missing. Run train_models.py. " + str(e))
        else:

            # ----- Feature engineering (SAME as preprocess.py) -----
            import pandas as pd
            ts = pd.Timestamp(year=Year, month=Month, day=Day)
            day_of_week = ts.dayofweek
            is_weekend = 1 if day_of_week >= 5 else 0

            # season
            if Month in [3,4,5]: season = 1
            elif Month in [6,7,8]: season = 2
            elif Month in [9,10,11]: season = 3
            else: season = 4

            # hour grouping
            if Hour < 6: hour_group = 0
            elif Hour < 12: hour_group = 1
            elif Hour < 18: hour_group = 2
            else: hour_group = 3

            # hotspot (top beats)
            TOP_BEATS = [332, 2523, 1933, 224, 1022, 1113, 414, 2023, 1221, 925]
            hotspot_area = 1 if Beat in TOP_BEATS else 0

            row = pd.DataFrame([{
                "year": Year,
                "month": Month,
                "day": Day,
                "hour": Hour,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "season": season,
                "hour_group": hour_group,
                "beat": Beat,
                "district": District,
                "ward": Ward,
                "community_area": CommArea,
                "hotspot_area": hotspot_area
            }])

            pred = model.predict(row).argmax(axis=1)
            label = le.inverse_transform(pred)[0]

            st.success(f"### 🟦 Predicted Crime Category: **{label}**")


 # Tab 1: Predict Location (Spatial Cluster)
# ============================
# TAB 1 – Predict Location
# ============================

with tabs[1]:
    st.header("📍 Predict Spatial Crime Cluster")

    Latitude = st.number_input("Latitude", value=41.8781, format="%.6f", key="loc_lat")
    Longitude = st.number_input("Longitude", value=-87.6298, format="%.6f", key="loc_lon")
    Beat = st.number_input("Beat", value=111, key="loc_beat")
    District = st.number_input("District", value=11, key="loc_district")
    Ward = st.number_input("Ward", value=29, key="loc_ward")
    CommArea = st.number_input("Community Area", value=35, key="loc_commarea")

    if st.button("🔍 Predict Location Cluster", key="loc_predict_btn"):
        try:
            model = joblib.load(os.path.join(MODEL_DIR, "rf_spatial_location.joblib"))
        except Exception as e:
            st.error("Train the location model first. " + str(e))
        else:
            df = pd.DataFrame([{
                "latitude": Latitude,
                "longitude": Longitude,
                "beat": Beat,
                "district": District,
                "ward": Ward,
                "community_area": CommArea
            }])

            pred_cluster = int(model.predict(df)[0])

            # ---- Retrieve Region Name ----
            region_name, region_desc = REGION_MAP.get(pred_cluster, ("Unknown", "No data"))
            risk = RISK_LEVEL.get(pred_cluster, "LOW")

            # ---- Display Result Card ----
            st.success(f"### 📍 Predicted Cluster: {pred_cluster}")

            st.info(f"""
### 🏙 Region: **{region_name}**
### ⚠ Risk Level: **{risk}**
#### 📝 Description:
{region_desc}
""")

            # ---- Optional: Color-coded risk ----
            if risk == "HIGH":
                st.error("🔥 This is a HIGH-RISK crime zone.")
            elif risk == "MEDIUM":
                st.warning("⚠ Moderate crime activity observed here.")
            else:
                st.success("🟢 This is generally a lower-risk area.")


 # ========================================
# TAB 2 : ULTRA ADVANCED FORECASTING (ALL)
# ========================================

with tabs[2]:
    st.header("🔮 Advanced Forecasting (Prophet + Insights)")

    # ================================
    # ACCURACY CARDS (STATIC FROM YOUR RESULTS)
    # ================================
    st.subheader("📊 Forecasting Accuracy (Based on Evaluation)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("**Yearly Accuracy**\n\n⭐ 80.4%")
    with col2:
        st.info("**Quarterly Accuracy**\n\n⭐ 66.8%")
    with col3:
        st.warning("**Monthly Accuracy**\n\n⭐ 65.0%")

    st.markdown("---")

    # =================================
    # USER FORECAST SELECTION
    # =================================
    st.subheader("⚙️ Forecast Settings")

    resolution = st.selectbox(
        "Select Forecast Resolution",
        ["Monthly", "Quarterly", "Yearly"],
        key="adv2_period"
    )

    PERIOD_MAP = {
        "Monthly": ("M", "ME"),
        "Quarterly": ("Q", "QE"),
        "Yearly": ("A", "YE")
    }

    model_key, freq = PERIOD_MAP[resolution]

    future_steps = st.number_input(
        "Number of Future Steps to Forecast:",
        min_value=1,
        value=12,
        key="steps2"
    )

    st.markdown("---")

    # =====================================================
    # LOAD AND RUN PROPHET MODEL
    # =====================================================
    if st.button("🔮 Generate Unified Forecast", key="fp_all"):
        model_path = os.path.join(PROPHET_DIR, f"prophet_{model_key}.joblib")

        if not os.path.exists(model_path):
            st.error(f"Model for {resolution} not found! Run forecast_prophet.py")
        else:
            model = joblib.load(model_path)
            future = model.make_future_dataframe(periods=future_steps, freq=freq)
            forecast = model.predict(future)

            st.success(f"{resolution} Forecast Generated Successfully!")

            # ================================================
            # INTERACTIVE PLOTLY FORECAST CHART
            # ================================================
            st.subheader("📈 Interactive Forecast Chart")

            import plotly.graph_objects as go

            fig = go.Figure()

            # yhat
            fig.add_trace(go.Scatter(
                x=forecast["ds"], y=forecast["yhat"],
                mode="lines", name="yhat (Forecast)",
                line=dict(width=3)
            ))

            # confidence interval
            fig.add_trace(go.Scatter(
                x=forecast["ds"], y=forecast["yhat_upper"],
                mode="lines", name="Upper Confidence",
                line=dict(width=1, dash="dot")
            ))

            fig.add_trace(go.Scatter(
                x=forecast["ds"], y=forecast["yhat_lower"],
                mode="lines", name="Lower Confidence",
                line=dict(width=1, dash="dot")
            ))

            fig.update_layout(
                title="Forecast with Confidence Interval",
                xaxis_title="Date",
                yaxis_title="Crime Count",
                template="plotly_white",
                hovermode="x"
            )

            st.plotly_chart(fig, use_container_width=True)

            # ============================================
            # FORECAST TABLE
            # ============================================
            st.subheader(f"📄 Forecast Values (Next {future_steps})")
            result_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(future_steps)
            st.write(result_df)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Forecast as CSV",
                data=csv,
                file_name=f"{resolution}_forecast.csv",
                mime="text/csv"
            )

            st.markdown("---")

            # ============================================
            # TREND + SEASONAL COMPONENTS
            # ============================================
            st.subheader("📊 Trend & Seasonal Components")

            with st.expander("Show Components"):
                import matplotlib.pyplot as plt
                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)

            st.markdown("---")

            # ============================================
            # 🔍 AI-STYLE INSIGHTS (cool explanation)
            # ============================================
            st.subheader("🧠 Automated Trend Insights")

            # Analyze trend
            trend_series = forecast["trend"].tail(30)
            direction = "rising 📈" if trend_series.iloc[-1] > trend_series.iloc[0] else "falling 📉"

            st.write(f"**Trend Direction:** Overall crime trend is **{direction}**.")

            # Seasonal info
            if resolution == "Monthly":
                st.write("• Monthly forecast shows **strong seasonality**, often influenced by weather patterns and social events.")
            elif resolution == "Quarterly":
                st.write("• Quarterly forecast reflects **seasonal shifts**, especially in Q2–Q3 where crime rates usually vary sharply.")
            else:
                st.write("• Yearly forecast is **more stable**, showing long-term cycles and policy impacts.")

            st.write("• Confidence intervals widen further into the future, indicating increasing uncertainty — a standard behavior in time-series forecasting models.")


# Tab 3: Hotspots & Map
with tabs[3]:
    st.header("Hotspots & Map")
    if st.button("Compute Hotspots (KMeans) / Generate Map", key="hot_button"):
        try:
            from hotspot_cluster import find_hotspots
            from map_generate import create_heatmap
            report = find_hotspots(n_clusters=12)
            create_heatmap()
            st.success("Hotspots computed and heatmap generated.")
            st.write(report.head(10))
        except Exception as e:
            st.error("Error computing hotspots: " + str(e))
    # show hotspot report if exists
    rep_path = os.path.join(HOT_DIR, "hotspot_report.csv")
    if os.path.exists(rep_path):
        rep = pd.read_csv(rep_path)
        st.subheader("Top Hotspot Clusters")
        st.write(rep.head(10))

# Tab 4: EDA (paper style)
with tabs[4]:
    st.header("Exploratory Data Analysis (Paper-style)")
    if st.button("Load EDA visuals", key="eda_button"):
        path = os.path.join(BASE_DIR, "cleaned_crimes.csv")
        if not os.path.exists(path):
            st.error("cleaned_crimes.csv not found. Run preprocess.py")
        else:
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            st.subheader("Crime by Year")
            fig1 = df.groupby(df["date"].dt.year)["primary_type"].count().sort_index()
            st.bar_chart(fig1)
            st.subheader("Crime by Month (aggregated)")
            fig2 = df.groupby(df["date"].dt.month)["primary_type"].count().reindex(range(1,13), fill_value=0)
            st.bar_chart(fig2)
            st.subheader("Crime by Hour (24h)")
            fig3 = df.groupby(df["date"].dt.hour)["primary_type"].count().reindex(range(0,24), fill_value=0)
            st.bar_chart(fig3)
            st.subheader("Top Crime Types")
            top = df["primary_type"].value_counts().head(20)
            st.bar_chart(top)
            st.subheader("Arrest vs Non-Arrest")
            if "arrest" in df.columns:
                st.write(df["arrest"].value_counts())
            st.subheader("Top Locations (grouped)")
            if "location_group" in df.columns:
                st.write(df["location_group"].value_counts().head(20))

# Tab 5: Data Explorer
with tabs[5]:
    st.header("Data Explorer")
    if st.button("Show sample", key="data_sample"):
        path = os.path.join(BASE_DIR, "cleaned_crimes.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            st.write(df.sample(min(1000, len(df))).reset_index(drop=True))
        else:
            st.error("cleaned_crimes.csv not found. Run preprocess.py")
