
# Crime Prediction Project (Updated for D:\crime_project)

## Overview
This project is configured to save all data, models and outputs to `D:\crime_project`.
Files included:
- data_download.py : download Chicago dataset to D:\crime_project\chicago_crimes.csv
- preprocess.py : clean and save to D:\crime_project\cleaned_crimes.csv
- train_models.py : trains RandomForest, saves to D:\crime_project\models\
- forecast_prophet.py : trains Prophet time-series model, saves to D:\crime_project\prophet_models\
- predict.py : simple prediction helper
- streamlit_app.py : demo UI for local predictions
- requirements.txt : python dependencies

## Quick start (Windows)
1. Open PowerShell, create folders on D: if not present:
   ```powershell
   mkdir D:\crime_project
   cd path\to\this\unzipped\project
   python -m venv venv
   .\\venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
2. Download dataset (or place your CSV):
   ```powershell
   python data_download.py --limit 200000
   ```
   Or place `Crimes_-_2001_to_Present.csv` renamed to `chicago_crimes.csv` in `D:\crime_project\`

3. Preprocess:
   ```powershell
   python preprocess.py
   ```

4. Train models:
   ```powershell
   python train_models.py
   ```

5. (Optional) Train Prophet forecast:
   ```powershell
   python forecast_prophet.py
   ```

6. Run Streamlit UI:
   ```powershell
   streamlit run streamlit_app.py
   ```

## Notes
- The code uses simple feature engineering and is meant as a reproducible starting point.
- Adjust RandomForest params in `train_models.py` to reduce size/time.
- If you prefer a different D: path, modify the `BASE_DIR` variable at the top of each script.
