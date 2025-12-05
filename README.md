# Crime Prediction Project

## Overview
This project is configured to save all data, models and outputs to `D:\crime_project`.
Files included:
- data_download.py : download Chicago dataset to D:\crime_project\chicago_crimes.csv
- preprocess.py : clean and save to D:\crime_project\cleaned_crimes.csv
- train_models.py : trains RandomForest, saves models to D:\crime_project\models\
- forecast_prophet.py : trains Prophet time-series model, saves to D:\crime_project\prophet_models\
- predict.py : simple prediction helper
- streamlit_app.py : demo UI for local predictions
- requirements.txt : python dependencies

## Quick Start (Windows)

### 1. Setup Environment
```powershell
mkdir D:\crime_project
cd path\to\this\unzipped\project
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

