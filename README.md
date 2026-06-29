# Rising Waters Flood Prediction

This project is a Machine Learning approach to predicting flood chances based on weather and rainfall data. It uses an XGBoost classifier, providing predictions via a Flask web application.

## Prerequisites

- Python 3.7+
- pip (Python package installer)

## Installation

1. Clone this repository (or download the source).
2. Create and activate a Python virtual environment to keep dependencies isolated:
   **Windows:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   **macOS/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required Python libraries. You can use the following command to install the dependencies:
   ```bash
   pip install Flask pandas numpy scikit-learn xgboost matplotlib seaborn joblib openpyxl
   ```

## Folder Structure

- `app/` - Contains the Flask web application, templates, and static files.
- `data/` - Contains raw Excel datasets and processed CSV data.
- `docs/` - Contains generated plots from the Exploratory Data Analysis.
- `models/` - Stores the trained ML models and scalers.
- `scripts/` - Contains the ML training pipeline script.

## Running the Project

### 1. Execute the Pipeline (Model Training)

Before running the web application, you must process the dataset, train the machine learning model, and save the model/scaler files.

Run the pipeline script:
```bash
python scripts/run_pipeline.py
```
This script will:
- Read and clean the real-world meteorological dataset from `data/raw/`.
- Perform Exploratory Data Analysis (EDA) and save plots to the `docs/plots/` folder.
- Preprocess the data (handling missing values and outliers).
- Train multiple models and compare them.
- Save the best model (`floods.save`) and the scaler (`transform.save`) into the `models/` directory.

### 2. Start the Flask Web Application

Once the pipeline has completed and saved the model/scaler files, start the web app:
```bash
python app/app.py
```
### 3. Access the Application

Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```
From the home page, enter the required weather parameters (Annual Rainfall, Cloud Visibility, Seasonal Rainfall, Temperature, Humidity) to get a prediction on the chance of a flood.
