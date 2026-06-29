from flask import Flask, render_template, request, redirect, url_for
# Trigger reload
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, '../models/floods.save')
scaler_path = os.path.join(BASE_DIR, '../models/transform.save')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"Error loading model or scaler. Ensure they are trained and saved first. Error: {e}")
    model = None
    scaler = None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if model is None or scaler is None:
            return render_template('500.html', error="Model not loaded. Please train the model first."), 500
        
        try:
            # Get values from form
            annual_rainfall = float(request.form.get('Annual_Rainfall', 0))
            cloud_visibility = float(request.form.get('Cloud_Visibility', 0))
            seasonal_rainfall = float(request.form.get('Seasonal_Rainfall', 0))
            temperature = float(request.form.get('Temperature', 0))
            humidity = float(request.form.get('Humidity', 0))
            
            # Create input DataFrame (order must match training)
            input_data = pd.DataFrame([[temperature, humidity, cloud_visibility, annual_rainfall, seasonal_rainfall]],
                                      columns=['Temperature', 'Humidity', 'Cloud_Visibility', 'Annual_Rainfall', 'Seasonal_Rainfall'])
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(input_scaled)
            
            # Redirect based on result
            if prediction[0] == 1:
                return redirect(url_for('chance'))
            else:
                return redirect(url_for('no_chance'))
                
        except Exception as e:
            return render_template('500.html', error=f"Invalid or missing data provided. Please check your inputs. Details: {str(e)}"), 400
            
    return render_template('index.html')

@app.route('/chance')
def chance():
    return render_template('chance.html')

@app.route('/no_chance')
def no_chance():
    return render_template('no_chance.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html', error="An unexpected internal server error occurred."), 500

if __name__ == '__main__':
    app.run(debug=True)
