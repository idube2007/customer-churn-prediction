from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from src.generate_data import generate_sample_data
from src.train import train_model

app = Flask(__name__)

# Emergency Fix: Check if model exists, if not, train it!
# This allows the app to fix itself on Render without changing the Build Command.
if not os.path.exists('models/churn_model.pkl'):
    print("Model not found. Starting emergency training...")
    # 1. Generate Data
    if not os.path.exists('data/customer_churn.csv'):
        print("Data not found. Generating data...")
        generate_sample_data()
    
    # 2. Train Model
    print("Training model...")
    train_model()
    print("Training complete!")

# Load the trained model and scaler
try:
    model = joblib.load('models/churn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
except FileNotFoundError:
    print("Error: Model or Scaler not found even after training attempt.")
    model = None
    scaler = None

@app.route('/')
def home():
    """Renders the main page with the input form."""
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles form submission and makes a prediction."""
    if model is None or scaler is None:
        return "Model not loaded. Please train the model first.", 500

    try:
        # 1. Get data from form
        tenure = float(request.form['tenure'])
        monthly_charges = float(request.form['monthly_charges'])
        total_charges = float(request.form['total_charges'])
        support_calls = float(request.form['support_calls'])

        # 2. Prepare data for model
        input_data = pd.DataFrame({
            'Tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'SupportCalls': [support_calls]
        })

        # 3. Preprocess and Predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        # Get probability of churn (class 1)
        probability = model.predict_proba(input_scaled)[0][1]
        
        # 4. Return result to the page
        return render_template(
            'index.html', 
            prediction=int(prediction), 
            probability=round(probability * 100, 2)
        )

    except Exception as e:
        return f"An error occurred: {str(e)}", 400

if __name__ == '__main__':
    # Start the Flask development server
    print("Starting Churn Predictor Web App...")
    app.run(debug=True, port=5000)
