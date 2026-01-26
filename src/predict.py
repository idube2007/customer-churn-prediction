import pandas as pd
import joblib

def make_prediction():
    # 1. Load the saved model and scaler
    try:
        model = joblib.load('models/churn_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
    except FileNotFoundError:
        print("Error: Model or Scaler not found. Please run src/train.py first.")
        return

    # 2. Define a new customer's data
    # Features: Tenure, MonthlyCharges, TotalCharges, SupportCalls
    new_customer = pd.DataFrame({
        'Tenure': [12],
        'MonthlyCharges': [80.0],
        'TotalCharges': [960.0],
        'SupportCalls': [5]
    })
    
    print("Predicting churn for customer:")
    print(new_customer)
    
    # 3. Preprocess the input (Scale it)
    new_customer_scaled = scaler.transform(new_customer)
    
    # 4. Make a prediction
    prediction = model.predict(new_customer_scaled)
    # Get the probability as well
    probability = model.predict_proba(new_customer_scaled)
    
    # 5. Show the result
    if prediction[0] == 1:
        print(f"\nResult: Likely to Churn (Churn Probability: {probability[0][1]*100:.2f}%)")
    else:
        print(f"\nResult: Likely to Stay (Churn Probability: {probability[0][1]*100:.2f}%)")

if __name__ == "__main__":
    make_prediction()
