import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def train_model():
    # 1. Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('data/customer_churn.csv')
    
    # 2. Basic Preprocessing
    # Separate features (X) and target (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Feature Scaling (Important for Logistic Regression)
    # This ensures all features are on a similar scale (e.g., 0 to 1 or -1 to 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Split data into Training and Testing sets
    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 4. Train the Logistic Regression Model
    print("Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # 5. Evaluate the model
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 6. Save the model and scaler for later use in prediction
    print("Saving model and scaler...")
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump(model, 'models/churn_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Model and Scaler saved at 'models/'.")

if __name__ == "__main__":
    train_model()
