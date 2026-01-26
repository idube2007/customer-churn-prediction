import pandas as pd
import numpy as np
import os

def generate_sample_data(num_samples=1000):
    """
    Creates a synthetic dataset for customer churn prediction.
    Features:
    - Tenure: Months with company (0 to 72)
    - MonthlyCharges: Range $20 to $120
    - TotalCharges: Approx Tenure * MonthlyCharges
    - SupportCalls: Number of calls to support (0 to 10)
    - Churn: 1 if left, 0 if stayed (dependent on high support calls and high monthly charges)
    """
    np.random.seed(42)
    
    # Generate random features
    tenure = np.random.randint(1, 73, num_samples)
    monthly_charges = np.random.uniform(20.0, 120.0, num_samples).round(2)
    support_calls = np.random.randint(0, 11, num_samples)
    
    # Total charges with some noise
    total_charges = (tenure * monthly_charges + np.random.normal(0, 50, num_samples)).round(2)
    total_charges = np.maximum(total_charges, monthly_charges) # Ensure Total >= Monthly
    
    # Logic for churn (Probability depends on support calls and charges)
    # High support calls and high charges INCREASE churn probability
    churn_prob = (0.1 * support_calls + 0.005 * monthly_charges - 0.01 * tenure)
    # Normalize prob between 0 and 1
    churn_prob = 1 / (1 + np.exp(-churn_prob)) 
    
    churn = (np.random.rand(num_samples) < churn_prob).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'SupportCalls': support_calls,
        'Churn': churn
    })
    
    # Ensure data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
        
    # Save to CSV
    df.to_csv('data/customer_churn.csv', index=False)
    print(f"Dataset created successfully with {num_samples} samples at 'data/customer_churn.csv'")

if __name__ == "__main__":
    generate_sample_data()
