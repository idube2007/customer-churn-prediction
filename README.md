# Customer Churn Prediction Project

## Objective
The goal of this project is to build a machine learning model that predicts whether a customer is likely to leave a service (churn) based on their usage patterns and charges. This is a classic classification problem in Data Science.

## Dataset
For this project, we use a synthetic dataset (`data/customer_churn.csv`) created to simulate real-world customer behavior.
- **Tenure**: How long the customer has been with the company (in months).
- **MonthlyCharges**: The amount charged to the customer each month.
- **TotalCharges**: Cumulative charges since the customer joined.
- **SupportCalls**: Number of interactions with customer support.
- **Churn (Target)**: 1 if the customer churned, 0 otherwise.

## Methodology: Logistic Regression
We used **Logistic Regression**, which is a fundamental statistical method used for binary classification. 
1. **Data Preprocessing**: We handled numerical features by scaling them (using `StandardScaler`).
2. **Model Training**: The data was split into training (80%) and testing (20%) sets.
3. **Evaluation**: We measured success using Accuracy, Precision, and Recall.

## Web Application (Flask)
The project includes a simple web interface built with **Flask**. This allows users to:
- Input customer details via a user-friendly form.
- Get instant churn predictions.
- View the probability (%) of the customer leaving.

## Tools Used
- **Python**: Core programming language.
- **Pandas**: For data manipulation.
- **Scikit-learn**: For machine learning.
- **Flask**: For the web interface.
- **Inter Font**: For clean typography.

## Project Structure
```text
customer-churn-predictor/
├── data/
│   └── customer_churn.csv     # Generated dataset
├── models/
│   ├── churn_model.pkl        # Trained model
│   └── scaler.pkl             # Scaler for preprocessing
├── src/
│   ├── generate_data.py       # Script to create data
│   ├── train.py                # Script to train model
│   └── predict.py              # Command-line prediction script
├── templates/
│   └── index.html             # Web UI template
├── app.py                     # Flask Web Application
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## How to Run
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Setup Data and Model**:
   ```bash
   python src/generate_data.py
   python src/train.py
   ```
3. **Run the Web App**:
   ```bash
   python app.py
   ```
   Open your browser and go to `http://127.0.0.1:5000`

4. **(Optional) Run CLI Prediction**:
   ```bash
   python src/predict.py
   ```

## Deployment
### Deploying to Render.com (Free)
1. Sign up on [Render.com](https://render.com).
2. Create a **New Web Service** and connect your GitHub repository.
3. Use the following settings:
   - **Runtime**: `Python`
   - **Build Command**: `pip install -r requirements.txt && python src/generate_data.py && python src/train.py`
   - **Start Command**: `gunicorn app:app`
4. Click **Deploy Web Service**.

## Conclusion
This project demonstrates a complete end-to-end machine learning pipeline, from data generation to a working web application. It highlights the practical application of Logistic Regression in a business context.
