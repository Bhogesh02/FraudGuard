# app_mongodb.py (Flask Backend with MongoDB)
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_bcrypt import Bcrypt
import joblib
import os
import pandas as pd
import numpy as np
from datetime import datetime
from mongodb_config import init_mongodb, create_user_collection, create_transaction_collection

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# Initialize MongoDB
mongo = init_mongodb(app)
bcrypt = Bcrypt(app)

# Create collections
users_collection = create_user_collection(mongo)
transactions_collection = create_transaction_collection(mongo)

# --- Load ML Model and Scaler at startup ---
ML_MODEL = None
AMOUNT_SCALER = None
MODEL_FEATURE_COLUMNS = None

try:
    ML_MODEL = joblib.load('fraud_detector_model.pkl')
    AMOUNT_SCALER = joblib.load('amount_scaler.pkl')
    MODEL_FEATURE_COLUMNS = joblib.load('model_feature_columns.pkl')
    print("Machine learning model, scaler, and feature columns loaded successfully.")
except FileNotFoundError:
    print("Error: ML model files not found. Please run fraud_model.py first.")
except Exception as e:
    print(f"An unexpected error occurred loading model assets: {e}")

# --- Constants for simulated V-feature adjustments ---
FRAUD_V_SHIFT_MAGNITUDE_LARGE = 10.0
FRAUD_V_SHIFT_MAGNITUDE_MEDIUM = 5.0
FRAUD_V_SHIFT_MAGNITUDE_SMALL = 1.0
LEGIT_V_INITIAL_MEAN = 0.0
LEGIT_V_INITIAL_STD = 0.1

# --- Helper Function for Feature Engineering ---
def _map_user_input_to_features(user_input: dict) -> pd.DataFrame:
    """Converts user-friendly inputs into numerical features expected by the model."""
    processed_features = {col: np.random.normal(LEGIT_V_INITIAL_MEAN, LEGIT_V_INITIAL_STD)
                          for col in MODEL_FEATURE_COLUMNS if col.startswith('V')}
    processed_features['Amount'] = 0.0

    processed_features['Amount'] = float(user_input.get('amount', 0.0))
    category = user_input.get('merchant_category', 'other').lower()
    location = user_input.get('location', 'same city').lower()
    time_of_day = user_input.get('time_of_day', 'day').lower()

    # Apply fraud patterns
    if (processed_features['Amount'] > 500 or processed_features['Amount'] < 10) or \
       location == 'international' or time_of_day == 'early morning' or \
       category in ['electronics', 'online shopping', 'travel']:

        # Strong negative shifts for fraud indicators
        processed_features['V14'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_LARGE * 2, FRAUD_V_SHIFT_MAGNITUDE_SMALL)
        processed_features['V12'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_LARGE * 1.5, FRAUD_V_SHIFT_MAGNITUDE_SMALL)
        processed_features['V10'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_LARGE * 1.5, FRAUD_V_SHIFT_MAGNITUDE_SMALL)
        processed_features['V17'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_LARGE * 1.5, FRAUD_V_SHIFT_MAGNITUDE_SMALL)
        
        # Strong positive shifts for fraud indicators
        processed_features['V4'] = np.random.normal(FRAUD_V_SHIFT_MAGNITUDE_LARGE, FRAUD_V_SHIFT_MAGNITUDE_SMALL)
        processed_features['V11'] = np.random.normal(FRAUD_V_SHIFT_MAGNITUDE_LARGE, FRAUD_V_SHIFT_MAGNITUDE_SMALL)
        processed_features['V21'] = np.random.normal(FRAUD_V_SHIFT_MAGNITUDE_MEDIUM, FRAUD_V_SHIFT_MAGNITUDE_SMALL)

    input_df = pd.DataFrame([processed_features])
    input_df = input_df[MODEL_FEATURE_COLUMNS]
    return input_df

def generate_fraud_explanation(user_input: dict, fraud_probability: float, is_fraud: bool) -> list:
    """Generates user-friendly explanations for the fraud detection outcome."""
    explanations = []
    prob_percent = (fraud_probability * 100)

    if is_fraud:
        explanations.append(f"This transaction is flagged as **POTENTIALLY FRAUDULENT** with a probability of {prob_percent:.2f}%.")
        explanations.append("Our system detected unusual patterns that deviate significantly from typical legitimate transactions.")

        amount = float(user_input.get('amount', 0))
        category = user_input.get('merchant_category', '').lower()
        location = user_input.get('location', '').lower()
        time_of_day = user_input.get('time_of_day', '').lower()

        specific_reasons = []
        if amount > 500:
            specific_reasons.append(f"The transaction amount (${amount:.2f}) is unusually high.")
        elif amount < 10 and amount > 0:
            specific_reasons.append(f"The transaction amount (${amount:.2f}) is unusually low, which can be a characteristic of card testing.")

        if location == 'international':
            specific_reasons.append("The transaction location is international, which is often associated with higher fraud risk.")
        elif location == 'different city':
            specific_reasons.append("The transaction location is in a different city/state than your usual activity.")

        if time_of_day == 'early morning':
            specific_reasons.append("The transaction occurred in the early morning hours (1 AM - 6 AM), a time often associated with suspicious activity.")
        elif time_of_day == 'night':
            specific_reasons.append("The transaction occurred late at night (10 PM - 1 AM).")

        if category in ['electronics', 'online shopping', 'travel']:
            specific_reasons.append(f"The merchant category '{category.title()}' is sometimes involved in fraudulent activities.")

        if specific_reasons:
            explanations.append("Key indicators identified:")
            explanations.extend([f"- {reason}" for reason in specific_reasons])
        else:
            explanations.append("The decision is based on a complex analysis of various transaction characteristics.")

    else:
        explanations.append(f"This transaction is classified as **LEGITIMATE** with a probability of {prob_percent:.2f}%.")
        explanations.append("The transaction characteristics align with typical safe patterns, and no significant fraud indicators were found by our system.")

    return explanations

# --- Frontend Routes ---
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Find user in MongoDB
        user = users_collection.find_one({"email": email})
        
        if user and bcrypt.check_password_hash(user['password'], password):
            session['logged_in'] = True
            session['user_id'] = str(user['_id'])
            return redirect(url_for('detect_page'))
        else:
            return render_template('login.html', message='Login Unsuccessful. Please check email and password', background_image="fraud_detection_bg.jpg")
    
    return render_template('login.html', background_image="fraud_detection_bg.jpg")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        # Check if user already exists
        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            return render_template('register.html', message='Registration failed: Email already registered.', background_image="fraud_detection_bg.jpg")
        
        # Create new user document
        new_user = {
            "email": email,
            "password": hashed_password,
            "created_at": datetime.utcnow(),
            "last_login": None
        }
        
        try:
            result = users_collection.insert_one(new_user)
            return redirect(url_for('login'))
        except Exception as e:
            return render_template('register.html', message=f'Registration failed: {e}', background_image="fraud_detection_bg.jpg")
    
    return render_template('register.html', background_image="fraud_detection_bg.jpg")

@app.route('/detect', methods=['GET'])
def detect_page():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('detect_user_friendly.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('user_id', None)
    return redirect(url_for('login'))

# --- API Endpoint for Fraud Detection ---
@app.route('/api/detect_fraud', methods=['POST'])
def detect_fraud_api():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized, please log in'}), 401
        
    if ML_MODEL is None or AMOUNT_SCALER is None or MODEL_FEATURE_COLUMNS is None:
        return jsonify({'error': 'Machine learning model assets not loaded. Please contact support.'}), 500

    data = request.json
    
    try:
        # Convert user-friendly input into model-expected numerical features
        processed_input_df = _map_user_input_to_features(data)
        
        # Scale the amount
        processed_input_df['Amount'] = AMOUNT_SCALER.transform(processed_input_df[['Amount']])
        
        # Make prediction
        fraud_probability = ML_MODEL.predict_proba(processed_input_df)[0][1]
        is_fraud = fraud_probability > 0.5
        
        # Generate explanation
        explanations = generate_fraud_explanation(data, fraud_probability, is_fraud)
        
        # Store transaction in MongoDB
        transaction_record = {
            "user_id": session.get('user_id'),
            "timestamp": datetime.utcnow(),
            "transaction_data": data,
            "fraud_probability": fraud_probability,
            "is_fraud": is_fraud,
            "explanations": explanations
        }
        
        transactions_collection.insert_one(transaction_record)
        
        return jsonify({
            'fraud_probability': fraud_probability,
            'is_fraud': is_fraud,
            'explanations': explanations
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# --- API Endpoint to Get User's Transaction History ---
@app.route('/api/transaction_history', methods=['GET'])
def get_transaction_history():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized, please log in'}), 401
    
    try:
        # Get user's transaction history from MongoDB
        transactions = list(transactions_collection.find(
            {"user_id": session.get('user_id')},
            {"_id": 0, "timestamp": 1, "transaction_data": 1, "fraud_probability": 1, "is_fraud": 1}
        ).sort("timestamp", -1).limit(10))
        
        return jsonify({'transactions': transactions})
        
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve transaction history: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 