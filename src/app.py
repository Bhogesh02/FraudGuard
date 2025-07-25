# app.py (Flask Backend)
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_bcrypt import Bcrypt
import joblib
import os
import pandas as pd
import numpy as np # Import numpy for numerical operations

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

bcrypt = Bcrypt(app)

# --- Load ML Model and Scaler at startup ---
ML_MODEL = None
AMOUNT_SCALER = None
MODEL_FEATURE_COLUMNS = None # To store the expected feature order

try:
    ML_MODEL = joblib.load('fraud_detector_model.pkl')
    AMOUNT_SCALER = joblib.load('amount_scaler.pkl')
    MODEL_FEATURE_COLUMNS = joblib.load('model_feature_columns.pkl')
    print("Machine learning model, scaler, and feature columns loaded successfully.")
    print(f"Expected model features: {MODEL_FEATURE_COLUMNS}")
except FileNotFoundError:
    print("Error: ML model, scaler, or feature columns not found. Please run fraud_model.py first to train and save them.")
    # In a production app, you might want to exit or show a clear error page
except Exception as e:
    print(f"An unexpected error occurred loading model assets: {e}")

# --- Constants for simulated V-feature adjustments ---
# These values are designed to push the V-features towards the extremes
# often seen in fraudulent transactions in the Kaggle dataset.
# They are illustrative and *not* exact transformations.
FRAUD_V_SHIFT_MAGNITUDE_LARGE = 10.0 # For very strong indicators
FRAUD_V_SHIFT_MAGNITUDE_MEDIUM = 5.0 # For moderately strong indicators
FRAUD_V_SHIFT_MAGNITUDE_SMALL = 1.0  # For weaker indicators

# Typical initial values for legitimate-like V-features (near zero after PCA scaling)
LEGIT_V_INITIAL_MEAN = 0.0
LEGIT_V_INITIAL_STD = 0.1 # Small random noise around mean

# --- Helper Function for Feature Engineering (Mapping User Input) ---
def _map_user_input_to_features(user_input: dict) -> pd.DataFrame:
    """
    Converts user-friendly inputs into the numerical features expected by the model.
    This is a SIMULATED mapping for demonstration purposes with V-features.
    The V-feature adjustments are based on common observations from Kaggle notebooks
    about which V-features tend to be highly positive or negative for fraudulent transactions.
    """
    # Initialize all V-features with a small random noise around zero (typical for legitimate transactions)
    processed_features = {col: np.random.normal(LEGIT_V_INITIAL_MEAN, LEGIT_V_INITIAL_STD)
                          for col in MODEL_FEATURE_COLUMNS if col.startswith('V')}
    processed_features['Amount'] = 0.0 # Initialize amount separately

    # 1. Handle Amount
    processed_features['Amount'] = float(user_input.get('amount', 0.0))

    # Get user-friendly categorical inputs
    category = user_input.get('merchant_category', 'other').lower()
    location = user_input.get('location', 'same city').lower()
    time_of_day = user_input.get('time_of_day', 'day').lower()

    # --- Apply simulated effects of user inputs on V-features to push towards "fraud" patterns ---
    # These adjustments are heuristic and aim to push the V-values towards
    # regions typically associated with fraud based on common patterns.
    # The magnitudes are chosen to be more aggressive than previous versions.

    # Strong fraud indicators:
    # V14, V10, V12, V17 are very often strongly negative for fraud
    # V4, V11, V21 are often strongly positive for fraud
    # V3, V7, V9, V16, V18, V1 make smaller negative contributions

    # Scenario 1: High amount AND unusual location/time/category
    if (processed_features['Amount'] > 500 or processed_features['Amount'] < 10) or \
       location == 'international' or time_of_day == 'early morning' or \
       category in ['electronics', 'online shopping', 'travel']:

        # Apply strong negative shifts to V-features commonly negative in fraud
        processed_features['V14'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_LARGE * 2, FRAUD_V_SHIFT_MAGNITUDE_SMALL) # Very strong negative
        processed_features['V12'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_LARGE * 1.5, FRAUD_V_SHIFT_MAGNITUDE_SMALL)
        processed_features['V10'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_LARGE * 1.5, FRAUD_V_SHIFT_MAGNITUDE_SMALL)
        processed_features['V17'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_LARGE * 1.5, FRAUD_V_SHIFT_MAGNITUDE_SMALL)
        processed_features['V3'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_MEDIUM, FRAUD_V_SHIFT_MAGNITUDE_SMALL)
        processed_features['V7'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_MEDIUM, FRAUD_V_SHIFT_MAGNITUDE_SMALL)
        processed_features['V16'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_MEDIUM, FRAUD_V_SHIFT_MAGNITUDE_SMALL)
        processed_features['V18'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_MEDIUM, FRAUD_V_SHIFT_MAGNITUDE_SMALL)
        processed_features['V1'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_SMALL, LEGIT_V_INITIAL_STD)
        processed_features['V9'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_MEDIUM, FRAUD_V_SHIFT_MAGNITUDE_SMALL)


        # Apply strong positive shifts to V-features commonly positive in fraud
        processed_features['V4'] = np.random.normal(FRAUD_V_SHIFT_MAGNITUDE_LARGE, FRAUD_V_SHIFT_MAGNITUDE_SMALL)
        processed_features['V11'] = np.random.normal(FRAUD_V_SHIFT_MAGNITUDE_LARGE, FRAUD_V_SHIFT_MAGNITUDE_SMALL)
        processed_features['V21'] = np.random.normal(FRAUD_V_SHIFT_MAGNITUDE_MEDIUM, FRAUD_V_SHIFT_MAGNITUDE_SMALL)

        # Other V-features that show some pattern, less dramatic
        processed_features['V2'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_SMALL, LEGIT_V_INITIAL_STD)
        processed_features['V5'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_SMALL, LEGIT_V_INITIAL_STD)
        processed_features['V6'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_SMALL, LEGIT_V_INITIAL_STD)
        processed_features['V8'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_SMALL, LEGIT_V_INITIAL_STD)
        processed_features['V13'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_SMALL, LEGIT_V_INITIAL_STD)
        processed_features['V15'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_SMALL, LEGIT_V_INITIAL_STD)
        processed_features['V19'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_SMALL, LEGIT_V_INITIAL_STD)
        processed_features['V20'] = np.random.normal(FRAUD_V_SHIFT_MAGNITUDE_SMALL, LEGIT_V_INITIAL_STD)
        processed_features['V22'] = np.random.normal(FRAUD_V_SHIFT_MAGNITUDE_SMALL, LEGIT_V_INITIAL_STD)
        processed_features['V23'] = np.random.normal(FRAUD_V_SHIFT_MAGNITUDE_SMALL, LEGIT_V_INITIAL_STD)
        processed_features['V24'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_SMALL, LEGIT_V_INITIAL_STD)
        processed_features['V25'] = np.random.normal(FRAUD_V_SHIFT_MAGNITUDE_SMALL, LEGIT_V_INITIAL_STD)
        processed_features['V26'] = np.random.normal(-FRAUD_V_SHIFT_MAGNITUDE_SMALL, LEGIT_V_INITIAL_STD)
        processed_features['V27'] = np.random.normal(FRAUD_V_SHIFT_MAGNITUDE_SMALL, LEGIT_V_INITIAL_STD)
        processed_features['V28'] = np.random.normal(FRAUD_V_SHIFT_MAGNITUDE_SMALL, LEGIT_V_INITIAL_STD)

    # Convert to DataFrame, ensuring column order matches the training data
    input_df = pd.DataFrame([processed_features])
    input_df = input_df[MODEL_FEATURE_COLUMNS] # This is critical for model consistency

    return input_df

def generate_fraud_explanation(user_input: dict, fraud_probability: float, is_fraud: bool) -> list:
    """
    Generates user-friendly explanations for the fraud detection outcome.
    """
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
        if amount > 500: # High value transaction
            specific_reasons.append(f"The transaction amount (${amount:.2f}) is unusually high.")
        elif amount < 10 and amount > 0: # Very low value, sometimes used for card testing
            specific_reasons.append(f"The transaction amount (${amount:.2f}) is unusually low, which can be a characteristic of card testing or suspicious small purchases.")

        if location == 'international':
            specific_reasons.append("The transaction location is international, which is often associated with higher fraud risk.")
        elif location == 'different city':
            specific_reasons.append("The transaction location is in a different city/state than your usual activity.")

        if time_of_day == 'early morning':
            specific_reasons.append("The transaction occurred in the early morning hours (1 AM - 6 AM), a time often associated with suspicious activity.")
        elif time_of_day == 'night':
            specific_reasons.append("The transaction occurred late at night (10 PM - 1 AM).")

        if category in ['electronics', 'online shopping', 'travel']: # High-risk categories
            specific_reasons.append(f"The merchant category '{category.title()}' is sometimes involved in fraudulent activities.")

        if specific_reasons:
            explanations.append("Key indicators identified:")
            explanations.extend([f"- {reason}" for reason in specific_reasons])
        else:
            explanations.append("The decision is based on a complex analysis of various transaction characteristics, even if no obvious specific reason stands out.")

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
        # TODO: Implement MongoDB authentication here
        # For now, we'll use a simple demo login
        if email == "demo@example.com" and password == "password":
            session['logged_in'] = True
            session['user_id'] = 1
            return redirect(url_for('detect_page'))
        else:
            return render_template('login.html', message='Login Unsuccessful. Please check email and password', background_image="fraud_detection_bg.jpg")
    return render_template('login.html', background_image="fraud_detection_bg.jpg")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '')
        email = request.form.get('email', '')
        country = request.form.get('country', '')
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Basic validation
        if not all([name, email, country, password, confirm_password]):
            return render_template('register.html', 
                                message='All fields are required.', 
                                background_image="fraud_detection_bg.jpg")
        
        if password != confirm_password:
            return render_template('register.html', 
                                message='Passwords do not match.', 
                                background_image="fraud_detection_bg.jpg")
        
        if len(password) < 6:
            return render_template('register.html', 
                                message='Password must be at least 6 characters long.', 
                                background_image="fraud_detection_bg.jpg")
        
        # TODO: Implement MongoDB user registration here
        # For now, we'll just redirect to login
        return redirect(url_for('login'))
    return render_template('register.html', background_image="fraud_detection_bg.jpg")

@app.route('/about')
def about():
    return render_template('about.html', background_image="fraud_detection_bg.jpg")

@app.route('/team')
def team():
    return render_template('team.html', background_image="fraud_detection_bg.jpg")

@app.route('/user-details')
def user_details():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('user_details.html', background_image="fraud_detection_bg.jpg", username="Demo User")

@app.route('/detect', methods=['GET'])
def detect_page():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('detect_user_friendly.html', username="Demo User")

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('user_id', None)
    return redirect(url_for('login'))


# --- API Endpoint for Fraud Detection ---
# @app.route('/api/detect_fraud', methods=['POST'])
# def detect_fraud_api():
#     if not session.get('logged_in'):
#         return jsonify({'error': 'Unauthorized, please log in'}), 401
        
#     if ML_MODEL is None or AMOUNT_SCALER is None or MODEL_FEATURE_COLUMNS is None:
#         return jsonify({'error': 'Machine learning model assets not loaded. Please contact support.'}), 500

#     data = request.json # This will contain user_amount, merchant_category, location, time_of_day
    
#     try:
#         # Convert user-friendly input into model-expected numerical features
#         processed_input_df = _map_user_input_to_features(data)
        
#         # Scale the 'Amount' feature using the loaded scaler
#         if 'Amount' in processed_input_df.columns:
#             processed_input_df['Amount'] = AMOUNT_SCALER.transform(processed_input_df[['Amount']])
#         else:
#             return jsonify({'error': 'Amount feature missing after processing user input.'}), 400

#         # Predict probability of fraud (class 1)
#         fraud_probability = ML_MODEL.predict_proba(processed_input_df)[:, 1][0]

#         # --- ADJUSTABLE FRAUD THRESHOLD ---
#         # IMPORTANT: Set this based on the 'Optimal Threshold' you found from running fraud_model.py.
#         # If your optimal threshold was, for example, 0.1234, you can set it here.
#         # This determines the sensitivity of your fraud detection. Lower means more fraud detected,
#         # but also potentially more false positives.
#         FRAUD_THRESHOLD =  0.8100 # <--- UPDATE THIS VALUE after running fraud_model.py!

#         is_fraud = bool(fraud_probability >= FRAUD_THRESHOLD)
        
#         # Generate user-friendly explanations
#         explanations = generate_fraud_explanation(data, fraud_probability, is_fraud)

#         return jsonify({
#             'is_fraud': is_fraud,
#             'fraud_probability': float(fraud_probability),
#             'message': 'Fraud detected! Please review this transaction.' if is_fraud else 'Transaction appears legitimate.',
#             'explanations': explanations
#         })
#     except ValueError as ve:
#         return jsonify({'error': f'Invalid input data: {ve}'}), 400
#     except KeyError as ke:
#         return jsonify({'error': f'Missing expected data key: {ke}'}), 400
#     except Exception as e:
#         print(f"Error during fraud detection: {e}") # Log the full error
#         return jsonify({'error': f'An unexpected error occurred during fraud detection. Details: {e}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)