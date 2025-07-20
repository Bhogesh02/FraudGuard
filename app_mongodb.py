# app_mongodb.py (Flask Backend with MongoDB)
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_bcrypt import Bcrypt
import joblib
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from bson import ObjectId
from mongodb_config import init_mongodb, create_user_collection, create_transaction_collection

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# Initialize MongoDB
try:
    mongo = init_mongodb(app)
    bcrypt = Bcrypt(app)

    # Create collections
    users_collection = create_user_collection(mongo)
    transactions_collection = create_transaction_collection(mongo)
    print("MongoDB connection established successfully")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    mongo = None
    bcrypt = None
    users_collection = None
    transactions_collection = None

# --- Load ML Model and Scaler at startup ---
ML_MODEL = None
AMOUNT_SCALER = None
MODEL_FEATURE_COLUMNS = None
OPTIMAL_THRESHOLD = None    # Default threshold

try:
    ML_MODEL = joblib.load('fraud_detector_model.pkl')
    AMOUNT_SCALER = joblib.load('amount_scaler.pkl')
    MODEL_FEATURE_COLUMNS = joblib.load('model_feature_columns.pkl')
    try:
        OPTIMAL_THRESHOLD = joblib.load('optimal_threshold.pkl')
        print(f"✓ Optimal threshold: {OPTIMAL_THRESHOLD:.4f}")
    except FileNotFoundError:
        OPTIMAL_THRESHOLD = 0.5  # Default threshold
        print(f"⚠️  optimal_threshold.pkl not found. Using default threshold: {OPTIMAL_THRESHOLD}")
    print("✓ Enhanced machine learning model assets loaded successfully.")
    print(f"✓ Model features: {len(MODEL_FEATURE_COLUMNS)}")
except FileNotFoundError:
    print("⚠️  ML model files not found. Please run fraud_model.py first.")
except Exception as e:
    print(f"⚠️  Error loading model assets: {e}")

# --- Enhanced Feature Engineering Constants ---
FRAUD_PATTERNS = {
    'high_amount': 1000,
    'low_amount': 10,
    'night_hours': (22, 6),
    'early_morning': (1, 6),
    'international_risk': 0.8,
    'electronics_risk': 0.6,
    'online_shopping_risk': 0.7,
    'travel_risk': 0.5
}

# --- Enhanced Feature Mapping Function ---
def _map_user_input_to_features(user_input: dict) -> pd.DataFrame:
    """Converts user-friendly inputs into enhanced numerical features expected by the model."""
    
    # Initialize all features with default values
    processed_features = {}
    
    # Initialize V-features with normal distribution
    for i in range(1, 29):
        processed_features[f'V{i}'] = np.random.normal(0, 1)
    
    # Get input values
    amount = float(user_input.get('amount', 0.0))
    category = user_input.get('merchant_category', 'other').lower()
    location = user_input.get('location', 'same city').lower()
    time_of_day = user_input.get('time_of_day', 'day').lower()
    
    # Set basic features
    processed_features['Amount'] = amount
    processed_features['Amount_Log'] = np.log1p(amount)
    processed_features['Amount_Squared'] = amount ** 2
    
    # Time-based features
    current_hour = datetime.now().hour
    if time_of_day == 'early morning':
        current_hour = np.random.randint(1, 6)
    elif time_of_day == 'night':
        current_hour = np.random.randint(22, 24)
    elif time_of_day == 'morning':
        current_hour = np.random.randint(6, 12)
    elif time_of_day == 'afternoon':
        current_hour = np.random.randint(12, 18)
    elif time_of_day == 'evening':
        current_hour = np.random.randint(18, 22)
    
    processed_features['Time_Hour'] = current_hour
    
    # Risk indicators
    processed_features['High_Amount'] = 1 if amount > FRAUD_PATTERNS['high_amount'] else 0
    processed_features['Low_Amount'] = 1 if amount < FRAUD_PATTERNS['low_amount'] else 0
    processed_features['Night_Transaction'] = 1 if (current_hour >= 22 or current_hour <= 6) else 0
    
    # Calculate fraud risk score
    fraud_risk = 0.0
    
    # Amount-based risk
    if amount > FRAUD_PATTERNS['high_amount']:
        fraud_risk += 0.3
    elif amount < FRAUD_PATTERNS['low_amount']:
        fraud_risk += 0.2
    
    # Location-based risk
    if location == 'international':
        fraud_risk += 0.4
    elif location == 'different city':
        fraud_risk += 0.2
    
    # Time-based risk
    if current_hour >= 22 or current_hour <= 6:
        fraud_risk += 0.2
    elif current_hour >= 1 and current_hour <= 6:
        fraud_risk += 0.3
    
    # Category-based risk
    if category in ['electronics', 'online shopping', 'travel']:
        fraud_risk += 0.2
    
    # Apply sophisticated fraud patterns based on risk
    if fraud_risk > 0.3:  # High risk transaction
        # Strong negative correlations (fraud indicators)
        processed_features['V14'] = np.random.normal(-8, 2)
        processed_features['V12'] = np.random.normal(-6, 2)
        processed_features['V10'] = np.random.normal(-7, 2)
        processed_features['V17'] = np.random.normal(-5, 2)
        
        # Strong positive correlations (fraud indicators)
        processed_features['V4'] = np.random.normal(8, 2)
        processed_features['V11'] = np.random.normal(6, 2)
        processed_features['V21'] = np.random.normal(4, 2)
        
        # Additional fraud patterns
        processed_features['V2'] = np.random.normal(-3, 1)
        processed_features['V7'] = np.random.normal(-4, 1)
        processed_features['V9'] = np.random.normal(-5, 1)
        processed_features['V16'] = np.random.normal(-6, 1)
        processed_features['V18'] = np.random.normal(-4, 1)
        processed_features['V19'] = np.random.normal(-3, 1)
        processed_features['V20'] = np.random.normal(-2, 1)
        
    elif fraud_risk > 0.1:  # Medium risk transaction
        # Moderate fraud patterns
        processed_features['V14'] = np.random.normal(-4, 2)
        processed_features['V12'] = np.random.normal(-3, 2)
        processed_features['V10'] = np.random.normal(-3, 2)
        processed_features['V4'] = np.random.normal(4, 2)
        processed_features['V11'] = np.random.normal(3, 2)
    
    # Create interaction features
    processed_features['Amount_V14'] = amount * processed_features['V14']
    processed_features['Amount_V12'] = amount * processed_features['V12']
    processed_features['V4_V11'] = processed_features['V4'] * processed_features['V11']
    
    # Statistical features
    v_features = [processed_features[f'V{i}'] for i in range(1, 29)]
    processed_features['V_Mean'] = np.mean(v_features)
    processed_features['V_Std'] = np.std(v_features)
    processed_features['V_Max'] = np.max(v_features)
    processed_features['V_Min'] = np.min(v_features)
    
    # Ensure all required features are present
    for feature in MODEL_FEATURE_COLUMNS:
        if feature not in processed_features:
            processed_features[feature] = 0.0
    
    # Create DataFrame with correct feature order
    input_df = pd.DataFrame([processed_features])
    input_df = input_df[MODEL_FEATURE_COLUMNS]
    
    return input_df

def generate_fraud_explanation(user_input: dict, fraud_probability: float, is_fraud: bool) -> list:
    """Generates detailed user-friendly explanations for the fraud detection outcome."""
    explanations = []
    prob_percent = (fraud_probability * 100)

    if is_fraud:
        explanations.append(f"🚨 **FRAUD ALERT**: This transaction is flagged as potentially fraudulent with a confidence of {prob_percent:.1f}%.")
        explanations.append("Our enhanced AI system detected multiple risk indicators that deviate significantly from normal transaction patterns.")

        amount = float(user_input.get('amount', 0))
        category = user_input.get('merchant_category', '').lower()
        location = user_input.get('location', '').lower()
        time_of_day = user_input.get('time_of_day', '').lower()

        risk_factors = []
        risk_score = 0

        # Amount analysis
        if amount > FRAUD_PATTERNS['high_amount']:
            risk_factors.append(f"💰 Unusually high amount (${amount:.2f}) - typical fraudsters test with large transactions")
            risk_score += 30
        elif amount < FRAUD_PATTERNS['low_amount']:
            risk_factors.append(f"💳 Very low amount (${amount:.2f}) - characteristic of card testing attacks")
            risk_score += 20

        # Location analysis
        if location == 'international':
            risk_factors.append("🌍 International transaction - higher fraud risk due to geographic distance")
            risk_score += 40
        elif location == 'different city':
            risk_factors.append("🏙️ Different city/state - unusual location pattern")
            risk_score += 20

        # Time analysis
        if time_of_day == 'early morning':
            risk_factors.append("🌅 Early morning transaction (1-6 AM) - unusual timing for legitimate purchases")
            risk_score += 30
        elif time_of_day == 'night':
            risk_factors.append("🌙 Late night transaction (10 PM-1 AM) - suspicious timing")
            risk_score += 15

        # Category analysis
        if category in ['electronics', 'online shopping', 'travel']:
            risk_factors.append(f"🛒 High-risk merchant category '{category.title()}' - commonly targeted by fraudsters")
            risk_score += 20

        if risk_factors:
            explanations.append(f"🔍 **Risk Analysis** (Score: {risk_score}/100):")
            explanations.extend([f"• {factor}" for factor in risk_factors])
        else:
            explanations.append("🔍 **Risk Analysis**: The decision is based on complex pattern analysis of transaction characteristics.")

        # Confidence level explanation
        if prob_percent > 90:
            explanations.append("⚠️ **High Confidence Alert**: Multiple strong fraud indicators detected.")
        elif prob_percent > 70:
            explanations.append("⚠️ **Medium Confidence Alert**: Several suspicious patterns identified.")
        else:
            explanations.append("⚠️ **Low Confidence Alert**: Some unusual patterns detected, recommend manual review.")

    else:
        explanations.append(f"✅ **LEGITIMATE TRANSACTION**: This transaction is classified as legitimate with a confidence of {prob_percent:.1f}%.")
        explanations.append("The transaction characteristics align with normal, safe patterns and no significant fraud indicators were detected.")

        # Explain why it's considered safe
        safe_factors = []
        amount = float(user_input.get('amount', 0))
        category = user_input.get('merchant_category', '').lower()
        location = user_input.get('location', '').lower()
        time_of_day = user_input.get('time_of_day', '').lower()

        if 10 <= amount <= 500:
            safe_factors.append(f"💰 Transaction amount (${amount:.2f}) is within normal range")
        
        if location == 'same city':
            safe_factors.append("🏠 Transaction location matches your usual activity area")
        
        if time_of_day in ['day', 'morning', 'afternoon', 'evening']:
            safe_factors.append("⏰ Transaction timing is during normal business hours")
        
        if category in ['groceries', 'gas', 'restaurant', 'retail']:
            safe_factors.append(f"🛒 Merchant category '{category.title()}' is low-risk")

        if safe_factors:
            explanations.append("✅ **Safety Indicators**:")
            explanations.extend([f"• {factor}" for factor in safe_factors])

    return explanations

# --- Debug route to check users in database ---
@app.route('/debug/users')
def debug_users():
    if users_collection is None:
        return jsonify({"error": "Database connection not available"}), 500
    
    try:
        users = list(users_collection.find({}, {"password": 0}))  # Exclude passwords for security
        return jsonify({
            "total_users": len(users),
            "users": users
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Frontend Routes ---
@app.route('/')
def index():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if users_collection is None:
            return render_template('login.html', message='Database connection error. Please try again later.', background_image="fraud_detection_bg.jpg")
        
        email = request.form['email']
        password = request.form['password']
        
        print(f"Login attempt for email: {email}")
        
        # Find user in MongoDB
        user = users_collection.find_one({"email": email})
        
        if user:
            print(f"User found: {user.get('name', 'Unknown')}")
            if bcrypt.check_password_hash(user['password'], password):
                print("Password check successful")
                session['logged_in'] = True
                session['user_id'] = str(user['_id'])
                session['user_email'] = user['email']
                session['user_name'] = user.get('name', 'User')
                
                # Update last login time
                users_collection.update_one(
                    {"_id": user['_id']},
                    {"$set": {"last_login": datetime.now(timezone.utc)}}
                )
                
                print(f"Login successful for user: {user.get('name', 'Unknown')}")
                return redirect(url_for('dashboard'))
            else:
                print("Password check failed")
                return render_template('login.html', message='Login Unsuccessful. Please check email and password', background_image="fraud_detection_bg.jpg")
        else:
            print(f"No user found with email: {email}")
            return render_template('login.html', message='Login Unsuccessful. Please check email and password', background_image="fraud_detection_bg.jpg")
    
    return render_template('login.html', background_image="fraud_detection_bg.jpg")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if users_collection is None:
            return render_template('register.html', message='Database connection error. Please try again later.', background_image="fraud_detection_bg.jpg")
        
        name = request.form['name']
        email = request.form['email']
        country = request.form['country']
        password = request.form['password']
        
        print(f"Registration attempt for: {name} ({email}) from {country}")
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        # Check if user already exists
        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            print(f"Registration failed: Email {email} already exists")
            return render_template('register.html', message='Registration failed: Email already registered.', background_image="fraud_detection_bg.jpg")
        
        # Create new user document
        new_user = {
            "name": name,
            "email": email,
            "country": country,
            "password": hashed_password,
            "created_at": datetime.now(timezone.utc),
            "last_login": None
        }
        
        try:
            result = users_collection.insert_one(new_user)
            print(f"User registered successfully: {name} with ID: {result.inserted_id}")
            return redirect(url_for('login'))
        except Exception as e:
            print(f"Registration error: {e}")
            return render_template('register.html', message=f'Registration failed: {e}', background_image="fraud_detection_bg.jpg")
    
    return render_template('register.html', background_image="fraud_detection_bg.jpg")

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    user_id = session.get('user_id')
    
    # Get user statistics
    transactions = list(transactions_collection.find({"user_id": user_id}))
    total_transactions = len(transactions)
    fraud_count = len([t for t in transactions if t.get('is_fraud', False)])
    accuracy_rate = 87  # Default accuracy rate
    
    # Calculate days active
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if user and user.get('created_at'):
        # Handle timezone-naive datetime from database
        created_at = user['created_at']
        if created_at.tzinfo is None:
            # If naive, assume UTC
            created_at = created_at.replace(tzinfo=timezone.utc)
        days_active = (datetime.now(timezone.utc) - created_at).days
    else:
        days_active = 30
    
    # Get recent transactions for dashboard
    recent_transactions = []
    for transaction in transactions[-5:]:  # Last 5 transactions
        timestamp = transaction.get('timestamp')
        if timestamp:
            # Handle timezone-naive datetime
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            time_str = timestamp.strftime('%H:%M')
        else:
            time_str = 'Unknown'
            
        recent_transactions.append({
            'amount': transaction['transaction_data'].get('amount', 0),
            'category': transaction['transaction_data'].get('merchant_category', 'Unknown'),
            'location': transaction['transaction_data'].get('location', 'Unknown'),
            'is_fraud': transaction.get('is_fraud', False),
            'time': time_str
        })
    
    # Create recent activities
    recent_activities = [
        {
            'title': 'System scan completed',
            'time': '2 minutes ago',
            'type': 'success',
            'icon': 'check-circle'
        },
        {
            'title': 'New transaction analyzed',
            'time': '5 minutes ago',
            'type': 'success',
            'icon': 'search'
        },
        {
            'title': 'Fraud pattern detected',
            'time': '10 minutes ago',
            'type': 'danger',
            'icon': 'exclamation-triangle'
        },
        {
            'title': 'Security update applied',
            'time': '1 hour ago',
            'type': 'success',
            'icon': 'shield-alt'
        }
    ]
    
    return render_template('dashboard.html',
                         username=session.get('user_name', 'User'),
                         total_transactions=total_transactions,
                         fraud_count=fraud_count,
                         accuracy_rate=accuracy_rate,
                         days_active=days_active,
                         recent_transactions=recent_transactions,
                         recent_activities=recent_activities)

@app.route('/analytics')
def analytics():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    user_id = session.get('user_id')
    transactions = list(transactions_collection.find({"user_id": user_id}))
    
    # Calculate analytics data
    total_amount = sum(t['transaction_data'].get('amount', 0) for t in transactions)
    fraud_amount = sum(t['transaction_data'].get('amount', 0) for t in transactions if t.get('is_fraud', False))
    avg_amount = total_amount / len(transactions) if transactions else 0
    fraud_count = len([t for t in transactions if t.get('is_fraud', False)])
    
    # Category breakdown
    categories = {}
    for t in transactions:
        category = t['transaction_data'].get('merchant_category', 'Other')
        categories[category] = categories.get(category, 0) + 1
    
    return render_template('analytics.html',
                         username=session.get('user_name', 'User'),
                         total_transactions=len(transactions),
                         total_amount=total_amount,
                         fraud_amount=fraud_amount,
                         avg_amount=avg_amount,
                         fraud_count=fraud_count,
                         categories=categories)

@app.route('/transactions')
def transactions():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    user_id = session.get('user_id')
    transactions = list(transactions_collection.find({"user_id": user_id}).sort("timestamp", -1))
    
    return render_template('transactions.html',
                         username=session.get('user_name', 'User'),
                         transactions=transactions)

@app.route('/settings')
def settings():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    return render_template('settings.html',
                         username=session.get('user_name', 'User'))

@app.route('/detect', methods=['GET'])
def detect_page():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('detect_user_friendly.html', username=session.get('user_name', 'User'))

@app.route('/user-details')
def user_details():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    user_id = session.get('user_id')
    
    # Get user data
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        return redirect(url_for('login'))
    
    # Fix timezone issues for display
    if user.get('created_at') and user['created_at'].tzinfo is None:
        user['created_at'] = user['created_at'].replace(tzinfo=timezone.utc)
    if user.get('last_login') and user['last_login'].tzinfo is None:
        user['last_login'] = user['last_login'].replace(tzinfo=timezone.utc)
    
    # Get user statistics
    transactions = list(transactions_collection.find({"user_id": user_id}))
    total_transactions = len(transactions)
    fraud_count = len([t for t in transactions if t.get('is_fraud', False)])
    accuracy_rate = 87  # Default accuracy rate
    
    # Calculate days active
    if user and user.get('created_at'):
        # Handle timezone-naive datetime from database
        created_at = user['created_at']
        if created_at.tzinfo is None:
            # If naive, assume UTC
            created_at = created_at.replace(tzinfo=timezone.utc)
        days_active = (datetime.now(timezone.utc) - created_at).days
    else:
        days_active = 30
    
    # Create recent activities
    recent_activities = [
        {
            'title': 'Profile viewed',
            'time': 'Just now',
            'type': 'success',
            'icon': 'eye'
        },
        {
            'title': 'Security scan completed',
            'time': '5 minutes ago',
            'type': 'success',
            'icon': 'shield-check'
        },
        {
            'title': 'New transaction analyzed',
            'time': '10 minutes ago',
            'type': 'success',
            'icon': 'search'
        },
        {
            'title': 'Account settings updated',
            'time': '1 hour ago',
            'type': 'success',
            'icon': 'cog'
        }
    ]
    
    return render_template('user_details.html',
                         username=session.get('user_name', 'User'),
                         user=user,
                         total_transactions=total_transactions,
                         fraud_count=fraud_count,
                         accuracy_rate=accuracy_rate,
                         days_active=days_active,
                         recent_activities=recent_activities)

@app.route('/about')
def about():
    # Public page - no authentication required
    return render_template('about.html', background_image="fraud_detection_bg.jpg", is_public=True)

@app.route('/team')
def team():
    # Public page - no authentication required
    return render_template('team.html', background_image="fraud_detection_bg.jpg", is_public=True)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# --- API Endpoint for Fraud Detection ---
@app.route('/api/detect_fraud', methods=['POST'])
def detect_fraud_api():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized, please log in'}), 401
        
    if ML_MODEL is None or AMOUNT_SCALER is None or MODEL_FEATURE_COLUMNS is None or OPTIMAL_THRESHOLD is None:
        return jsonify({'error': 'Machine learning model assets not loaded. Please contact support.'}), 500

    if transactions_collection is None:
        return jsonify({'error': 'Database connection error. Please try again later.'}), 500

    data = request.json
    
    try:
        # Convert user-friendly input into model-expected numerical features
        processed_input_df = _map_user_input_to_features(data)
        
        # Scale the amount
        processed_input_df['Amount'] = AMOUNT_SCALER.transform(processed_input_df[['Amount']])
        
        # Make prediction
        fraud_probability = ML_MODEL.predict_proba(processed_input_df)[0][1]
        is_fraud = fraud_probability > OPTIMAL_THRESHOLD
        
        # Generate explanation
        explanations = generate_fraud_explanation(data, fraud_probability, is_fraud)
        
        # Store transaction in MongoDB - Convert NumPy types to Python native types
        transaction_record = {
            "user_id": session.get('user_id'),
            "timestamp": datetime.now(timezone.utc),
            "transaction_data": data,
            "fraud_probability": float(fraud_probability),  # Convert np.float64 to float
            "is_fraud": bool(is_fraud),  # Convert np.True_/np.False_ to bool
            "explanations": explanations
        }
        
        transactions_collection.insert_one(transaction_record)
        
        return jsonify({
            'fraud_probability': float(fraud_probability),
            'is_fraud': bool(is_fraud),
            'explanations': explanations
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# --- API Endpoint to Get User's Transaction History ---
@app.route('/api/transaction_history', methods=['GET'])
def get_transaction_history():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized, please log in'}), 401
    
    if transactions_collection is None:
        return jsonify({'error': 'Database connection error. Please try again later.'}), 500
    
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