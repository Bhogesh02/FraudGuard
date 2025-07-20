# 🛡️ FraudGuard - AI-Powered Fraud Detection System

A modern, professional fraud detection web application built with Flask, MongoDB, and Machine Learning.

## 👥 **Team Members**
- **Katna Lavanya** - Project Lead & Architect
- **Molli Tejaswi** - ML Engineer  
- **Mutchi Divya** - MERN stack developer 
- **Kuppili Shirisha Rao** - Backend Developer

---

## 🚀 **Getting Started: Step-by-Step**

### 1. **Clone the Repository**
```bash
git clone <repository-url>
cd Credit_Project
```

### 2. **Set Up a Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Configure MongoDB**
- Install MongoDB locally **or** create a free cluster on [MongoDB Atlas](https://www.mongodb.com/atlas).
- Update your connection string in `config/mongodb_config.py`.

### 5. **Train the Model (First-Time Setup Only)**
> **IMPORTANT:**
> If you are cloning the project for the first time and the following files are NOT present in `data/models/`:
> - `fraud_detector_model.pkl`
> - `amount_scaler.pkl`
> - `model_feature_columns.pkl`
> - (optionally) `optimal_threshold.pkl`
>
> **You MUST run the training script:**
> ```bash
> python train_model.py
> ```
> This will generate the required model files for fraud detection.

### 6. **Run the Application**
```bash
python run_app.py
```
- The app will be available at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 7. **Login & Explore**
- Register a new user or use demo credentials if provided.
- Explore dashboard, fraud detection, analytics, and more!

---

## 📁 **Folder Structure**

```
Credit_Project/
├── src/
│   ├── controllers/          # Flask route handlers
│   │   └── app_mongodb.py   # Main application controller
│   ├── models/              # ML models and data models
│   │   └── fraud_model.py   # Fraud detection model
│   ├── services/            # Business logic and services
│   │   └── migrate_to_mongodb.py
│   └── utils/               # Helper functions
├── config/                  # Configuration files
│   └── mongodb_config.py    # MongoDB configuration
├── data/
│   ├── models/              # Trained ML models (.pkl files)
│   └── datasets/            # Raw data files
├── static/                  # CSS, JS, images, favicon
├── templates/               # HTML templates
├── tests/                   # Test files
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── logs/                   # Application logs
├── docs/                   # Documentation
├── requirements.txt         # Python dependencies
└── run_app.py              # Application entry point
```

---

## 🌐 **Key Pages & Routes**

### **Public Pages**
- `/` - Home/Login
- `/about` - About FraudGuard
- `/team` - Development Team
- `/support` - Support & Help Center

### **Protected Pages (Login Required)**
- `/dashboard` - Main dashboard
- `/detect` - Fraud detection
- `/analytics` - Analytics & charts
- `/transactions` - All transactions (with pagination)
- `/user-details` - Profile
- `/settings` - Account settings

### **API Endpoints**
- `/api/detect_fraud` - Fraud detection API
- `/api/transaction_history` - Transaction history API

---

## 🧪 **Testing**

### **Run Automated Tests**
```bash
# Unit tests
python -m pytest tests/unit/
# Integration tests
python -m pytest tests/integration/
```

### **Manual Testing**
1. Start the app: `python run_app.py`
2. Register/login and navigate all pages
3. Test fraud detection, analytics, and support
4. Try edge cases (invalid login, large transactions, etc.)

---

## 🚀 **Deployment Guide**

### **Local Development**
```bash
python run_app.py
```

### **Production Deployment (Recommended)**
1. Set `debug=False` in `run_app.py`
2. Use a WSGI server (e.g., Gunicorn):
   ```bash
   gunicorn run_app:app
   ```
3. Set environment variables for secrets and DB connection
4. Deploy to a cloud platform (Heroku, Render, AWS, etc.)
5. Use a production MongoDB (Atlas or secured server)

---

## 🛠️ **Features**
- Real-time fraud detection (ML-powered)
- User authentication & session management
- MongoDB integration
- Responsive, modern UI
- Analytics dashboard
- Support/contact system
- Pagination for transactions
- Professional error handling

---

## 📞 **Support**
- **Email**: support@fraudguard.com
- **Team Lead**: Katna Lavanya

---

## 🎯 **Future Enhancements**
- Real-time notifications
- Advanced analytics
- Mobile app
- API rate limiting
- Multi-language support

---

**© 2024 FraudGuard. All rights reserved.**

*Built with ❤️ by the FraudGuard Team* 