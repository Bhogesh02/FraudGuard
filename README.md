# 🛡️ FraudGuard - AI-Powered Fraud Detection System

A modern, professional fraud detection web application built with Flask, MongoDB, and Machine Learning.

## 👥 **Team Members**

- **Katna Lavanya** - Project Lead & Architect
- **Molli Tejaswi** - ML Engineer  
- **Mutchi Divya** - MERN stack developer 
- **Kuppili Shirisha Rao** - Backend Developer

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8+
- MongoDB (local or cloud)
- Required Python packages (see `requirements.txt`)

### **Installation & Setup**

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Credit_Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up MongoDB**
   - Install MongoDB locally or use MongoDB Atlas
   - Update connection settings in `config/mongodb_config.py`

4. **Run the application**
   ```bash
   python run_app.py
   ```

5. **Access the application**
   - Open your browser and go to: `http://127.0.0.1:5000`
   - Login with demo credentials: `demo@example.com` / `password`

## 📁 **Project Structure**

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
├── static/                  # CSS, JS, images
├── templates/               # HTML templates
├── tests/                   # Test files
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── logs/                   # Application logs
├── docs/                   # Documentation
├── requirements.txt         # Python dependencies
└── run_app.py              # Application entry point
```

## 🌐 **Available Pages**

### **Public Pages** (No login required)
- `/` - Home/Login page
- `/about` - About FraudGuard
- `/team` - Meet the team
- `/support` - Support & Help Center

### **Protected Pages** (Login required)
- `/dashboard` - Main dashboard with analytics
- `/detect` - Fraud detection interface
- `/analytics` - Detailed analytics and charts
- `/user-details` - User profile management
- `/settings` - Account settings

### **API Endpoints**
- `/api/detect_fraud` - Fraud detection API
- `/api/transaction_history` - Transaction history API

## 🛠️ **Features**

### **Core Functionality**
- ✅ Real-time fraud detection using ML models
- ✅ User authentication and session management
- ✅ MongoDB database integration
- ✅ Responsive modern UI with animations
- ✅ Analytics dashboard with charts
- ✅ Support system with contact forms

### **Technical Features**
- ✅ Machine Learning model integration
- ✅ RESTful API endpoints
- ✅ Secure password hashing
- ✅ Session-based authentication
- ✅ Real-time data visualization
- ✅ Professional error handling

### **UI/UX Features**
- ✅ Modern dark theme design
- ✅ Responsive navigation with active states
- ✅ Animated cards and hover effects
- ✅ Professional footer with team branding
- ✅ Consistent styling across all pages

## 🔧 **Configuration**

### **MongoDB Setup**
1. Update `config/mongodb_config.py` with your MongoDB connection string
2. Ensure MongoDB is running locally or accessible via cloud

### **Environment Variables**
- `SECRET_KEY` - Flask secret key (auto-generated)
- `MONGODB_URI` - MongoDB connection string

## 🧪 **Testing**

### **Run Tests**
```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/
```

### **Manual Testing**
1. Start the application: `python run_app.py`
2. Test all navigation links
3. Test fraud detection functionality
4. Test user authentication
5. Test support form submission

## 📞 **Support**

- **Email**: support@fraudguard.com
- **Phone**: +1 234 567 890
- **Team Lead**: Katna Lavanya
- **Support Team**: Molli Tejaswi, Mutchi Divya, Kuppili Shirisha Rao

## 🚀 **Deployment**

### **Local Development**
```bash
python run_app.py
```

### **Production Deployment**
1. Set `debug=False` in `run_app.py`
2. Use a production WSGI server (Gunicorn, uWSGI)
3. Configure environment variables
4. Set up proper MongoDB security

## 📊 **Model Information**

- **Model Type**: Machine Learning (scikit-learn)
- **Features**: 29 engineered features
- **Accuracy**: 95%+ detection rate
- **Real-time**: Sub-second prediction time

## 🔒 **Security Features**

- Password hashing with bcrypt
- Session-based authentication
- CSRF protection
- Input validation and sanitization
- Secure MongoDB connections

## 📈 **Performance**

- **Response Time**: < 1 second for fraud detection
- **Concurrent Users**: Supports multiple simultaneous users
- **Database**: Optimized MongoDB queries
- **Frontend**: Optimized CSS and JavaScript

## 🎯 **Future Enhancements**

- [ ] Real-time notifications
- [ ] Advanced analytics dashboard
- [ ] Mobile app development
- [ ] API rate limiting
- [ ] Advanced ML model training
- [ ] Multi-language support

---

**© 2024 FraudGuard. All rights reserved.**

*Built with ❤️ by the FraudGuard Team* 