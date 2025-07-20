# MongoDB Migration Guide

This guide will help you migrate your fraud detection project from SQLite to MongoDB.

## **Prerequisites**

### **1. Install MongoDB**

**On Windows:**
1. Download MongoDB Community Server from [mongodb.com](https://www.mongodb.com/try/download/community)
2. Install with default settings
3. MongoDB will run as a Windows service

**On macOS:**
```bash
brew install mongodb-community
brew services start mongodb-community
```

**On Linux (Ubuntu):**
```bash
sudo apt update
sudo apt install mongodb
sudo systemctl start mongodb
```

### **2. Install Python Dependencies**
```bash
pip install pymongo flask-pymongo
```

## **Migration Steps**

### **Step 1: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 2: Start MongoDB**
Make sure MongoDB is running:
```bash
# Check if MongoDB is running
mongosh
# or
mongo
```

### **Step 3: Run Migration**
```bash
python migrate_to_mongodb.py
```

### **Step 4: Test MongoDB App**
```bash
python app_mongodb.py
```

## **Key Differences**

### **SQLite vs MongoDB**

| Feature | SQLite | MongoDB |
|---------|--------|---------|
| **Database Type** | Relational | Document-based |
| **Schema** | Fixed tables | Flexible documents |
| **Queries** | SQL | JSON-like queries |
| **Scalability** | Limited | Highly scalable |
| **Performance** | Good for small data | Excellent for large data |

### **Data Structure Comparison**

**SQLite (Users Table):**
```sql
CREATE TABLE user (
    id INTEGER PRIMARY KEY,
    email VARCHAR(120) UNIQUE,
    password VARCHAR(60)
);
```

**MongoDB (Users Collection):**
```json
{
  "_id": ObjectId("..."),
  "email": "user@example.com",
  "password": "$2b$12$...",
  "created_at": ISODate("2024-01-01T00:00:00Z"),
  "last_login": ISODate("2024-01-01T12:00:00Z")
}
```

## **New Features with MongoDB**

### **1. Transaction History**
- All fraud detection attempts are stored
- Users can view their history
- Better analytics and reporting

### **2. Enhanced User Management**
- User creation timestamps
- Last login tracking
- Better user analytics

### **3. Scalability**
- Can handle thousands of users
- Better performance with large datasets
- Cloud-ready for MongoDB Atlas

## **MongoDB Collections**

### **Users Collection**
```json
{
  "_id": ObjectId("..."),
  "email": "user@example.com",
  "password": "hashed_password",
  "created_at": ISODate("..."),
  "last_login": ISODate("..."),
  "migrated_from_sqlite": true
}
```

### **Transactions Collection**
```json
{
  "_id": ObjectId("..."),
  "user_id": "user_object_id",
  "timestamp": ISODate("..."),
  "transaction_data": {
    "amount": 1000,
    "merchant_category": "electronics",
    "location": "international",
    "time_of_day": "early morning"
  },
  "fraud_probability": 0.85,
  "is_fraud": true,
  "explanations": ["High amount", "International location"]
}
```

## **API Endpoints**

### **New Endpoints in MongoDB Version**

1. **GET /api/transaction_history**
   - Returns user's fraud detection history
   - Shows last 10 transactions
   - Includes fraud probability and explanations

## **Benefits of MongoDB Migration**

### **1. Performance**
- ✅ Faster queries for large datasets
- ✅ Better indexing capabilities
- ✅ Horizontal scaling

### **2. Flexibility**
- ✅ Schema-less design
- ✅ Easy to add new fields
- ✅ JSON-like data structure

### **3. Scalability**
- ✅ Can handle millions of users
- ✅ Cloud deployment ready
- ✅ Better for production use

### **4. Analytics**
- ✅ Better transaction history
- ✅ User behavior tracking
- ✅ Fraud pattern analysis

## **Troubleshooting**

### **Common Issues**

1. **MongoDB Connection Error**
   ```bash
   # Make sure MongoDB is running
   mongosh
   ```

2. **Port Already in Use**
   ```bash
   # Check if MongoDB is running on port 27017
   netstat -an | findstr 27017
   ```

3. **Migration Fails**
   ```bash
   # Check if SQLite database exists
   ls instance/site.db
   ```

## **Next Steps**

1. **Test the migration** with existing users
2. **Deploy to cloud** using MongoDB Atlas
3. **Add more features** like user analytics
4. **Scale the application** for production use

## **Production Considerations**

### **MongoDB Atlas (Cloud)**
```python
# Update mongodb_config.py
MONGO_URI = "mongodb+srv://username:password@cluster.mongodb.net/fraud_detection_db"
```

### **Security**
- Use environment variables for database credentials
- Enable MongoDB authentication
- Use SSL connections in production

### **Backup**
- Set up automated backups
- Test restore procedures
- Monitor database performance 