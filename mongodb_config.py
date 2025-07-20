# mongodb_config.py
from flask_pymongo import PyMongo
from pymongo import MongoClient
import os

# MongoDB Configuration - Updated to MongoDB Atlas
MONGO_URI = "mongodb+srv://DivyaBhogesh:Bhogesh0227@fraudguard.oobjhge.mongodb.net/fraud_detection_db"

# For local MongoDB (if needed):
# MONGO_URI = "mongodb://localhost:27017/fraud_detection_db"

def init_mongodb(app):
    """Initialize MongoDB connection"""
    app.config["MONGO_URI"] = MONGO_URI
    mongo = PyMongo(app)
    return mongo

def get_mongodb_client():
    """Get MongoDB client for direct operations"""
    return MongoClient(MONGO_URI)

def create_user_collection(mongo):
    """Create users collection with indexes"""
    users_collection = mongo.db.users
    
    # Create unique index on email
    users_collection.create_index("email", unique=True)
    
    # Create index on user_id for faster lookups
    users_collection.create_index("user_id")
    
    return users_collection

def create_transaction_collection(mongo):
    """Create transactions collection for storing fraud detection history"""
    transactions_collection = mongo.db.transactions
    
    # Create indexes for better performance
    transactions_collection.create_index("user_id")
    transactions_collection.create_index("timestamp")
    transactions_collection.create_index("fraud_probability")
    
    return transactions_collection 