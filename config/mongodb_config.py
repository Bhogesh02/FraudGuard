# mongodb_config.py
import os
from dotenv import load_dotenv
load_dotenv()
from flask_pymongo import PyMongo

MONGO_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/fraud_detection_db")

def init_mongodb(app):
    app.config["MONGO_URI"] = MONGO_URI
    return PyMongo(app)

def create_user_collection(mongo):
    return mongo.db.users

def create_transaction_collection(mongo):
    return mongo.db.transactions 