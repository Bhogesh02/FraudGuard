# migrate_to_mongodb.py
import sqlite3
try:
    from pymongo import MongoClient
except ImportError as e:
    raise ImportError("pymongo is not installed. Please install it with 'pip install pymongo'") from e
from flask_bcrypt import Bcrypt
from datetime import datetime
import os

def migrate_sqlite_to_mongodb():
    """Migrate user data from SQLite to MongoDB Atlas"""
    
    # MongoDB Atlas connection
    mongo_client = MongoClient("mongodb+srv://DivyaBhogesh:Bhogesh0227@fraudguard.oobjhge.mongodb.net/")
    db = mongo_client.fraud_detection_db
    users_collection = db.users
    
    # SQLite connection
    sqlite_conn = sqlite3.connect('instance/site.db')
    sqlite_cursor = sqlite_conn.cursor()
    
    # Initialize Bcrypt for password verification
    bcrypt = Bcrypt()
    
    try:
        # Get all users from SQLite
        sqlite_cursor.execute("SELECT id, email, password FROM user")
        users = sqlite_cursor.fetchall()
        
        print(f"Found {len(users)} users in SQLite database")
        
        migrated_count = 0
        skipped_count = 0
        
        for user_id, email, password in users:
            # Check if user already exists in MongoDB
            existing_user = users_collection.find_one({"email": email})
            
            if existing_user:
                print(f"User {email} already exists in MongoDB Atlas, skipping...")
                skipped_count += 1
                continue
            
            # Create new user document for MongoDB
            new_user = {
                "email": email,
                "password": password,  # Password is already hashed
                "created_at": datetime.utcnow(),
                "last_login": None,
                "migrated_from_sqlite": True,
                "original_sqlite_id": user_id
            }
            
            # Insert into MongoDB Atlas
            result = users_collection.insert_one(new_user)
            print(f"Migrated user {email} to MongoDB Atlas with ID: {result.inserted_id}")
            migrated_count += 1
        
        print(f"\nMigration completed!")
        print(f"Migrated: {migrated_count} users")
        print(f"Skipped: {skipped_count} users (already existed)")
        
        # Create indexes
        users_collection.create_index("email", unique=True)
        users_collection.create_index("user_id")
        
        print("MongoDB Atlas indexes created successfully")
        
    except Exception as e:
        print(f"Migration failed: {e}")
    finally:
        sqlite_conn.close()
        mongo_client.close()

def verify_migration():
    """Verify that migration was successful"""
    
    # MongoDB Atlas connection
    mongo_client = MongoClient("mongodb+srv://DivyaBhogesh:Bhogesh0227@fraudguard.oobjhge.mongodb.net/")
    db = mongo_client.fraud_detection_db
    users_collection = db.users
    
    # SQLite connection
    sqlite_conn = sqlite3.connect('instance/site.db')
    sqlite_cursor = sqlite_conn.cursor()
    
    try:
        # Count users in both databases
        sqlite_cursor.execute("SELECT COUNT(*) FROM user")
        sqlite_count = sqlite_cursor.fetchone()[0]
        
        mongo_count = users_collection.count_documents({})
        
        print(f"SQLite users: {sqlite_count}")
        print(f"MongoDB Atlas users: {mongo_count}")
        
        if mongo_count >= sqlite_count:
            print("✅ Migration verification successful!")
        else:
            print("❌ Migration verification failed - some users may not have been migrated")
        
        # Show sample users from MongoDB Atlas
        print("\nSample users in MongoDB Atlas:")
        for user in users_collection.find().limit(3):
            print(f"- {user['email']} (ID: {user['_id']})")
        
    except Exception as e:
        print(f"Verification failed: {e}")
    finally:
        sqlite_conn.close()
        mongo_client.close()

def test_mongodb_connection():
    """Test MongoDB Atlas connection"""
    try:
        mongo_client = MongoClient("mongodb+srv://DivyaBhogesh:Bhogesh0227@fraudguard.oobjhge.mongodb.net/")
        db = mongo_client.fraud_detection_db
        
        # Test connection by listing collections
        collections = db.list_collection_names()
        print(f"✅ MongoDB Atlas connection successful!")
        print(f"Available collections: {collections}")
        
        mongo_client.close()
        return True
        
    except Exception as e:
        print(f"❌ MongoDB Atlas connection failed: {e}")
        return False

if __name__ == "__main__":
    print("=== SQLite to MongoDB Atlas Migration Tool ===")
    
    print("1. Testing MongoDB Atlas connection...")
    if not test_mongodb_connection():
        print("Migration aborted due to connection failure.")
        exit(1)
    
    print("\n2. Migrating users from SQLite to MongoDB Atlas...")
    migrate_sqlite_to_mongodb()
    
    print("\n3. Verifying migration...")
    verify_migration()
    
    print("\n=== Migration Complete ===")
    print("You can now use app_mongodb.py instead of app.py")
    print("SQLite is no longer needed!") 