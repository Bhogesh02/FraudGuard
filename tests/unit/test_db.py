#!/usr/bin/env python3
"""
Test script to verify MongoDB connection and user operations
"""

from app_mongodb import users_collection, transactions_collection
from datetime import datetime, timezone

def test_database_connection():
    """Test if database connection is working"""
    try:
        # Test users collection
        if users_collection is not None:
            print("✅ Users collection is available")
            
            # Count existing users
            user_count = users_collection.count_documents({})
            print(f"📊 Total users in database: {user_count}")
            
            # List all users (without passwords)
            users = list(users_collection.find({}, {"password": 0}))
            if users:
                print("👥 Users in database:")
                for user in users:
                    print(f"  - {user.get('name', 'Unknown')} ({user.get('email', 'No email')}) from {user.get('country', 'Unknown')}")
            else:
                print("📝 No users found in database")
        else:
            print("❌ Users collection is not available")
            
        # Test transactions collection
        if transactions_collection is not None:
            print("✅ Transactions collection is available")
            
            # Count existing transactions
            transaction_count = transactions_collection.count_documents({})
            print(f"📊 Total transactions in database: {transaction_count}")
        else:
            print("❌ Transactions collection is not available")
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")

def test_user_creation():
    """Test creating a test user"""
    try:
        if users_collection is not None:
            # Create a test user
            test_user = {
                "name": "Test User",
                "email": "test@example.com",
                "country": "United States",
                "password": "hashed_password_here",
                "created_at": datetime.now(timezone.utc),
                "last_login": None
            }
            
            # Check if test user already exists
            existing = users_collection.find_one({"email": "test@example.com"})
            if existing:
                print("ℹ️  Test user already exists")
            else:
                # Insert test user
                result = users_collection.insert_one(test_user)
                print(f"✅ Test user created with ID: {result.inserted_id}")
                
        else:
            print("❌ Cannot create test user - collection not available")
            
    except Exception as e:
        print(f"❌ User creation test failed: {e}")

if __name__ == "__main__":
    print("🔍 Testing FraudGuard Database Connection")
    print("=" * 50)
    
    test_database_connection()
    print("\n" + "=" * 50)
    test_user_creation()
    print("\n" + "=" * 50)
    print("🏁 Database test completed!") 