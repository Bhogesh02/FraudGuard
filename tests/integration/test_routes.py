#!/usr/bin/env python3
"""
Test script to verify About and Team routes are working
=====================================================

This script tests the About and Team page routes to ensure they're accessible.
"""

import requests
import time

def test_routes():
    """Test the About and Team routes"""
    
    base_url = "http://127.0.0.1:5000"
    
    print("🧪 Testing FraudGuard Routes")
    print("=" * 40)
    
    # Test About route
    print("\n📋 Testing About Route...")
    try:
        response = requests.get(f"{base_url}/about", timeout=5)
        if response.status_code == 200:
            print("✅ About page is accessible!")
            print(f"   Status Code: {response.status_code}")
            print(f"   Content Length: {len(response.text)} characters")
        else:
            print(f"❌ About page returned status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error accessing About page: {e}")
    
    # Test Team route
    print("\n👥 Testing Team Route...")
    try:
        response = requests.get(f"{base_url}/team", timeout=5)
        if response.status_code == 200:
            print("✅ Team page is accessible!")
            print(f"   Status Code: {response.status_code}")
            print(f"   Content Length: {len(response.text)} characters")
        else:
            print(f"❌ Team page returned status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error accessing Team page: {e}")
    
    # Test Home route
    print("\n🏠 Testing Home Route...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Home page is accessible!")
            print(f"   Status Code: {response.status_code}")
        else:
            print(f"❌ Home page returned status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error accessing Home page: {e}")
    
    print("\n🎯 Route Testing Complete!")
    print("\n💡 To manually test:")
    print("   1. Open browser and go to: http://127.0.0.1:5000")
    print("   2. Navigate to About page: http://127.0.0.1:5000/about")
    print("   3. Navigate to Team page: http://127.0.0.1:5000/team")

if __name__ == "__main__":
    print("🚀 Starting route tests...")
    print("Make sure the Flask app is running on http://127.0.0.1:5000")
    print()
    
    # Wait a moment for the app to be ready
    time.sleep(2)
    
    test_routes() 