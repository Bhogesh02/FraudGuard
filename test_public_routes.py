#!/usr/bin/env python3
"""
Test script to verify About and Team routes are working as public pages
=====================================================================

This script tests that the About and Team pages are accessible without login.
"""

import urllib.request
import urllib.error
import time

def test_public_routes():
    """Test the About and Team routes as public pages"""
    
    base_url = "http://127.0.0.1:5000"
    
    print("🧪 Testing FraudGuard Public Routes")
    print("=" * 45)
    
    # Test About route
    print("\n📋 Testing About Route (Public)...")
    try:
        req = urllib.request.Request(f"{base_url}/about")
        response = urllib.request.urlopen(req, timeout=5)
        if response.getcode() == 200:
            print("✅ About page is accessible as public page!")
            print(f"   Status Code: {response.getcode()}")
            print(f"   Content Length: {len(response.read())} characters")
        else:
            print(f"❌ About page returned status code: {response.getcode()}")
    except urllib.error.URLError as e:
        print(f"❌ Error accessing About page: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test Team route
    print("\n👥 Testing Team Route (Public)...")
    try:
        req = urllib.request.Request(f"{base_url}/team")
        response = urllib.request.urlopen(req, timeout=5)
        if response.getcode() == 200:
            print("✅ Team page is accessible as public page!")
            print(f"   Status Code: {response.getcode()}")
            print(f"   Content Length: {len(response.read())} characters")
        else:
            print(f"❌ Team page returned status code: {response.getcode()}")
    except urllib.error.URLError as e:
        print(f"❌ Error accessing Team page: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test Login route (should redirect to login)
    print("\n🔐 Testing Login Route...")
    try:
        req = urllib.request.Request(f"{base_url}/dashboard")
        response = urllib.request.urlopen(req, timeout=5)
        print(f"   Dashboard redirect status: {response.getcode()}")
    except urllib.error.HTTPError as e:
        if e.code == 302:  # Redirect
            print("✅ Dashboard correctly redirects to login (as expected)")
        else:
            print(f"❌ Dashboard returned unexpected status: {e.code}")
    except Exception as e:
        print(f"❌ Error testing dashboard: {e}")
    
    print("\n🎯 Public Route Testing Complete!")
    print("\n💡 To manually test:")
    print("   1. Open browser and go to: http://127.0.0.1:5000")
    print("   2. Navigate to About page: http://127.0.0.1:5000/about")
    print("   3. Navigate to Team page: http://127.0.0.1:5000/team")
    print("   4. Verify both pages load without login requirement")

if __name__ == "__main__":
    print("🚀 Starting public route tests...")
    print("Make sure the Flask app is running on http://127.0.0.1:5000")
    print()
    
    # Wait a moment for the app to be ready
    time.sleep(2)
    
    test_public_routes() 