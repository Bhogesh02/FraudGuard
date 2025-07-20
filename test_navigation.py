#!/usr/bin/env python3
"""
Test script to verify navigation highlighting is working
=====================================================

This script tests that the active navigation states are working correctly.
"""

import urllib.request
import urllib.error
import time
import re

def test_navigation_highlighting():
    """Test that navigation highlighting is working on different pages"""
    
    base_url = "http://127.0.0.1:5000"
    
    print("🧪 Testing FraudGuard Navigation Highlighting")
    print("=" * 50)
    
    # Test pages and their expected active nav items
    test_pages = [
        ("/dashboard", "Dashboard"),
        ("/detect", "Detect Fraud"),
        ("/analytics", "Analytics"),
        ("/user-details", "Profile")
    ]
    
    for page_url, expected_active in test_pages:
        print(f"\n📋 Testing {page_url} - Expected active: {expected_active}")
        try:
            req = urllib.request.Request(f"{base_url}{page_url}")
            response = urllib.request.urlopen(req, timeout=5)
            
            if response.getcode() == 200:
                content = response.read().decode('utf-8')
                
                # Check if the expected active class is present
                if f'class="nav-link active"' in content:
                    # Find which nav item is active
                    active_match = re.search(r'class="nav-link active"[^>]*>.*?<i[^>]*></i>\s*([^<]+)', content)
                    if active_match:
                        actual_active = active_match.group(1).strip()
                        if actual_active == expected_active:
                            print(f"✅ {page_url} - Active nav item: {actual_active} ✓")
                        else:
                            print(f"❌ {page_url} - Expected '{expected_active}', got '{actual_active}'")
                    else:
                        print(f"⚠️  {page_url} - Active class found but couldn't determine which item")
                else:
                    print(f"❌ {page_url} - No active navigation class found")
                    
            else:
                print(f"❌ {page_url} - Status code: {response.getcode()}")
                
        except urllib.error.HTTPError as e:
            if e.code == 302:  # Redirect to login
                print(f"✅ {page_url} - Correctly redirects to login (as expected)")
            else:
                print(f"❌ {page_url} - HTTP Error: {e.code}")
        except Exception as e:
            print(f"❌ {page_url} - Error: {e}")
    
    print("\n🎯 Navigation Testing Complete!")
    print("\n💡 To manually test:")
    print("   1. Login to the application")
    print("   2. Navigate between Dashboard, Detect, Analytics, and Profile")
    print("   3. Verify the current page is highlighted in blue in the navigation")

if __name__ == "__main__":
    print("🚀 Starting navigation highlighting tests...")
    print("Make sure the Flask app is running on http://127.0.0.1:5000")
    print()
    
    # Wait a moment for the app to be ready
    time.sleep(2)
    
    test_navigation_highlighting() 