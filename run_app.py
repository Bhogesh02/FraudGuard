#!/usr/bin/env python3
"""
FraudGuard Application Entry Point
=================================

This script runs the FraudGuard Flask application with the new folder structure.
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the main app from controllers
from controllers.app_mongodb import app

if __name__ == '__main__':
    print("🚀 Starting FraudGuard Application...")
    print("📁 Using new optimized folder structure")
    print("🌐 Server will be available at: http://127.0.0.1:5000")
    print("📧 Support: support@fraudguard.com")
    print("👥 Team: Katna Lavanya, Molli Tejaswi, Mutchi Divya, Kuppili Shirisha Rao")
    print()
    
    # Run the Flask app
    app.run(debug=True, host='127.0.0.1', port=5000) 