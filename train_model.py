#!/usr/bin/env python3
"""
FraudGuard Model Training Script with Progress Indicators
=======================================================

This script demonstrates the enhanced training process with:
- Real-time progress bars
- Time estimates
- Animated headers
- Step-by-step progress tracking
- Beautiful completion messages

Usage:
    python train_model.py
"""

import subprocess
import sys
import os

def main():
    """Run the enhanced model training with progress indicators"""
    
    print("🚀 Starting FraudGuard Model Training with Progress Indicators")
    print("=" * 70)
    
    # Check if required files exist
    if not os.path.exists('fraud_model.py'):
        print("❌ Error: fraud_model.py not found!")
        print("Please make sure you're in the correct directory.")
        return
    
    # Check if creditcard.csv exists
    if not os.path.exists('creditcard.csv'):
        print("⚠️  Warning: creditcard.csv not found.")
        print("The script will create synthetic data for training.")
        print()
    
    # Run the training script
    try:
        print("📊 Starting training process...")
        print("💡 You'll see:")
        print("   • Animated progress bars")
        print("   • Real-time time estimates")
        print("   • Step-by-step progress tracking")
        print("   • Beautiful completion messages")
        print()
        
        # Run the fraud_model.py script
        result = subprocess.run([sys.executable, 'fraud_model.py'], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n🎉 Training completed successfully!")
            print("✅ Model files have been saved:")
            print("   • fraud_detector_model.pkl")
            print("   • amount_scaler.pkl")
            print("   • model_feature_columns.pkl")
            print("   • optimal_threshold.pkl")
        else:
            print(f"\n❌ Training failed with return code: {result.returncode}")
            
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")

if __name__ == "__main__":
    main() 