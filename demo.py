#!/usr/bin/env python3
"""
Demo script for Product Price Predictor
This script demonstrates the data collection and shows sample data
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🚀 Product Price Predictor - Demo")
    print("=" * 50)
    
    # Test data collection
    print("\n1. Testing data collection...")
    try:
        from data.collect_data import main as collect_data
        collect_data()
        print("✅ Data collection completed successfully!")
    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        return
    
    # Check if data was created
    data_file = "data/products.csv"
    if os.path.exists(data_file):
        print(f"\n2. Data file created: {data_file}")
        
        # Show sample data
        try:
            import pandas as pd
            df = pd.read_csv(data_file)
            print(f"\n📊 Dataset Overview:")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
            print(f"   Categories: {df['category'].nunique()}")
            
            print(f"\n📋 Sample Data (first 3 rows):")
            print(df.head(3).to_string(index=False))
            
        except Exception as e:
            print(f"❌ Error reading data: {e}")
    else:
        print(f"❌ Data file not found: {data_file}")
    
    print("\n" + "=" * 50)
    print("🎯 Next Steps:")
    print("1. Run 'python notebooks/train_model.py' to train the model")
    print("2. Set your GEMINI_API_KEY environment variable")
    print("3. Run 'python app.py' for Flask app or 'streamlit run streamlit_app.py' for Streamlit")
    print("4. Open your browser to the provided URL")
    print("\nHappy predicting! 🚀")

if __name__ == "__main__":
    main()
