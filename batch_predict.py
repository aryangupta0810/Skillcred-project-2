#!/usr/bin/env python3
"""
Batch Prediction Script for Product Price Predictor
This script allows you to make multiple predictions at once
"""

import joblib
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

def load_model():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('price_model.pkl')
        scaler = joblib.load('price_scaler.pkl')
        
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        
        print("✅ Model loaded successfully!")
        return model, scaler, model_info
        
    except FileNotFoundError as e:
        print(f"❌ Error loading model: {e}")
        print("Please run the training script first!")
        return None, None, None

def predict_batch(model, scaler, input_data):
    """Make batch predictions"""
    try:
        # Convert to numpy array
        features = np.array(input_data)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make predictions
        predictions = model.predict(features_scaled)
        
        return predictions
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None

def main():
    print("🚀 Product Price Predictor - Batch Prediction")
    print("=" * 60)
    
    # Load model
    model, scaler, model_info = load_model()
    if model is None:
        return
    
    print(f"Model: {model_info['model_name']}")
    print(f"Performance: RMSE=${model_info['performance']['rmse']:.2f}, R²={model_info['performance']['r2']:.4f}")
    
    # Sample batch data
    print("\n📊 Sample Batch Data:")
    sample_data = [
        [100, 4.5, 50, 200, 2],    # Electronics with good rating
        [50, 3.8, 30, 150, 0],     # Men's clothing with moderate rating
        [200, 4.9, 80, 300, 3],    # Women's clothing with excellent rating
        [25, 4.2, 40, 180, 1],     # Jewelry with good rating
        [150, 4.7, 60, 250, 2],    # Electronics with very good rating
    ]
    
    # Feature names
    feature_names = ['rating_count', 'rating_rate', 'title_length', 'description_length', 'category_encoded']
    categories = {0: "men's clothing", 1: "jewelery", 2: "electronics", 3: "women's clothing"}
    
    # Display sample data
    print("\nInput Features:")
    for i, data in enumerate(sample_data):
        print(f"\nProduct {i+1}:")
        for j, (name, value) in enumerate(zip(feature_names, data)):
            if name == 'category_encoded':
                print(f"  {name}: {value} ({categories[value]})")
            else:
                print(f"  {name}: {value}")
    
    # Make predictions
    print("\n🎯 Making Predictions...")
    predictions = predict_batch(model, scaler, sample_data)
    
    if predictions is not None:
        print("\n📊 Prediction Results:")
        print("-" * 60)
        
        results = []
        for i, (data, pred) in enumerate(zip(sample_data, predictions)):
            result = {
                'product_id': i + 1,
                'rating_count': data[0],
                'rating_rate': data[1],
                'title_length': data[2],
                'description_length': data[3],
                'category': categories[data[4]],
                'predicted_price': round(pred, 2)
            }
            results.append(result)
            
            print(f"Product {i+1}:")
            print(f"  Category: {result['category']}")
            print(f"  Rating: {result['rating_rate']}/5 ({result['rating_count']} reviews)")
            print(f"  Predicted Price: ${result['predicted_price']}")
            print()
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        output_file = f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"✅ Results saved to: {output_file}")
        
        # Summary statistics
        print("\n📈 Summary Statistics:")
        print(f"  Total predictions: {len(predictions)}")
        print(f"  Average predicted price: ${np.mean(predictions):.2f}")
        print(f"  Price range: ${np.min(predictions):.2f} - ${np.max(predictions):.2f}")
        print(f"  Standard deviation: ${np.std(predictions):.2f}")
        
        # Category analysis
        print("\n🏷️  Analysis by Category:")
        category_analysis = results_df.groupby('category')['predicted_price'].agg(['count', 'mean', 'min', 'max'])
        print(category_analysis.to_string())
        
    print("\n" + "=" * 60)
    print("🎉 Batch prediction completed!")
    print("You can now use the web app for interactive predictions:")
    print("  - Flask: python app.py")
    print("  - Streamlit: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
