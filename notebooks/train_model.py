# Product Price Prediction Model Training
# This script trains a regression model to predict product prices based on product features.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def main():
    print("=== Product Price Prediction Model Training ===")
    
    # 1. Load and explore the dataset
    print("\n1. Loading dataset...")
    try:
        df = pd.read_csv('../data/products.csv')
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print("❌ Error: products.csv not found. Please run data/collect_data.py first.")
        return
    
    # 2. Basic data exploration
    print("\n2. Data exploration...")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nUnique categories: {df['category'].nunique()}")
    print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"Average price: ${df['price'].mean():.2f}")
    
    # 3. Feature engineering and preprocessing
    print("\n3. Feature engineering...")
    feature_columns = ['rating_count', 'rating_rate', 'title_length', 'description_length', 'category_encoded']
    target_column = 'price'
    
    X = df[feature_columns]
    y = df[target_column]
    
    print(f"Feature columns: {feature_columns}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled successfully!")
    
    # 4. Model training and evaluation
    print("\n4. Training models...")
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        
        # Train the model
        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred
        }
        
        print(f"{name} Results:")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAE: ${mae:.2f}")
        print(f"  R² Score: {r2:.4f}")
    
    # 5. Find the best model
    print("\n5. Model comparison...")
    best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
    best_model = results[best_model_name]['model']
    best_rmse = results[best_model_name]['rmse']
    best_r2 = results[best_model_name]['r2']
    
    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL: {best_model_name}")
    print(f"RMSE: ${best_rmse:.2f}")
    print(f"R² Score: {best_r2:.4f}")
    print(f"{'='*60}")
    
    # 6. Save the trained model
    print("\n6. Saving model...")
    model_filename = '../price_model.pkl'
    scaler_filename = '../price_scaler.pkl'
    
    # Save the model
    joblib.dump(best_model, model_filename)
    print(f"✅ Model saved to {model_filename}")
    
    # Save the scaler
    joblib.dump(scaler, scaler_filename)
    print(f"✅ Scaler saved to {scaler_filename}")
    
    # Save feature names for later use
    feature_info = {
        'feature_columns': feature_columns,
        'target_column': target_column,
        'model_name': best_model_name,
        'performance': {
            'rmse': best_rmse,
            'r2': best_r2
        }
    }
    
    with open('../model_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"✅ Model info saved to ../model_info.json")
    
    # 7. Test the saved model
    print("\n7. Testing saved model...")
    loaded_model = joblib.load(model_filename)
    loaded_scaler = joblib.load(scaler_filename)
    
    print("✅ Model and scaler loaded successfully!")
    
    # Test with a sample input
    sample_input = np.array([[100, 4.5, 50, 200, 2]])  # rating_count, rating_rate, title_length, description_length, category_encoded
    sample_input_scaled = loaded_scaler.transform(sample_input)
    
    if hasattr(loaded_model, 'predict'):
        prediction = loaded_model.predict(sample_input_scaled)
        print(f"\nSample prediction:")
        print(f"Input features: {sample_input[0]}")
        print(f"Predicted price: ${prediction[0]:.2f}")
    else:
        print("\nModel doesn't have predict method!")
    
    print(f"\n🎉 Model training completed!")
    print(f"Best model: {best_model_name}")
    print(f"Model file: {model_filename}")
    print(f"Scaler file: {scaler_filename}")

if __name__ == "__main__":
    main()
