import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from datetime import datetime
import json

app = Flask(__name__)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your-gemini-api-key-here')
genai.configure(api_key=GEMINI_API_KEY)

# Load the trained model and scaler
try:
    model = joblib.load('price_model.pkl')
    scaler = joblib.load('price_scaler.pkl')
    
    # Load model info
    with open('model_info.json', 'r') as f:
        model_info = json.load(f)
    
    print("✅ Model loaded successfully!")
    print(f"Model: {model_info['model_name']}")
    print(f"Performance: RMSE=${model_info['performance']['rmse']:.2f}, R²={model_info['performance']['r2']:.4f}")
    
except FileNotFoundError as e:
    print(f"❌ Error loading model: {e}")
    print("Please run the training script first!")
    model = None
    scaler = None
    model_info = None

# Category mapping for the UI
CATEGORIES = {
    0: "men's clothing",
    1: "jewelery", 
    2: "electronics",
    3: "women's clothing"
}

def get_anomaly_explanation(predicted_price, features):
    """
    Get anomaly explanation from Gemini API
    """
    try:
        # Create a prompt for Gemini
        prompt = f"""
        As an AI expert in product pricing analysis, analyze this price prediction:

        Predicted Price: ${predicted_price:.2f}

        Product Features:
        - Rating Count: {features[0]}
        - Rating Rate: {features[1]}
        - Title Length: {features[2]} characters
        - Description Length: {features[3]} characters
        - Category: {CATEGORIES.get(features[4], 'Unknown')}

        Please provide a brief analysis (2-3 sentences) explaining:
        1. Whether this price prediction seems reasonable given the features
        2. Any potential anomalies or unusual patterns
        3. What factors might be influencing this price

        Keep the explanation clear and concise for a general audience.
        """
        
        # Generate response using Gemini
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        return response.text.strip()
        
    except Exception as e:
        return f"Unable to generate anomaly explanation: {str(e)}"

@app.route('/')
def index():
    """Main page with the prediction form"""
    return render_template('index.html', categories=CATEGORIES, model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for price prediction"""
    try:
        # Get input data
        data = request.get_json()
        
        rating_count = int(data.get('rating_count', 0))
        rating_rate = float(data.get('rating_rate', 0.0))
        title_length = int(data.get('title_length', 0))
        description_length = int(data.get('description_length', 0))
        category_encoded = int(data.get('category_encoded', 0))
        
        # Validate inputs
        if rating_count < 0 or rating_rate < 0 or rating_rate > 5:
            return jsonify({'error': 'Invalid input values'}), 400
        
        # Prepare features
        features = np.array([[rating_count, rating_rate, title_length, description_length, category_encoded]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        predicted_price = model.predict(features_scaled)[0]
        
        # Get anomaly explanation
        anomaly_explanation = get_anomaly_explanation(predicted_price, features[0])
        
        # Prepare response
        response = {
            'predicted_price': round(predicted_price, 2),
            'anomaly_explanation': anomaly_explanation,
            'input_features': {
                'rating_count': rating_count,
                'rating_rate': rating_rate,
                'title_length': title_length,
                'description_length': description_length,
                'category': CATEGORIES.get(category_encoded, 'Unknown')
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model-info')
def model_info_endpoint():
    """Get model information"""
    if model_info:
        return jsonify(model_info)
    else:
        return jsonify({'error': 'Model not loaded'}), 404

if __name__ == '__main__':
    if model is None or scaler is None:
        print("❌ Cannot start app: Model not loaded!")
        print("Please run the training script first.")
    else:
        print("🚀 Starting Flask app...")
        print("📱 Open http://localhost:5000 in your browser")
        app.run(debug=True, host='0.0.0.0', port=5000)
