from flask import Flask, render_template, request, jsonify
import random
import json
from datetime import datetime

app = Flask(__name__)

# Simple mock model for demonstration
class SimplePricePredictor:
    def __init__(self):
        self.base_prices = {
            "men's clothing": 1200.0,  # ₹1200 base price for men's clothing
            "jewelery": 3500.0,        # ₹3500 base price for jewelry
            "electronics": 2500.0,     # ₹2500 base price for electronics
            "women's clothing": 1500.0  # ₹1500 base price for women's clothing
        }
    
    def predict_price(self, rating_count, rating_rate, title_length, description_length, category):
        """Simple price prediction logic"""
        base_price = self.base_prices.get(category, 50.0)
        
        # Adjust based on rating
        rating_multiplier = 1.0 + (rating_rate - 3.0) * 0.2
        
        # Adjust based on rating count (more ratings = more confidence)
        confidence_multiplier = 1.0 + min(rating_count / 1000, 0.3)
        
        # Adjust based on content length
        content_multiplier = 1.0 + (title_length + description_length) / 1000
        
        predicted_price = base_price * rating_multiplier * confidence_multiplier * content_multiplier
        
        # Add some randomness to make it realistic
        predicted_price *= random.uniform(0.8, 1.2)
        
        # Convert to Indian Rupees (multiply by ~83 for USD to INR conversion)
        predicted_price_inr = predicted_price * 83
        
        return round(max(predicted_price_inr, 400.0), 2)
    
    def get_analysis(self, predicted_price, rating_count, rating_rate, title_length, description_length, category):
        """Generate intelligent analysis"""
        analysis = []
        
        # Price analysis (in Indian Rupees)
        if predicted_price > 15000:
            analysis.append("This is a premium-priced product, likely due to high quality or brand value.")
        elif predicted_price < 1000:
            analysis.append("This appears to be a budget-friendly option with competitive pricing.")
        else:
            analysis.append("This falls within the mid-range pricing category, offering good value.")
        
        # Rating analysis
        if rating_count > 500:
            analysis.append("High number of ratings suggests strong market presence and customer trust.")
        elif rating_count < 50:
            analysis.append("Limited ratings indicate this might be a newer or niche product.")
        
        if rating_rate > 4.5:
            analysis.append("Excellent rating suggests high customer satisfaction and quality.")
        elif rating_rate < 3.5:
            analysis.append("Lower rating may indicate quality concerns or customer service issues.")
        
        # Category insights
        if category == "electronics":
            analysis.append("Electronics typically have higher price volatility based on technology trends.")
        elif category in ["men's clothing", "women's clothing"]:
            analysis.append("Clothing prices often reflect brand positioning and material quality.")
        elif category == "jewelery":
            analysis.append("Jewelry pricing heavily depends on materials, craftsmanship, and brand prestige.")
        
        return " ".join(analysis)

# Initialize the predictor
predictor = SimplePricePredictor()

@app.route('/')
def index():
    """Main page with the prediction form"""
    return render_template('simple_index.html')

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
        category = data.get('category', 'electronics')
        
        # Validate inputs
        if rating_count < 0 or rating_rate < 0 or rating_rate > 5:
            return jsonify({'error': 'Invalid input values'}), 400
        
        # Make prediction
        predicted_price = predictor.predict_price(
            rating_count, rating_rate, title_length, description_length, category
        )
        
        # Get analysis
        analysis = predictor.get_analysis(
            predicted_price, rating_count, rating_rate, title_length, description_length, category
        )
        
        # Prepare response
        response = {
            'predicted_price': predicted_price,
            'analysis': analysis,
            'input_features': {
                'rating_count': rating_count,
                'rating_rate': rating_rate,
                'title_length': title_length,
                'description_length': description_length,
                'category': category
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
        'model_loaded': True,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/sample-data')
def sample_data():
    """Get sample product data for demonstration"""
    sample_products = [
        {
            'name': 'Premium Wireless Headphones',
            'category': 'electronics',
            'rating_count': 1250,
            'rating_rate': 4.7,
            'title_length': 65,
            'description_length': 320,
            'actual_price': 7499.00
        },
        {
            'name': 'Designer Denim Jacket',
            'category': "women's clothing",
            'rating_count': 890,
            'rating_rate': 4.5,
            'title_length': 45,
            'description_length': 280,
            'actual_price': 5599.00
        },
        {
            'name': 'Sterling Silver Necklace',
            'category': 'jewelery',
            'rating_count': 456,
            'rating_rate': 4.8,
            'title_length': 38,
            'description_length': 195,
            'actual_price': 12000.00
        }
    ]
    return jsonify(sample_products)

if __name__ == '__main__':
    print("🚀 Starting Simple Product Price Predictor...")
    print("📱 Open http://localhost:5000 in your browser")
    print("✨ This version works without external dependencies!")
    print("🇮🇳 Prices are now in Indian Rupees (₹)!")
    app.run(debug=True, host='0.0.0.0', port=5000)
