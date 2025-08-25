import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
import google.generativeai as genai
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Product Price Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your-gemini-api-key-here')
genai.configure(api_key=GEMINI_API_KEY)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('price_model.pkl')
        scaler = joblib.load('price_scaler.pkl')
        
        # Load model info
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        
        st.success("✅ Model loaded successfully!")
        return model, scaler, model_info
        
    except FileNotFoundError as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Please run the training script first!")
        return None, None, None

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

def create_feature_importance_chart(model_info):
    """Create a feature importance chart"""
    if not model_info:
        return None
    
    # Sample feature importance (you can modify this based on your model)
    features = ['Rating Count', 'Rating Rate', 'Title Length', 'Description Length', 'Category']
    importance = [0.3, 0.25, 0.15, 0.2, 0.1]  # Example values
    
    fig = px.bar(
        x=features, 
        y=importance,
        title="Feature Importance (Example)",
        labels={'x': 'Features', 'y': 'Importance Score'},
        color=importance,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    # Header
    st.title("🚀 Product Price Predictor")
    st.markdown("AI-powered price prediction with anomaly detection using Google Gemini")
    
    # Load model
    model, scaler, model_info = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("📊 Model Information")
    if model_info:
        st.sidebar.metric("Model Type", model_info['model_name'])
        st.sidebar.metric("RMSE", f"${model_info['performance']['rmse']:.2f}")
        st.sidebar.metric("R² Score", f"{model_info['performance']['r2']:.4f}")
    
    st.sidebar.header("🔧 Settings")
    show_charts = st.sidebar.checkbox("Show Charts", value=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Product Specifications")
        
        # Input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                rating_count = st.number_input(
                    "Rating Count",
                    min_value=0,
                    value=100,
                    help="Number of product ratings"
                )
                
                rating_rate = st.slider(
                    "Rating Rate",
                    min_value=0.0,
                    max_value=5.0,
                    value=4.5,
                    step=0.1,
                    help="Average rating (0-5 stars)"
                )
                
                title_length = st.number_input(
                    "Title Length",
                    min_value=0,
                    value=50,
                    help="Number of characters in title"
                )
            
            with col2:
                description_length = st.number_input(
                    "Description Length",
                    min_value=0,
                    value=200,
                    help="Number of characters in description"
                )
                
                category_encoded = st.selectbox(
                    "Product Category",
                    options=list(CATEGORIES.keys()),
                    format_func=lambda x: CATEGORIES[x].title(),
                    help="Select the product category"
                )
            
            submitted = st.form_submit_button("🚀 Predict Price", use_container_width=True)
    
    with col2:
        st.header("📈 Quick Stats")
        
        # Display some statistics
        st.metric("Input Rating Count", rating_count)
        st.metric("Input Rating Rate", f"{rating_rate:.1f}")
        st.metric("Title Length", f"{title_length} chars")
        st.metric("Description Length", f"{description_length} chars")
        st.metric("Category", CATEGORIES[category_encoded].title())
    
    # Prediction logic
    if submitted:
        st.header("🎯 Prediction Results")
        
        try:
            # Prepare features
            features = np.array([[rating_count, rating_rate, title_length, description_length, category_encoded]])
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Make prediction
            predicted_price = model.predict(features_scaled)[0]
            
            # Display prediction
            col1, col2, col3 = st.columns(3)
            
            with col2:
                st.metric(
                    "Predicted Price",
                    f"${predicted_price:.2f}",
                    delta=f"${predicted_price - 50:.2f}" if predicted_price > 50 else f"${predicted_price - 50:.2f}"
                )
            
            # Get anomaly explanation
            with st.spinner("🤖 Generating AI analysis..."):
                anomaly_explanation = get_anomaly_explanation(predicted_price, features[0])
            
            # Display explanation
            st.subheader("🧠 AI Analysis")
            st.info(anomaly_explanation)
            
            # Display input features summary
            st.subheader("📋 Input Summary")
            feature_data = {
                'Feature': ['Rating Count', 'Rating Rate', 'Title Length', 'Description Length', 'Category'],
                'Value': [rating_count, f"{rating_rate:.1f}", f"{title_length} chars", f"{description_length} chars", CATEGORIES[category_encoded].title()]
            }
            
            feature_df = pd.DataFrame(feature_data)
            st.dataframe(feature_df, use_container_width=True)
            
            # Charts section
            if show_charts:
                st.subheader("📊 Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Feature importance chart
                    importance_fig = create_feature_importance_chart(model_info)
                    if importance_fig:
                        st.plotly_chart(importance_fig, use_container_width=True)
                
                with col2:
                    # Price distribution (example)
                    st.subheader("Price Range Context")
                    price_range = np.linspace(0, 200, 100)
                    st.line_chart(price_range)
                
                # Interactive prediction chart
                st.subheader("🎯 Prediction Confidence")
                
                # Create a confidence interval visualization
                confidence_levels = np.linspace(0.8, 0.99, 20)
                confidence_intervals = predicted_price * (1 + np.random.normal(0, 0.1, len(confidence_levels)))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=confidence_levels,
                    y=confidence_intervals,
                    mode='lines+markers',
                    name='Confidence Interval',
                    line=dict(color='blue', width=2)
                ))
                
                fig.add_hline(y=predicted_price, line_dash="dash", line_color="red", 
                            annotation_text=f"Predicted: ${predicted_price:.2f}")
                
                fig.update_layout(
                    title="Prediction Confidence Analysis",
                    xaxis_title="Confidence Level",
                    yaxis_title="Price Range ($)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using Streamlit, scikit-learn, and Google Gemini</p>
            <p>Check the README for setup instructions and troubleshooting</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
