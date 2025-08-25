# 🚀 Product Price Predictor

An AI-powered product price prediction system that combines machine learning regression with Google Gemini API for intelligent anomaly detection and explanation.

## ✨ Features

- **Data Collection**: Automated scraping from FakeStoreAPI
- **ML Pipeline**: Comprehensive model training with multiple algorithms
- **Price Prediction**: Real-time price estimation based on product features
- **AI Analysis**: Gemini-powered anomaly detection and explanation
- **Beautiful UI**: Modern, responsive web interface
- **Model Persistence**: Save and load trained models

## 🏗️ Architecture

```
product-price-predictor/
│── data/                    # Data collection and storage
│   ├── collect_data.py     # API scraping script
│   └── products.csv        # Collected dataset
│── notebooks/              # Jupyter notebooks
│   ├── train_model.ipynb  # Model training notebook
│   └── train_model.py     # Python script version
│── templates/              # Flask templates
│   └── index.html         # Main web interface
│── app.py                 # Flask web application
│── price_model.pkl        # Trained ML model
│── price_scaler.pkl       # Feature scaler
│── model_info.json        # Model metadata
│── requirements.txt        # Python dependencies
│── README.md              # This file
```

## 🛠️ Tech Stack

- **Python 3.10+**
- **Machine Learning**: scikit-learn, pandas, numpy
- **Web Framework**: Flask
- **AI Integration**: Google Gemini API
- **Data Visualization**: matplotlib, seaborn
- **Model Persistence**: joblib
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript

## 📋 Prerequisites

1. **Python 3.10 or higher**
2. **Google Gemini API Key** ([Get it here](https://makersuite.google.com/app/apikey))
3. **Git** (for cloning the repository)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd product-price-predictor

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
# Windows
set GEMINI_API_KEY=your_actual_api_key_here

# macOS/Linux
export GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Collect Data

```bash
# Run data collection script
python data/collect_data.py
```

This will:
- Fetch product data from FakeStoreAPI
- Process and clean the data
- Save to `data/products.csv`

### 4. Train the Model

#### Option A: Run Python Script
```bash
python notebooks/train_model.py
```

#### Option B: Use Jupyter Notebook
```bash
# Start Jupyter
jupyter notebook notebooks/train_model.ipynb
```

The training will:
- Load and explore the dataset
- Train multiple regression models
- Compare performance metrics
- Save the best model as `price_model.pkl`
- Save the scaler as `price_scaler.pkl`

### 5. Start the Web Application

```bash
python app.py
```

Open your browser and navigate to: `http://localhost:5000`

## 📊 Model Training Details

### Features Used
- **rating_count**: Number of product ratings
- **rating_rate**: Average rating (0-5 stars)
- **title_length**: Number of characters in product title
- **description_length**: Number of characters in description
- **category_encoded**: Encoded product category

### Algorithms Tested
1. **Linear Regression**
2. **Ridge Regression**
3. **Lasso Regression**
4. **Random Forest**
5. **Gradient Boosting**

### Model Selection
The system automatically selects the best model based on **RMSE (Root Mean Square Error)** and provides comprehensive performance metrics.

## 🌐 Web Application Usage

### 1. Input Product Specifications
- **Rating Count**: Number of ratings the product has received
- **Rating Rate**: Average rating (0-5 stars)
- **Title Length**: Number of characters in the product title
- **Description Length**: Number of characters in the description
- **Category**: Product category (men's clothing, electronics, etc.)

### 2. Get Predictions
- **Price Prediction**: ML model output in USD
- **AI Analysis**: Gemini-powered explanation of the prediction
- **Anomaly Detection**: Identifies unusual patterns or pricing

### 3. Features
- Real-time predictions
- Beautiful, responsive UI
- Input validation
- Loading indicators
- Error handling
- Model performance display

## 🔧 API Endpoints

### Health Check
```
GET /api/health
```
Returns application status and model loading status.

### Model Information
```
GET /api/model-info
```
Returns trained model details and performance metrics.

### Price Prediction
```
POST /predict
Content-Type: application/json

{
    "rating_count": 100,
    "rating_rate": 4.5,
    "title_length": 50,
    "description_length": 200,
    "category_encoded": 2
}
```

## 📈 Model Performance

The system provides comprehensive model evaluation:
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error
- **R² Score**: Coefficient of determination (higher is better)
- **Feature Importance**: Understanding of feature contributions

## 🚨 Troubleshooting

### Common Issues

1. **Model Not Found Error**
   ```
   ❌ Error loading model: [Errno 2] No such file or directory
   ```
   **Solution**: Run the training script first to generate the model files.

2. **Gemini API Error**
   ```
   Unable to generate anomaly explanation: API key not valid
   ```
   **Solution**: Set the correct `GEMINI_API_KEY` environment variable.

3. **Data Collection Failed**
   ```
   Failed to fetch data. Status code: 429
   ```
   **Solution**: Wait a few minutes and try again (rate limiting).

4. **Port Already in Use**
   ```
   Address already in use
   ```
   **Solution**: Change the port in `app.py` or kill the existing process.

### Performance Tips

- Use a virtual environment to avoid dependency conflicts
- Ensure sufficient RAM for model training (recommended: 4GB+)
- For production, consider using Gunicorn or uWSGI instead of Flask's development server

## 🔮 Future Enhancements

- [ ] Support for more product APIs
- [ ] Advanced feature engineering
- [ ] Model versioning and A/B testing
- [ ] Real-time model retraining
- [ ] Docker containerization
- [ ] Cloud deployment scripts
- [ ] Additional ML algorithms
- [ ] Batch prediction API

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FakeStoreAPI** for providing product data
- **Google Gemini** for AI-powered analysis
- **scikit-learn** team for the excellent ML library
- **Flask** community for the web framework

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the error logs in the console
3. Open an issue on GitHub
4. Check the model training output for insights

---

**Happy Predicting! 🎯✨**
