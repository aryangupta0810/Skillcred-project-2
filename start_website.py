#!/usr/bin/env python3
"""
Startup script for the Product Price Predictor Website
This script will install Flask and start the website
"""

import subprocess
import sys
import os

def install_flask():
    """Install Flask using pip"""
    print("🔧 Installing Flask...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
        print("✅ Flask installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Flask")
        return False

def start_website():
    """Start the Flask website"""
    print("🚀 Starting Product Price Predictor Website...")
    try:
        # Import and run the Flask app
        from simple_app import app
        print("📱 Website is starting...")
        print("🌐 Open your browser and go to: http://localhost:5000")
        print("✨ Press Ctrl+C to stop the website")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except ImportError as e:
        print(f"❌ Error importing Flask app: {e}")
        return False
    except Exception as e:
        print(f"❌ Error starting website: {e}")
        return False

def main():
    print("🎉 Welcome to Product Price Predictor!")
    print("=" * 50)
    
    # Check if Flask is already installed
    try:
        import flask
        print("✅ Flask is already installed!")
    except ImportError:
        print("📦 Flask not found, installing...")
        if not install_flask():
            print("❌ Cannot continue without Flask")
            return
    
    # Start the website
    print("\n🎯 Starting the website...")
    start_website()

if __name__ == "__main__":
    main()
