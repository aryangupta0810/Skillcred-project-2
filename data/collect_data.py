import requests
import pandas as pd
import json
from datetime import datetime

def fetch_products():
    """
    Fetch product data from FakeStoreAPI
    """
    print("Fetching product data from FakeStoreAPI...")
    
    # Fetch all products
    response = requests.get('https://fakestoreapi.com/products')
    
    if response.status_code == 200:
        products = response.json()
        print(f"Successfully fetched {len(products)} products")
        return products
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None

def process_products(products):
    """
    Process and clean the product data
    """
    if not products:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(products)
    
    # Clean and process the data
    df['rating_count'] = df['rating'].apply(lambda x: x.get('count', 0) if isinstance(x, dict) else 0)
    df['rating_rate'] = df['rating'].apply(lambda x: x.get('rate', 0) if isinstance(x, dict) else 0)
    
    # Drop the original rating column and other unnecessary columns
    df = df.drop(['rating', 'id'], axis=1)
    
    # Convert price to numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Fill missing values
    df['rating_count'] = df['rating_count'].fillna(0)
    df['rating_rate'] = df['rating_rate'].fillna(0)
    df['price'] = df['price'].fillna(df['price'].mean())
    
    # Create some additional features
    df['title_length'] = df['title'].str.len()
    df['description_length'] = df['description'].str.len()
    
    # Encode categories
    df['category_encoded'] = pd.Categorical(df['category']).codes
    
    print(f"Processed {len(df)} products")
    print(f"Features: {list(df.columns)}")
    
    return df

def save_data(df, filename='products.csv'):
    """
    Save the processed data to CSV
    """
    if df is not None:
        filepath = f"data/{filename}"
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        
        # Display basic statistics
        print("\nDataset Summary:")
        print(f"Total products: {len(df)}")
        print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        print(f"Average price: ${df['price'].mean():.2f}")
        print(f"Categories: {df['category'].nunique()}")
        print(f"Rating count range: {df['rating_count'].min()} - {df['rating_count'].max()}")
        
        return filepath
    return None

def main():
    """
    Main function to collect and process data
    """
    print("=== Product Data Collection ===")
    print(f"Started at: {datetime.now()}")
    
    # Fetch data
    products = fetch_products()
    
    if products:
        # Process data
        df = process_products(products)
        
        # Save data
        if df is not None:
            save_data(df)
            print("\n✅ Data collection completed successfully!")
        else:
            print("❌ Failed to process data")
    else:
        print("❌ Failed to fetch data")

if __name__ == "__main__":
    main()
