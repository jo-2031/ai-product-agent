"""
Clean Data Creation Script - Properly handles Amazon and Flipkart data sources
"""
import pandas as pd
import os
from pathlib import Path
import uuid
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths
RAW_DATA_PATH = Path(__file__).parent.parent.parent / "Raw_product_data"
OUTPUT_PATH = Path(__file__).parent.parent / "source_product_input" / "merged_product_data.csv"

print("=" * 80)
print("🚀 STARTING CLEAN DATA CREATION PROCESS")
print("=" * 80)

def clean_price(price_str):
    """Clean and standardize price format"""
    if pd.isna(price_str) or str(price_str).strip() == "":
        return None
    
    # Remove currency symbols and extra spaces
    cleaned = str(price_str).replace('₹', '').replace(',', '').strip()
    
    # Extract numeric value
    match = re.search(r'[\d,]+\.?\d*', cleaned)
    if match:
        return f"₹ {match.group()}"
    return None

def clean_discount(discount_str):
    """Clean and standardize discount format"""
    if pd.isna(discount_str) or str(discount_str).strip() == "":
        return "0% off"
    
    discount_str = str(discount_str).strip()
    
    # If already has "off", return as is
    if "off" in discount_str.lower():
        return discount_str
    
    # Extract percentage
    match = re.search(r'(\d+)', discount_str)
    if match:
        return f"{match.group()}% off"
    
    return "0% off"

def clean_rating(rating_str):
    """Clean and standardize rating format"""
    if pd.isna(rating_str) or str(rating_str).strip() == "":
        return "0.0"
    
    # Extract numeric value
    match = re.search(r'(\d+\.?\d*)', str(rating_str))
    if match:
        rating = float(match.group())
        # Ensure rating is between 0 and 5
        if 0 <= rating <= 5:
            return str(rating)
    
    return "0.0"

def categorize_product(product_name):
    """Use LLM to categorize product into predefined categories"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a product categorization expert. Categorize the product into EXACTLY ONE of these categories: Laptop, Phone, Tablet, Watch, Earbuds, Dress, Clothing, Electronics, Accessories, Other. Return ONLY the category name, nothing else."
                },
                {
                    "role": "user",
                    "content": f"Categorize this product: {product_name}"
                }
            ],
            temperature=0,
            max_tokens=10
        )
        category = response.choices[0].message.content.strip()
        return category
    except Exception as e:
        print(f"      ❌ Error categorizing: {e}")
        return "Other"

def extract_brand(product_description):
    """Use LLM to extract brand name from product description"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Extract ONLY the brand name from the product description. Return just the brand name, nothing else. Examples: 'OnePlus', 'Apple', 'Samsung', 'HP', 'Boat', 'Noise'. If no brand found, return 'Unknown'."
                },
                {
                    "role": "user",
                    "content": f"Extract brand from: {product_description}"
                }
            ],
            temperature=0,
            max_tokens=10
        )
        brand = response.choices[0].message.content.strip()
        return brand
    except Exception as e:
        return "Unknown"

def create_product_description(full_text):
    """Use LLM to create a clean product description"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Create a concise product name (max 100 characters) from the given text. Include brand, model, and key features. Remove redundant details."
                },
                {
                    "role": "user",
                    "content": f"Create product name from: {full_text}"
                }
            ],
            temperature=0,
            max_tokens=50
        )
        description = response.choices[0].message.content.strip()
        return description
    except Exception as e:
        return full_text[:100]

def validate_columns(df, filepath, required_columns, file_type):
    """Validate that required columns exist in the dataframe"""
    df_columns_lower = [col.lower().strip() for col in df.columns]
    missing_columns = []
    
    for req_col in required_columns:
        # Check if any variant of the required column exists
        col_variants = req_col if isinstance(req_col, list) else [req_col]
        if not any(variant.lower() in df_columns_lower for variant in col_variants):
            missing_columns.append(req_col[0] if isinstance(req_col, list) else req_col)
    
    if missing_columns:
        print(f"   ❌ ERROR: Missing required columns in {filepath.name}")
        print(f"   ❌ File Type: {file_type}")
        print(f"   ❌ Missing Columns: {', '.join(missing_columns)}")
        print(f"   📋 Available Columns: {list(df.columns)}")
        return False
    
    return True

def process_flipkart_file(filepath, provider_name):
    """Process Flipkart CSV file"""
    print(f"\n📁 Processing Flipkart: {filepath.name}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"   📊 Original rows: {len(df)}")
        
        # Check what columns exist
        print(f"   📋 Columns: {list(df.columns)}")
        
        # Validate required columns
        required_columns = [
            ['title', 'product', 'name', 'product_url'],  # Product name
            ['image', 'image_url', 'product image url'],   # Image
            ['selling_price', 'price', 'selling price'],   # Price
        ]
        
        if not validate_columns(df, filepath, required_columns, 'Flipkart'):
            return pd.DataFrame()
        
        # Rename and map columns (case-insensitive)
        df.columns = df.columns.str.lower().str.strip()
        
        column_mapping = {
            'title': 'Product',
            'product_url': 'Product',
            'name': 'Product',
            'brand': 'Brands',
            'selling_price': 'Selling Price',
            'mrp': 'MRP Price',
            'discount': 'Discount',
            'rating': 'Rating',
            'reviews': 'Bought Count',
            'review_count': 'Bought Count',
            'delivery': 'Delivery Date',
            'delivery_time': 'Delivery Date',
            'image': 'Product Image URL',
            'image_url': 'Product Image URL',
            'product image url': 'Product Image URL'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Create standardized dataframe (Flipkart)
        processed_df = pd.DataFrame()
        processed_df['Product ID'] = [str(uuid.uuid4()) for _ in range(len(df))]
        processed_df['Product'] = df['Product'] if 'Product' in df.columns else pd.Series([''] * len(df))
        processed_df['Brands'] = df['Brands'] if 'Brands' in df.columns else pd.Series(['Unknown'] * len(df))
        
        if 'Selling Price' in df.columns:
            processed_df['Selling Price'] = df['Selling Price'].apply(clean_price)
        else:
            processed_df['Selling Price'] = None
            
        if 'MRP Price' in df.columns:
            processed_df['MRP Price'] = df['MRP Price'].apply(clean_price)
        else:
            processed_df['MRP Price'] = None
            
        if 'Discount' in df.columns:
            processed_df['Discount'] = df['Discount'].apply(clean_discount)
        elif 'Disount' in df.columns:
            processed_df['Discount'] = df['Disount'].apply(clean_discount)
        else:
            processed_df['Discount'] = '0% off'
            
        if 'Rating' in df.columns:
            processed_df['Rating'] = df['Rating'].apply(clean_rating)
        else:
            processed_df['Rating'] = '0.0'
        
        processed_df['Bought Count'] = df['Bought Count'] if 'Bought Count' in df.columns else 'Not specified'
        processed_df['Delivery Date'] = df['Delivery Date'] if 'Delivery Date' in df.columns else 'Not specified'
        processed_df['Product Image URL'] = df['Product Image URL'] if 'Product Image URL' in df.columns else ''
        processed_df['Provider'] = provider_name
        
        # Remove rows with missing critical data
        before_clean = len(processed_df)
        processed_df = processed_df[
            (processed_df['Product'].notna()) & 
            (processed_df['Product'] != '') &
            (processed_df['Selling Price'].notna()) &
            (processed_df['Product Image URL'].notna()) &
            (processed_df['Product Image URL'] != '')
        ]
        print(f"   ✅ After cleaning: {len(processed_df)} rows (removed {before_clean - len(processed_df)} incomplete records)")
        
        return processed_df
        
    except Exception as e:
        print(f"   ❌ Error processing {filepath.name}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def process_amazon_file(filepath, provider_name):
    """Process Amazon CSV file"""
    print(f"\n📁 Processing Amazon: {filepath.name}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"   📊 Original rows: {len(df)}")
        
        # Check what columns exist
        print(f"   📋 Columns: {list(df.columns)}")
        
        # Validate required columns
        required_columns = [
            ['title', 'product_name', 'name', 'product'],  # Product name
            ['image', 'image_url', 'product image url'],   # Image
            ['price', 'selling_price', 'selling price'],   # Price
        ]
        
        if not validate_columns(df, filepath, required_columns, 'Amazon'):
            return pd.DataFrame()
        
        # Rename and map columns (case-insensitive)
        df.columns = df.columns.str.lower().str.strip()
        
        column_mapping = {
            'title': 'Product',
            'product_name': 'Product',
            'name': 'Product',
            'brand': 'Brands',
            'price': 'Selling Price',
            'actual_price': 'MRP Price',
            'mrp': 'MRP Price',
            'discount': 'Discount',
            'discount_percentage': 'Discount',
            'rating': 'Rating',
            'ratings_count': 'Bought Count',
            'reviews': 'Bought Count',
            'delivery': 'Delivery Date',
            'image': 'Product Image URL',
            'image_url': 'Product Image URL',
            'product image url': 'Product Image URL'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Create standardized dataframe (Amazon)
        processed_df = pd.DataFrame()
        processed_df['Product ID'] = [str(uuid.uuid4()) for _ in range(len(df))]
        processed_df['Product'] = df['Product'] if 'Product' in df.columns else pd.Series([''] * len(df))
        processed_df['Brands'] = df['Brands'] if 'Brands' in df.columns else pd.Series(['Unknown'] * len(df))
        
        if 'Selling Price' in df.columns:
            processed_df['Selling Price'] = df['Selling Price'].apply(clean_price)
        else:
            processed_df['Selling Price'] = None
            
        if 'MRP Price' in df.columns:
            processed_df['MRP Price'] = df['MRP Price'].apply(clean_price)
        else:
            processed_df['MRP Price'] = None
            
        if 'Discount' in df.columns:
            processed_df['Discount'] = df['Discount'].apply(clean_discount)
        else:
            processed_df['Discount'] = '0% off'
            
        if 'Rating' in df.columns:
            processed_df['Rating'] = df['Rating'].apply(clean_rating)
        else:
            processed_df['Rating'] = '0.0'
        
        processed_df['Bought Count'] = df['Bought Count'] if 'Bought Count' in df.columns else 'Not specified'
        processed_df['Delivery Date'] = df['Delivery Date'] if 'Delivery Date' in df.columns else 'Not specified'
        processed_df['Product Image URL'] = df['Product Image URL'] if 'Product Image URL' in df.columns else ''
        processed_df['Provider'] = provider_name
        
        # Remove rows with missing critical data
        before_clean = len(processed_df)
        processed_df = processed_df[
            (processed_df['Product'].notna()) & 
            (processed_df['Product'] != '') &
            (processed_df['Selling Price'].notna()) &
            (processed_df['Product Image URL'].notna()) &
            (processed_df['Product Image URL'] != '')
        ]
        print(f"   ✅ After cleaning: {len(processed_df)} rows (removed {before_clean - len(processed_df)} incomplete records)")
        
        return processed_df
        
    except Exception as e:
        print(f"   ❌ Error processing {filepath.name}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def main():
    """Main function to process all data"""
    
    all_data = []
    
    # Process Flipkart files
    print("\n" + "=" * 80)
    print("📂 PROCESSING FLIPKART DATA")
    print("=" * 80)
    
    flipkart_path = RAW_DATA_PATH / "flipcart"
    if flipkart_path.exists():
        csv_files = list(flipkart_path.glob("*.csv"))
        print(f"Found {len(csv_files)} Flipkart CSV files")
        for csv_file in csv_files:
            df = process_flipkart_file(csv_file, "Flipkart")
            if not df.empty:
                all_data.append(df)
    else:
        print(f"❌ Flipkart folder not found: {flipkart_path}")
    
    # Process Amazon files
    print("\n" + "=" * 80)
    print("📂 PROCESSING AMAZON DATA")
    print("=" * 80)
    
    amazon_path = RAW_DATA_PATH / "Amazon"
    if amazon_path.exists():
        csv_files = list(amazon_path.glob("*.csv"))
        print(f"Found {len(csv_files)} Amazon CSV files")
        for csv_file in csv_files:
            df = process_amazon_file(csv_file, "Amazon")
            if not df.empty:
                all_data.append(df)
    else:
        print(f"❌ Amazon folder not found: {amazon_path}")
    
    # Merge all data
    print("\n" + "=" * 80)
    print("🔄 MERGING ALL DATA")
    print("=" * 80)
    
    if not all_data:
        print("❌ No data to merge!")
        return
    
    merged_df = pd.concat(all_data, ignore_index=True)
    print(f"📊 Total rows before deduplication: {len(merged_df)}")
    
    # Remove duplicates based on Product name only (keep first occurrence)
    print("\n🔍 Removing duplicate product names...")
    before_dedup = len(merged_df)
    merged_df = merged_df.drop_duplicates(
        subset=['Product'],
        keep='first'
    )
    print(f"✅ Removed {before_dedup - len(merged_df)} duplicate products")
    print(f"📊 Final row count: {len(merged_df)}")
    
    # Add Category column using LLM
    print("\n" + "=" * 80)
    print("🤖 PROCESSING PRODUCTS WITH LLM")
    print("=" * 80)
    print("Extracting brands, creating descriptions, and categorizing...")
    
    clean_products = []
    clean_brands = []
    categories = []
    
    for idx, row in merged_df.iterrows():
        if (idx + 1) % 10 == 0:
            print(f"   Progress: {idx + 1}/{len(merged_df)}")
        
        # Get the full product description from 'Brands' column
        full_description = str(row['Brands'])
        
        # Extract brand name
        brand = extract_brand(full_description)
        
        # Create clean product description
        product_desc = create_product_description(full_description)
        
        # Categorize product
        category = categorize_product(product_desc)
        
        clean_products.append(product_desc)
        clean_brands.append(brand)
        categories.append(category)
        
        if (idx + 1) % 10 == 0:
            print(f"      Example: {product_desc[:60]}... | Brand: {brand} | Category: {category}")
    
    # Update dataframe
    merged_df['Product'] = clean_products
    merged_df['Brands'] = clean_brands
    merged_df['Category'] = categories
    print(f"✅ Processed {len(merged_df)} products")
    
    # Verify data quality
    print("\n" + "=" * 80)
    print("✅ DATA QUALITY CHECKS")
    print("=" * 80)
    
    print(f"✅ Total products: {len(merged_df)}")
    print(f"✅ Products with images: {merged_df['Product Image URL'].notna().sum()}")
    print(f"✅ Products with valid prices: {merged_df['Selling Price'].notna().sum()}")
    print(f"✅ Products with ratings: {(merged_df['Rating'] != '0.0').sum()}")
    print(f"✅ Products with discount: {(merged_df['Discount'] != '0% off').sum()}")
    print(f"✅ Amazon products: {(merged_df['Provider'] == 'Amazon').sum()}")
    print(f"✅ Flipkart products: {(merged_df['Provider'] == 'Flipkart').sum()}")
    
    # Show category distribution
    print("\n📊 Category Distribution:")
    print(merged_df['Category'].value_counts().to_string())
    
    # Reorder columns
    column_order = [
        'Product ID', 'Product', 'Category', 'Provider', 'Selling Price', 'Brands',
        'Discount', 'Rating', 'Bought Count', 'MRP Price',
        'Delivery Date', 'Product Image URL'
    ]
    merged_df = merged_df[column_order]
    
    # Save to CSV
    print("\n" + "=" * 80)
    print("💾 SAVING DATA")
    print("=" * 80)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Data saved to: {OUTPUT_PATH}")
    print(f"📊 Total records: {len(merged_df)}")
    
    # Show sample
    print("\n" + "=" * 80)
    print("📋 SAMPLE DATA (First 3 rows)")
    print("=" * 80)
    print(merged_df.head(3)[['Product', 'Brands', 'Selling Price', 'Rating', 'Provider']].to_string())
    
    print("\n" + "=" * 80)
    print("✅ DATA CREATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
