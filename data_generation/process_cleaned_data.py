"""
Process Cleaned Data Script - Handles files with and without brand columns
Extracts brands using LLM when needed, adds UUIDs, and removes duplicates
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
CLEANED_DATA_PATH = Path(__file__).parent.parent.parent / "Raw_product_data" / "cleaned"
OUTPUT_PATH = Path(__file__).parent.parent / "source_product_input" / "merged_product_data.csv"

print("=" * 80)
print("🚀 PROCESSING CLEANED DATA FILES")
print("=" * 80)
print(f"📁 Input Path: {CLEANED_DATA_PATH}")
print(f"💾 Output Path: {OUTPUT_PATH}")

def extract_brand_with_llm(product_name):
    """Use LLM to extract brand from product name"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Extract ONLY the brand name from the product name. Common brands: HP, Dell, Lenovo, Acer, ASUS, Apple, Samsung, OnePlus, Xiaomi, Realme, Oppo, Vivo, Motorola, Boat, Noise, Sony, JBL, etc. Return just the brand name, nothing else. If no brand found, return 'Unknown'."
                },
                {
                    "role": "user",
                    "content": f"Extract brand from: {product_name}"
                }
            ],
            temperature=0,
            max_tokens=10
        )
        brand = response.choices[0].message.content.strip()
        return brand
    except Exception as e:
        print(f"      ⚠️ LLM Error: {e}")
        return "Unknown"

def categorize_product(product_name):
    """Use LLM to categorize product"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Categorize the product into EXACTLY ONE of these categories: Laptop, Phone, Tablet, Watch, Earbuds, Dress, Clothing, Electronics, Accessories, Other. Return ONLY the category name."
                },
                {
                    "role": "user",
                    "content": f"Categorize: {product_name}"
                }
            ],
            temperature=0,
            max_tokens=10
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Other"

def clean_price(price_str):
    """Clean and standardize price format"""
    if pd.isna(price_str) or str(price_str).strip() == "":
        return None
    cleaned = str(price_str).replace('₹', '').replace(',', '').strip()
    match = re.search(r'[\d,]+\.?\d*', cleaned)
    if match:
        return f"₹ {match.group()}"
    return None

def clean_discount(discount_str):
    """Clean and standardize discount format"""
    if pd.isna(discount_str) or str(discount_str).strip() == "":
        return "0% off"
    discount_str = str(discount_str).strip()
    if "off" in discount_str.lower():
        return discount_str
    match = re.search(r'(\d+)', discount_str)
    if match:
        return f"{match.group()}% off"
    return "0% off"

def clean_rating(rating_str):
    """Clean and standardize rating format"""
    if pd.isna(rating_str) or str(rating_str).strip() == "":
        return "0.0"
    match = re.search(r'(\d+\.?\d*)', str(rating_str))
    if match:
        rating = float(match.group())
        if 0 <= rating <= 5:
            return str(rating)
    return "0.0"

def clean_bought_count(bought_str):
    """Extract numeric value from bought count string"""
    if pd.isna(bought_str) or str(bought_str).strip() == "":
        return "0"
    
    bought_str = str(bought_str).strip()
    
    # Remove commas first
    bought_str = bought_str.replace(',', '')
    
    # Extract first number found (handle cases like "500+ bought", "2508 Ratings", etc.)
    match = re.search(r'(\d+)', bought_str)
    if match:
        return match.group(1)
    
    return "0"

def process_csv_file(filepath):
    """Process a single CSV file"""
    print(f"\n📁 Processing: {filepath.name}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"   📊 Original rows: {len(df)}")
        print(f"   📋 Columns: {list(df.columns)}")
        
        # Check if required columns exist
        if 'Product' not in df.columns:
            print(f"   ❌ Missing 'Product' column, skipping file")
            return pd.DataFrame()
        
        # Standardize column names (case-insensitive)
        column_mapping = {
            'product': 'Product',
            'brand': 'Brands',
            'brands': 'Brands',
            'selling price': 'Selling Price',
            'price': 'Selling Price',
            'mrp price': 'MRP Price',
            'mrp': 'MRP Price',
            'discount': 'Discount',
            'rating': 'Rating',
            'bought count': 'Bought Count',
            'delivery date': 'Delivery Date',
            'delivery': 'Delivery Date',
            'product image url': 'Product Image URL',
            'image': 'Product Image URL',
            'image_url': 'Product Image URL'
        }
        
        # Rename columns (case-insensitive)
        df.columns = df.columns.str.strip()
        df = df.rename(columns={col: column_mapping.get(col.lower(), col) for col in df.columns})
        
        # Detect if this is a Flipkart file
        is_flipkart = 'flipkart' in filepath.name.lower() or 'flipcart' in filepath.name.lower()
        
        # Create standardized dataframe
        processed_df = pd.DataFrame()
        
        # Add Product ID
        processed_df['Product ID'] = [str(uuid.uuid4()) for _ in range(len(df))]
        
        # Product name
        processed_df['Product'] = df['Product']
        
        # Handle Brands column - Always extract from Product column using LLM
        print(f"   🤖 Extracting brands from Product column with LLM...")
        brands = []
        for idx, product in enumerate(df['Product']):
            if (idx + 1) % 10 == 0:
                print(f"      Progress: {idx + 1}/{len(df)}")
            brand = extract_brand_with_llm(str(product))
            brands.append(brand)
        processed_df['Brands'] = brands
        print(f"   ✅ Extracted {len(brands)} brands")
        
        # Other columns with cleaning
        processed_df['Selling Price'] = df['Selling Price'].apply(clean_price) if 'Selling Price' in df.columns else None
        processed_df['MRP Price'] = df['MRP Price'].apply(clean_price) if 'MRP Price' in df.columns else None
        processed_df['Discount'] = df['Discount'].apply(clean_discount) if 'Discount' in df.columns else '0% off'
        processed_df['Rating'] = df['Rating'].apply(clean_rating) if 'Rating' in df.columns else '0.0'
        processed_df['Bought Count'] = df['Bought Count'].apply(clean_bought_count) if 'Bought Count' in df.columns else '0'
        processed_df['Delivery Date'] = df['Delivery Date'] if 'Delivery Date' in df.columns else 'Not specified'
        processed_df['Product Image URL'] = df['Product Image URL'] if 'Product Image URL' in df.columns else ''
        
        # Detect provider from filename
        filename_lower = filepath.name.lower()
        if 'amazon' in filename_lower:
            provider = 'Amazon'
        elif 'flipkart' in filename_lower or 'flipcart' in filename_lower:
            provider = 'Flipkart'
        else:
            provider = 'Unknown'
        processed_df['Provider'] = provider
        
        # Remove rows with missing critical data
        before_clean = len(processed_df)
        processed_df = processed_df[
            (processed_df['Product'].notna()) & 
            (processed_df['Product'] != '') &
            (processed_df['Selling Price'].notna()) &
            (processed_df['Product Image URL'].notna()) &
            (processed_df['Product Image URL'] != '')
        ]
        removed = before_clean - len(processed_df)
        if removed > 0:
            print(f"   🧹 Removed {removed} incomplete records")
        print(f"   ✅ Final rows: {len(processed_df)}")
        
        return processed_df
        
    except Exception as e:
        print(f"   ❌ Error processing {filepath.name}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def main():
    """Main function"""
    
    # Check if cleaned folder exists
    if not CLEANED_DATA_PATH.exists():
        print(f"\n❌ Cleaned data folder not found: {CLEANED_DATA_PATH}")
        print(f"💡 Please ensure your cleaned CSV files are in: {CLEANED_DATA_PATH}")
        return
    
    # Get all CSV files
    csv_files = list(CLEANED_DATA_PATH.glob("*.csv"))
    
    if not csv_files:
        print(f"\n❌ No CSV files found in: {CLEANED_DATA_PATH}")
        return
    
    print(f"\n📊 Found {len(csv_files)} CSV files")
    for file in csv_files:
        print(f"  - {file.name}")
    
    # Process all files
    print("\n" + "=" * 80)
    print("📂 PROCESSING FILES")
    print("=" * 80)
    
    all_data = []
    for csv_file in csv_files:
        df = process_csv_file(csv_file)
        if not df.empty:
            all_data.append(df)
    
    if not all_data:
        print("\n❌ No data processed successfully!")
        return
    
    # Merge all data
    print("\n" + "=" * 80)
    print("🔄 MERGING DATA")
    print("=" * 80)
    
    merged_df = pd.concat(all_data, ignore_index=True)
    print(f"📊 Total rows before deduplication: {len(merged_df)}")
    
    # Remove duplicates based on Product name
    print("\n🔍 Removing duplicates...")
    before_dedup = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['Product'], keep='first')
    removed = before_dedup - len(merged_df)
    print(f"✅ Removed {removed} duplicate products")
    print(f"📊 Final count: {len(merged_df)} unique products")
    
    # Add Category using LLM
    print("\n" + "=" * 80)
    print("🤖 CATEGORIZING PRODUCTS")
    print("=" * 80)
    
    categories = []
    for idx, product in enumerate(merged_df['Product']):
        if (idx + 1) % 10 == 0:
            print(f"   Progress: {idx + 1}/{len(merged_df)}")
        category = categorize_product(str(product))
        categories.append(category)
    
    merged_df['Category'] = categories
    print(f"✅ Categorized {len(categories)} products")
    
    # Data quality summary
    print("\n" + "=" * 80)
    print("✅ DATA QUALITY SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Total products: {len(merged_df)}")
    print(f"📊 Unique products: {merged_df['Product'].nunique()}")
    print(f"📊 Unique brands: {merged_df['Brands'].nunique()}")
    print(f"📊 Products with images: {(merged_df['Product Image URL'] != '').sum()}")
    print(f"📊 Products with ratings: {(merged_df['Rating'] != '0.0').sum()}")
    print(f"📊 Products with discounts: {(merged_df['Discount'] != '0% off').sum()}")
    
    print("\n📋 Provider Distribution:")
    print(merged_df['Provider'].value_counts().to_string())
    
    print("\n📋 Category Distribution:")
    print(merged_df['Category'].value_counts().to_string())
    
    print("\n📋 Top 10 Brands:")
    print(merged_df['Brands'].value_counts().head(10).to_string())
    
    # Reorder columns
    column_order = [
        'Product ID', 'Product', 'Category', 'Brands', 'Provider', 
        'Selling Price', 'MRP Price', 'Discount', 'Rating', 
        'Bought Count', 'Delivery Date', 'Product Image URL'
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
    print(merged_df.head(3)[['Product', 'Brands', 'Category', 'Selling Price', 'Provider']].to_string())
    
    print("\n" + "=" * 80)
    print("✅ PROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n💡 Next steps:")
    print(f"   1. Review the output file: {OUTPUT_PATH}")
    print(f"   2. Run rebuild_db.py to update ChromaDB")
    print(f"   3. Restart Streamlit app to see new data")

if __name__ == "__main__":
    main()
