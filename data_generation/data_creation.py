import pandas as pd
import glob
import uuid
import os
import random
import hashlib
import json

def get_file_hash(file_path):
    """Generate hash of file to detect changes"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_cache():
    """Load file processing cache"""
    cache_file = 'file_cache.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """Save file processing cache"""
    with open('file_cache.json', 'w') as f:
        json.dump(cache, indent=2, fp=f)

base_path = '/Users/jothikaravichandran/Documents/Self_Projects/personal_project/ai_geeks_product_agent/Raw_product_data'
provider_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and not f.startswith('.')]

required_columns = [
    'Product', 'Brands', 'Discount', 'Rating', 
    'Bought Count', 'MRP Price', 'Selling Price', 
    'Delivery Date', 'Product Image URL'
]

dataframes = []
cache = load_cache()
total_files = 0
processed_files = 0
skipped_files = 0
file_record_counts = {}
total_records_from_files = 0

print("Starting data processing...")
print(f"Found {len(provider_folders)} provider folders: {provider_folders}")
print("="*80)

for provider_folder in provider_folders:
    folder_path = os.path.join(base_path, provider_folder)
    files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    print(f"\nProcessing Provider: {provider_folder}")
    print(f"Found {len(files)} CSV files")
    print("-"*80)
    
    for file in files:
        total_files += 1
        file_name = os.path.basename(file)
        file_hash = get_file_hash(file)
        
        if file in cache and cache[file] == file_hash:
            print(f"  [SKIPPED] {file_name} (no changes detected)")
            skipped_files += 1
            continue
        
        print(f"  [PROCESSING] {file_name}")
        
        df = pd.read_csv(file)
        original_rows = len(df)
        file_record_counts[file_name] = original_rows
        total_records_from_files += original_rows
        
        print(f"    - Original rows: {original_rows}")
        print(f"    - Original columns: {list(df.columns)}")
        
        available_cols = [col for col in required_columns if col in df.columns]
        df = df[available_cols]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        df['Provider'] = provider_folder
        dataframes.append(df)
        
        cache[file] = file_hash
        processed_files += 1
        
        print(f"    - Rows added to merge: {len(df)}")
        print(f"    - Missing columns filled: {[col for col in required_columns if col not in available_cols]}")

print("\n" + "="*80)
print(f"File Processing Summary:")
print(f"  Total files found: {total_files}")
print(f"  Files processed: {processed_files}")
print(f"  Files skipped: {skipped_files}")
print(f"  Total records from all files: {total_records_from_files}")
print("="*80)

print("\nRecord count per file:")
for file_name, count in file_record_counts.items():
    print(f"  {file_name}: {count} records")
print(f"\nSum of all file records: {sum(file_record_counts.values())}")
print("="*80)

if dataframes:
    print("\nMerging all dataframes...")
    final_df = pd.concat(dataframes, ignore_index=True)
    merged_count = len(final_df)
    print(f"Total rows after merge: {merged_count}")
    
    print("\nVerifying record counts...")
    if merged_count == total_records_from_files:
        print(f"  VERIFIED: Merged count ({merged_count}) matches sum of all files ({total_records_from_files})")
    else:
        print(f"  WARNING: Merged count ({merged_count}) does NOT match sum of files ({total_records_from_files})")
        print(f"  Difference: {abs(merged_count - total_records_from_files)} records")
    print("="*80)
    
    print("\nFilling missing Brands from Product names...")
    mask = (final_df['Brands'].isna()) | (final_df['Brands'] == '') | (final_df['Brands'] == 'None')
    if mask.any():
        brand_values = (
            final_df.loc[mask, 'Product']
            .fillna('')
            .str.split(r'[,:|]', n=1, expand=True)[0]
            .str.strip()
        )
        final_df.loc[mask, 'Brands'] = brand_values
        print(f"Filled {mask.sum()} missing Brands")
    
    print("\nFilling missing Ratings with random values (3.5-5.0)...")
    rating_mask = (final_df['Rating'].isna()) | (final_df['Rating'] == '') | (final_df['Rating'] == 'None')
    final_df.loc[rating_mask, 'Rating'] = final_df.loc[rating_mask, 'Rating'].apply(
        lambda x: round(random.uniform(3.5, 5.0), 1)
    )
    print(f"Filled {rating_mask.sum()} missing Ratings")
    
    print("\nFilling missing Bought Count with random values...")
    bought_mask = (final_df['Bought Count'].isna()) | (final_df['Bought Count'] == '') | (final_df['Bought Count'] == 'None')
    final_df.loc[bought_mask, 'Bought Count'] = final_df.loc[bought_mask, 'Bought Count'].apply(
        lambda x: f"{random.randint(100, 5000)}+ bought in past month"
    )
    print(f"Filled {bought_mask.sum()} missing Bought Counts")
    
    print("\nFilling missing Discount with random values...")
    discount_mask = (final_df['Discount'].isna()) | (final_df['Discount'] == '') | (final_df['Discount'] == 'None')
    final_df.loc[discount_mask, 'Discount'] = final_df.loc[discount_mask, 'Discount'].apply(
        lambda x: f"({random.randint(5, 50)}% off)"
    )
    print(f"Filled {discount_mask.sum()} missing Discounts")
    
    print("\nFilling missing Delivery Date...")
    delivery_mask = (final_df['Delivery Date'].isna()) | (final_df['Delivery Date'] == '') | (final_df['Delivery Date'] == 'None')
    delivery_options = [
        "FREE delivery Tomorrow",
        "FREE delivery in 2 days",
        "FREE delivery in 3 days",
        "Delivery by next week"
    ]
    final_df.loc[delivery_mask, 'Delivery Date'] = final_df.loc[delivery_mask, 'Delivery Date'].apply(
        lambda x: random.choice(delivery_options)
    )
    print(f"Filled {delivery_mask.sum()} missing Delivery Dates")
    
    print("\nFilling missing prices...")
    mrp_mask = (final_df['MRP Price'].isna()) | (final_df['MRP Price'] == '') | (final_df['MRP Price'] == 'None')
    selling_mask = (final_df['Selling Price'].isna()) | (final_df['Selling Price'] == '') | (final_df['Selling Price'] == 'None')
    
    for idx in final_df.index:
        if pd.isna(final_df.loc[idx, 'Selling Price']) or final_df.loc[idx, 'Selling Price'] == '' or final_df.loc[idx, 'Selling Price'] == 'None':
            final_df.loc[idx, 'Selling Price'] = f"Rs. {random.randint(500, 50000)}"
        
        if pd.isna(final_df.loc[idx, 'MRP Price']) or final_df.loc[idx, 'MRP Price'] == '' or final_df.loc[idx, 'MRP Price'] == 'None':
            try:
                selling_price = int(str(final_df.loc[idx, 'Selling Price']).replace('Rs.', '').replace(',', '').strip())
                mrp_price = int(selling_price * random.uniform(1.1, 1.5))
                final_df.loc[idx, 'MRP Price'] = f"Rs. {mrp_price}"
            except:
                final_df.loc[idx, 'MRP Price'] = f"Rs. {random.randint(1000, 60000)}"
    
    print(f"Filled {mrp_mask.sum()} missing MRP Prices")
    print(f"Filled {selling_mask.sum()} missing Selling Prices")
    
    print("\nChecking for duplicate products...")
    print("Duplicate criteria: Same Product Name + Same Provider + Same Selling Price")
    print("-"*80)
    
    duplicate_products = final_df[final_df.duplicated(subset=['Product', 'Provider', 'Selling Price'], keep=False)]
    if len(duplicate_products) > 0:
        duplicate_groups = final_df.groupby(['Product', 'Provider', 'Selling Price']).size()
        duplicate_groups = duplicate_groups[duplicate_groups > 1].sort_values(ascending=False)
        
        print(f"Found {len(duplicate_groups)} unique product combinations with duplicates")
        print(f"Total duplicate records: {len(duplicate_products)}")
        
        print("\nTop 5 examples of duplicates:")
        for i, ((product_name, provider, price), count) in enumerate(duplicate_groups.head(5).items(), 1):
            print(f"\n{i}. Product: '{product_name[:50]}...'")
            print(f"   Provider: {provider}")
            print(f"   Selling Price: {price}")
            print(f"   Appears: {count} times")
            
            example_rows = final_df[(final_df['Product'] == product_name) & 
                                   (final_df['Provider'] == provider) & 
                                   (final_df['Selling Price'] == price)]
            print(f"   Brands: {example_rows['Brands'].unique().tolist()}")
            print(f"   Ratings: {example_rows['Rating'].unique().tolist()}")
    else:
        print("No duplicates found!")
    
    print("\nMerging duplicate products (same Product, Provider, and Selling Price)...")
    before_merge = len(final_df)
    aggregation = {
        'Brands': 'first',
        'Discount': 'first',
        'Rating': 'max',
        'Bought Count': 'first',
        'MRP Price': 'first',
        'Delivery Date': 'first',
        'Product Image URL': 'first'
    }
    
    final_df = final_df.groupby(['Product', 'Provider', 'Selling Price'], as_index=False).agg(aggregation)
    print(f"Removed {before_merge - len(final_df)} duplicate products")
    print(f"Final unique products: {len(final_df)}")
    
    print("\nAdding Product IDs...")
    final_df.insert(0, 'Product ID', [str(uuid.uuid4()) for _ in range(len(final_df))])

    output_path = '/Users/jothikaravichandran/Documents/Self_Projects/personal_project/ai_geeks_product_agent/ai-product-agent/source_product_input/merged_product_data.csv'
    final_df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("Data Processing Complete!")
    print(f"Output file: {output_path}")
    print(f"\nRecord Count Summary:")
    print(f"  Input files total: {total_records_from_files} records")
    print(f"  After merge: {merged_count} records")
    print(f"  After deduplication: {len(final_df)} records")
    print(f"  Duplicates removed: {merged_count - len(final_df)} records")
    print(f"\nProviders: {final_df['Provider'].unique().tolist()}")
    print(f"Columns: {list(final_df.columns)}")
    print("="*80)
    
    save_cache(cache)
    print("\nCache saved for future runs")
else:
    print("\nNo dataframes to merge!")

print("\nAll files merged successfully into one structured file!")
