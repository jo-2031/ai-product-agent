import pandas as pd
import glob
import uuid

files = glob.glob('/Users/jothikaravichandran/Documents/Self_Projects/personal_project/ai_geeks_product_agent/Raw_product_data/csv_format/*.csv')

required_columns = [
    'Product', 'Brands', 'Discount', 'Rating', 
    'Bought Count', 'MRP Price', 'Selling Price', 
    'Delivery Date', 'image'
]

dataframes = []

for file in files:
    df = pd.read_csv(file)
    
    # Add missing columns as NaN
    df = df.reindex(columns=required_columns)
    
    dataframes.append(df)

if dataframes:
# Merge all files vertically (row-wise)
    final_df = pd.concat(dataframes, ignore_index=True)
    final_df.insert(0, 'Product ID', [str(uuid.uuid4()) for _ in range(len(final_df))])  
    final_df.rename(columns={'image': 'Product Image URL'}, inplace=True)
    
    # Condition → Brands is empty
    mask = (final_df['Brands'].isna() | (final_df['Brands'] == ''))

    # Extract text before first comma, colon, or pipe
    brand_values = (
        final_df.loc[mask, 'Product']
        .str.split(r'[,:|]', n=1, expand=True)[0]
        .str.strip()
    )

    # Copy into Brands column
    final_df.loc[mask, 'Brands'] = brand_values

    final_df.to_csv(
        '/Users/jothikaravichandran/Documents/Self_Projects/personal_project/ai_geeks_product_agent/ai-product-agent/source_product_input/merged_product_data.csv',
        index=False
    )

print("All files merged successfully into one structured file!")
