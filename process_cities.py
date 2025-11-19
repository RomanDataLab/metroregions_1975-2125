import pandas as pd
import re

def convert_numeric_columns(df):
    """Convert columns containing digits (numeric data) to numeric type"""
    numeric_columns = []
    # Text columns that should remain as strings
    text_columns = ['Location', 'Notes', 'ISO3_Code', 'ISO2_Code', 'City_Name', 'Pop_plausibility']
    
    for col in df.columns:
        # Skip known text columns
        if col in text_columns:
            continue
        
        # Try to convert to numeric
        try:
            # Attempt conversion, coercing errors to NaN
            converted = pd.to_numeric(df[col], errors='coerce')
            # Check if most values are numeric (at least 80% are not NaN)
            non_null_ratio = converted.notna().sum() / len(df)
            if non_null_ratio > 0.8:
                # For City_Code, convert to int if possible
                if col == 'City_Code':
                    df[col] = converted.fillna(0).astype(int)
                else:
                    df[col] = converted
                numeric_columns.append(col)
        except Exception as e:
            # Keep original if conversion fails
            pass
    
    return df, numeric_columns

# Read the CSV file
print("Reading cities_50k.csv...")
df = pd.read_csv('cities_50k.csv')

print(f"Original shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Convert numeric columns
print("\nConverting numeric columns...")
df, numeric_cols = convert_numeric_columns(df)
print(f"Converted {len(numeric_cols)} columns to numeric")

# Filter for Year = 2025 and Pop_1Jan >= 500,000
# Note: Pop_1Jan values are in thousands, so 500 = 500,000
print("\nFiltering for Year = 2025 and Pop_1Jan >= 500 (500,000)...")
df_filtered = df[(df['Year'] == 2025) & (df['Pop_1Jan'] >= 500)]

print(f"Filtered shape: {df_filtered.shape}")
print(f"Number of rows matching criteria: {len(df_filtered)}")

# Get unique City_Code values
print("\nExtracting unique City_Code values...")
unique_city_codes = df_filtered['City_Code'].unique()

print(f"Number of unique City_Code values: {len(unique_city_codes)}")

# Create output DataFrame
output_df = pd.DataFrame({
    'City_Code': unique_city_codes
})

# Sort by City_Code
output_df = output_df.sort_values('City_Code').reset_index(drop=True)

# Save to CSV
output_file = 'citycode25.csv'
output_df.to_csv(output_file, index=False)

print(f"\nOutput saved to {output_file}")
print(f"Total unique City_Code values: {len(output_df)}")

# Show first few rows
print("\nFirst 10 City_Code values:")
print(output_df.head(10))

