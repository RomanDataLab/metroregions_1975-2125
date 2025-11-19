import pandas as pd
import numpy as np

print("Reading cities_filtered.csv...")
df = pd.read_csv('cities_filtered.csv')

print(f"Original shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Get unique City_Code values
city_codes = df['City_Code'].unique()
print(f"\nNumber of unique City_Code values: {len(city_codes)}")

# Years from 1975 to 2050
years = list(range(1975, 2051))
print(f"Years range: {min(years)} to {max(years)}")

# Pivot: City_Code as index, Year as columns, Pop_1Jan as values
print("\nPivoting data (City_Code x Year)...")
pivot_df = df.pivot_table(
    index='City_Code',
    columns='Year',
    values='Pop_1Jan',
    aggfunc='first'  # Use first value if duplicates exist
)

# Rename columns to year format (e.g., 1975, 1976, etc.)
pivot_df.columns = [f'Pop_{int(col)}' for col in pivot_df.columns]

# Ensure all years from 1975-2050 are present (fill missing with NaN)
for year in years:
    col_name = f'Pop_{year}'
    if col_name not in pivot_df.columns:
        pivot_df[col_name] = np.nan

# Reorder columns to match years sequence
year_columns = [f'Pop_{year}' for year in years]
pivot_df = pivot_df[year_columns]

print(f"Pivot shape: {pivot_df.shape}")

# Get metadata columns for each City_Code (should be same for all years of same city)
print("\nExtracting metadata columns...")
metadata_cols = ['LocID', 'Location', 'Notes', 'ISO3_Code', 'ISO2_Code', 'City_Name', 
                 'PWCent_Latitude', 'PWCent_Longitude']

# Get unique metadata for each City_Code (take first occurrence)
metadata_df = df.groupby('City_Code')[metadata_cols].first().reset_index()

print(f"Metadata shape: {metadata_df.shape}")

# Get remaining columns (excluding Year, Pop_1Jan, and metadata columns already included)
all_cols = list(df.columns)
excluded_cols = ['Year', 'Pop_1Jan'] + metadata_cols
remaining_cols = [col for col in all_cols if col not in excluded_cols and col != 'City_Code']

print(f"\nRemaining columns to add: {remaining_cols}")

# Get remaining columns data (take first occurrence for each City_Code)
if remaining_cols:
    remaining_df = df.groupby('City_Code')[remaining_cols].first().reset_index()
    print(f"Remaining columns shape: {remaining_df.shape}")
else:
    remaining_df = pd.DataFrame({'City_Code': city_codes})

# Merge everything together
print("\nMerging all data...")
result_df = pivot_df.reset_index()

# Merge with metadata
result_df = result_df.merge(metadata_df, on='City_Code', how='left')

# Merge with remaining columns
if remaining_cols:
    result_df = result_df.merge(remaining_df, on='City_Code', how='left')

# Reorder columns: City_Code first, then year columns, then metadata, then remaining
final_columns = ['City_Code'] + year_columns + metadata_cols
if remaining_cols:
    final_columns += remaining_cols

# Ensure all columns exist
final_columns = [col for col in final_columns if col in result_df.columns]
result_df = result_df[final_columns]

print(f"\nFinal shape: {result_df.shape}")
print(f"Final columns ({len(result_df.columns)}):")
for i, col in enumerate(result_df.columns, 1):
    print(f"  {i}. {col}")

# Save to CSV
output_file = 'metro_exist.csv'
result_df.to_csv(output_file, index=False)
print(f"\nOutput saved to {output_file}")

# Show sample
print("\nSample data (first 5 rows, first 10 columns):")
print(result_df.iloc[:5, :10])


