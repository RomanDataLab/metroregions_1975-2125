import pandas as pd
import numpy as np

print("Reading metro_exist.csv...")
df = pd.read_csv('metro_exist.csv')

print(f"Original shape: {df.shape}")

# Get all Pop_ columns (years from 1975 to 2050)
# Filter to only include columns that match Pop_YYYY pattern (4-digit year)
all_pop_cols = [col for col in df.columns if col.startswith('Pop_')]
print(f"\nFound {len(all_pop_cols)} columns starting with 'Pop_'")

# Extract year numbers from column names (only if it's a 4-digit year)
pop_columns = []
year_columns = {}
for col in all_pop_cols:
    try:
        year_str = col.replace('Pop_', '')
        # Check if it's a 4-digit year (1975-2050)
        if len(year_str) == 4:
            year = int(year_str)
            if 1975 <= year <= 2050:
                pop_columns.append(col)
                year_columns[col] = year
    except:
        pass

print(f"Found {len(pop_columns)} Pop_ year columns (1975-2050)")

# Sort columns by year
sorted_pop_cols = sorted(pop_columns, key=lambda x: year_columns[x])
print(f"Year range: {year_columns[sorted_pop_cols[0]]} to {year_columns[sorted_pop_cols[-1]]}")

# Check for missing values
print("\nChecking for missing values...")
missing_counts = {}
for col in sorted_pop_cols:
    missing = df[col].isna().sum()
    if missing > 0:
        missing_counts[col] = missing

if missing_counts:
    print(f"\nFound {len(missing_counts)} columns with missing values:")
    for col, count in sorted(missing_counts.items(), key=lambda x: year_columns[x[0]]):
        print(f"  {col}: {count} missing values ({count/len(df)*100:.1f}%)")
else:
    print("No missing values found in Pop_ columns!")

# Convert Pop_ columns to numeric
print("\nConverting Pop_ columns to numeric...")
for col in sorted_pop_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Count missing values before interpolation
missing_before = df[sorted_pop_cols].isna().sum().sum()
print(f"Total missing values before interpolation: {missing_before}")

# Interpolate missing values row by row (since each row represents a city's time series)
print("\nInterpolating missing values...")
interpolated_count = 0

# Create a temporary dataframe with just Pop_ columns for easier manipulation
pop_df = df[sorted_pop_cols].copy()

# For each row, interpolate missing values
for idx in pop_df.index:
    row_values = pop_df.loc[idx].values
    
    # Check if there are any NaN values
    if pd.isna(row_values).any():
        # Get years
        years = [year_columns[col] for col in sorted_pop_cols]
        
        # Create a series for interpolation
        pop_series = pd.Series(row_values, index=years)
        
        # Interpolate missing values using linear interpolation
        interpolated = pop_series.interpolate(method='linear', limit_direction='both')
        
        # Fill any remaining NaN at the edges with forward/backward fill
        interpolated = interpolated.ffill().bfill()
        
        # Count how many values were interpolated
        original_nans = pd.isna(row_values).sum()
        new_nans = pd.isna(interpolated.values).sum()
        interpolated_count += (original_nans - new_nans)
        
        # Update the dataframe
        pop_df.loc[idx] = interpolated.values

# Update the original dataframe with interpolated values
df[sorted_pop_cols] = pop_df

print(f"Interpolated {interpolated_count} missing values")

# Verify no missing values remain
print("\nVerifying interpolation...")
remaining_missing = {}
for col in sorted_pop_cols:
    missing = df[col].isna().sum()
    if missing > 0:
        remaining_missing[col] = missing

if remaining_missing:
    print(f"Warning: {len(remaining_missing)} columns still have missing values:")
    for col, count in sorted(remaining_missing.items(), key=lambda x: year_columns[x[0]]):
        print(f"  {col}: {count} missing values")
else:
    print("All missing values have been interpolated!")

# Show sample of interpolated data
print("\nSample of interpolated data (first row with interpolations):")
sample_row = None
for idx, row in df.iterrows():
    # Check if this row had interpolations by comparing with original
    # For now, just show first row
    if idx == 0:
        sample_row = row
        break

if sample_row is not None:
    print(f"City_Code: {sample_row['City_Code']}, City_Name: {sample_row['City_Name']}")
    print("Sample Pop_ values (every 10 years):")
    sample_years = [1975, 1985, 1995, 2005, 2015, 2025, 2035, 2045]
    for year in sample_years:
        col = f'Pop_{year}'
        if col in df.columns:
            print(f"  {col}: {sample_row[col]:.2f}")

# Save the updated file
output_file = 'metro_exist.csv'
df.to_csv(output_file, index=False)
print(f"\nUpdated file saved to {output_file}")

