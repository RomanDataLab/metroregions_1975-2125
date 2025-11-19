import pandas as pd
import numpy as np

print("Reading metro_exist.csv...")
df = pd.read_csv('metro_exist.csv')

print(f"Original shape: {df.shape}")

# Get all Pop_ columns (years from 1975 to 2050)
# Filter to only include columns that match Pop_YYYY pattern (4-digit year)
pop_columns = []
for col in df.columns:
    if col.startswith('Pop_'):
        year_str = col.replace('Pop_', '')
        # Check if it's a 4-digit year (1975-2050)
        if len(year_str) == 4:
            try:
                year = int(year_str)
                if 1975 <= year <= 2050:
                    pop_columns.append(col)
            except:
                pass

pop_columns = sorted(pop_columns, key=lambda x: int(x.replace('Pop_', '')))

print(f"\nFound {len(pop_columns)} Pop_ year columns")

# Show sample values before rounding
print("\nSample values before rounding (first row):")
sample_before = df.iloc[0][pop_columns[:5]].tolist()
for i, col in enumerate(pop_columns[:5]):
    print(f"  {col}: {df.iloc[0][col]}")

# Round all Pop_ columns to 0 decimal places
print("\nRounding all Pop_ columns to 0 decimal places...")
for col in pop_columns:
    # Convert to numeric, round to nearest integer
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].round(0).astype('Int64')  # Int64 supports NaN

print("Rounding complete!")

# Show sample values after rounding
print("\nSample values after rounding (first row):")
sample_after = df.iloc[0][pop_columns[:5]].tolist()
for i, col in enumerate(pop_columns[:5]):
    print(f"  {col}: {df.iloc[0][col]}")

# Check for any remaining NaN values
print("\nChecking for NaN values...")
nan_counts = {}
for col in pop_columns:
    nan_count = df[col].isna().sum()
    if nan_count > 0:
        nan_counts[col] = nan_count

if nan_counts:
    print(f"Warning: {len(nan_counts)} columns still have NaN values:")
    for col, count in sorted(nan_counts.items())[:10]:
        print(f"  {col}: {count} NaN values")
else:
    print("No NaN values found in Pop_ columns")

# Save the updated file
output_file = 'metro_exist.csv'
df.to_csv(output_file, index=False)
print(f"\nUpdated file saved to {output_file}")

# Show summary statistics
print("\nSummary statistics for Pop_2025:")
pop_2025 = df['Pop_2025']
print(f"  Min: {pop_2025.min()}")
print(f"  Max: {pop_2025.max()}")
print(f"  Mean: {pop_2025.mean():.0f}")
print(f"  Non-null count: {pop_2025.notna().sum()}")

