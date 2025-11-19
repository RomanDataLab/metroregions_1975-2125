import pandas as pd

print("Reading citycode25.csv...")
city_codes_df = pd.read_csv('citycode25.csv')
city_codes = set(city_codes_df['City_Code'].unique())
print(f"Found {len(city_codes)} unique City_Code values to filter")

print("\nReading cities_50k.csv...")
cities_df = pd.read_csv('cities_50k.csv')
print(f"Original shape: {cities_df.shape}")

# Convert City_Code to numeric for proper comparison
cities_df['City_Code'] = pd.to_numeric(cities_df['City_Code'], errors='coerce')

# Filter rows where City_Code is in the list
print("\nFiltering rows where City_Code matches citycode25.csv...")
filtered_df = cities_df[cities_df['City_Code'].isin(city_codes)]

print(f"Filtered shape: {filtered_df.shape}")
print(f"Number of rows extracted: {len(filtered_df)}")
print(f"Number of unique City_Code values in filtered data: {filtered_df['City_Code'].nunique()}")

# Show some statistics
if len(filtered_df) > 0:
    print(f"\nYear range: {filtered_df['Year'].min()} - {filtered_df['Year'].max()}")
    print(f"Sample rows:")
    print(filtered_df[['City_Code', 'City_Name', 'Year', 'Pop_1Jan']].head(10))

# Save the filtered data
output_file = 'cities_filtered.csv'
filtered_df.to_csv(output_file, index=False)
print(f"\nFiltered data saved to {output_file}")


