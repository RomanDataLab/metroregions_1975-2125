import pandas as pd
import re

# Common city name translations
city_translations = {
    # Portuguese/Spanish cities
    'Lisboa': 'Lisbon',
    'Lisboa (Lisbon)': 'Lisbon',
    'La Habana': 'Havana',
    'La Habana (Havana)': 'Havana',
    'Asunción': 'Asuncion',
    'São Paulo': 'Sao Paulo',
    'Bogotá': 'Bogota',
    'México': 'Mexico City',
    'Ciudad de México': 'Mexico City',
    'Panamá': 'Panama City',
    'Ciudad de Panamá': 'Panama City',
    'San José': 'San Jose',
    
    # Arabic cities
    'Al-Manāmah': 'Manama',
    'Al-Manāmah (Manama)': 'Manama',
    'Ad-Dawhah': 'Doha',
    'Ad-Dawhah (Doha)': 'Doha',
    'Al Kuwayt': 'Kuwait City',
    'Al Kuwayt (Kuwait City)': 'Kuwait City',
    'Abū Ẓabī': 'Abu Dhabi',
    'Abū Ẓabī (Abu Dhabi)': 'Abu Dhabi',
    
    # Other common translations
    'Guatemala': 'Guatemala City',
    'Ciudad de Guatemala': 'Guatemala City',
    "M'bour": 'Mbour',  # Keep as is, standard transliteration
}

print("Reading metro_exist.csv...")
df = pd.read_csv('metro_exist.csv')

print(f"Total rows: {len(df)}")
print(f"Processing city name translations...")

# Function to extract English name from parentheses or translate
def translate_city_name(name):
    if pd.isna(name):
        return name
    
    name_str = str(name).strip()
    
    # Check if name is already in translation dictionary
    if name_str in city_translations:
        return city_translations[name_str]
    
    # Check if name contains parentheses with English translation
    # Pattern: "Foreign Name (English Name)"
    match = re.search(r'\(([^)]+)\)', name_str)
    if match:
        english_name = match.group(1)
        # Remove the parentheses part and use the English name
        return english_name
    
    # Check if name starts with common prefixes that might have translations
    # Remove diacritics and check
    name_clean = name_str
    # Common patterns
    if name_clean.startswith('Al-') or name_clean.startswith('Ad-') or name_clean.startswith('Abū'):
        # These are often Arabic names, check if we have a translation
        pass
    
    # For now, return the name as-is if no translation found
    # Most city names are already in standard English transliteration
    return name_str

# Translate all city names
print("Translating city names...")
df['City_Name'] = df['City_Name'].apply(translate_city_name)

print("Translation complete!")

# Save the updated file
output_file = 'metro_exist.csv'
df.to_csv(output_file, index=False)
print(f"\nUpdated file saved to {output_file}")
print(f"Total cities processed: {len(df)}")

