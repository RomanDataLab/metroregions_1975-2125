import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("Loading metro_exist.csv...")
df = pd.read_csv('metro_exist.csv')

# Extract year columns (1975-2050)
year_cols = [f'Pop_{year}' for year in range(1975, 2051)]
print(f"Found {len(year_cols)} year columns")

# Create a mapping of countries to typical characteristics
# This is a simplified model - in production, you'd use real data sources
country_factors = {}

# Initialize country factors based on location
for location in df['Location'].unique():
    if pd.isna(location):
        continue
    
    location_str = str(location).lower()
    
    # Birth rate factors (declining globally, but varies by region)
    # Made less aggressive to prevent forced decline
    if any(x in location_str for x in ['africa', 'nigeria', 'ethiopia', 'egypt', 'kenya']):
        birth_rate_factor = 1.01  # Higher birth rates (reduced from 1.02)
    elif any(x in location_str for x in ['india', 'pakistan', 'bangladesh', 'indonesia']):
        birth_rate_factor = 1.005  # Moderate-high (reduced from 1.01)
    elif any(x in location_str for x in ['china', 'japan', 'korea', 'taiwan', 'singapore']):
        birth_rate_factor = 0.995  # Low/declining (less negative, was 0.98)
    elif any(x in location_str for x in ['europe', 'germany', 'france', 'italy', 'spain', 'uk', 'russia']):
        birth_rate_factor = 0.998  # Low (less negative, was 0.99)
    else:
        birth_rate_factor = 1.0  # Neutral
    
    # War conflict probability (historical patterns)
    conflict_risk = 0.0
    if any(x in location_str for x in ['syria', 'yemen', 'iraq', 'afghanistan', 'ukraine', 'israel', 'palestine']):
        conflict_risk = 0.15  # Higher conflict risk
    elif any(x in location_str for x in ['libya', 'sudan', 'somalia', 'congo', 'myanmar']):
        conflict_risk = 0.10  # Moderate conflict risk
    elif any(x in location_str for x in ['mexico', 'colombia', 'venezuela']):
        conflict_risk = 0.05  # Lower but present
    else:
        conflict_risk = 0.02  # Low baseline
    
    country_factors[location] = {
        'birth_rate_factor': birth_rate_factor,
        'conflict_risk': conflict_risk
    }

print(f"Processed {len(country_factors)} countries")

# Function to identify if a city is in Asia or Africa
def is_asian_or_african(location):
    """
    Check if a city is located in Asia or Africa based on Location string
    """
    if pd.isna(location):
        return False
    
    location_str = str(location).lower()
    
    # African countries/regions
    african_keywords = [
        'africa', 'nigeria', 'ethiopia', 'egypt', 'kenya', 'south africa',
        'ghana', 'tanzania', 'uganda', 'sudan', 'morocco', 'algeria',
        'angola', 'mozambique', 'madagascar', 'cameroon', 'ivory coast',
        'niger', 'mali', 'burkina faso', 'malawi', 'zambia', 'senegal',
        'chad', 'somalia', 'guinea', 'rwanda', 'benin', 'burundi',
        'tunisia', 'south sudan', 'togo', 'sierra leone', 'libya',
        'liberia', 'central african republic', 'mauritania', 'eritrea',
        'gambia', 'botswana', 'namibia', 'gabon', 'lesotho', 'guinea-bissau',
        'equatorial guinea', 'mauritius', 'eswatini', 'djibouti', 'comoros',
        'cabo verde', 'sao tome and principe', 'seychelles', 'congo',
        'democratic republic', 'zimbabwe', 'libya'
    ]
    
    # Asian countries/regions
    asian_keywords = [
        'asia', 'china', 'india', 'indonesia', 'pakistan', 'bangladesh',
        'japan', 'philippines', 'vietnam', 'thailand', 'south korea',
        'myanmar', 'iraq', 'afghanistan', 'saudi arabia', 'uzbekistan',
        'yemen', 'nepal', 'north korea', 'sri lanka', 'kazakhstan',
        'syria', 'cambodia', 'jordan', 'azerbaijan', 'united arab emirates',
        'tajikistan', 'israel', 'hong kong', 'laos', 'kyrgyzstan',
        'turkmenistan', 'singapore', 'lebanon', 'palestine', 'oman',
        'kuwait', 'georgia', 'mongolia', 'armenia', 'qatar', 'bahrain',
        'bhutan', 'maldives', 'brunei', 'macau', 'taiwan', 'iran',
        'turkey', 'russia', 'malaysia', 'bangladesh'
    ]
    
    return any(keyword in location_str for keyword in african_keywords + asian_keywords)

# Function to apply parabolic decay of growth, plateau, and light negative progression
def apply_growth_decay_with_plateau(predictions):
    """
    Apply a three-phase model for Asian/African cities:
    1. Parabolic decay of growth rate (2051-2085): Growth rate decreases parabolically
    2. Plateau phase (2085-2100): Growth rate ~0, population stabilizes
    3. Light negative progression (2100-2125): Small negative growth rate, gradual decline
    
    Args:
        predictions: List of population predictions (2051-2125)
    
    Returns:
        Modified predictions list with growth decay, plateau, and decline applied
    """
    if len(predictions) < 75:  # Need all 75 years
        return predictions
    
    # Phase boundaries (indices relative to 2051)
    # 2051 is index 0, 2085 is index 34, 2100 is index 49, 2125 is index 74
    growth_decay_end = 34  # 2085
    plateau_end = 49      # 2100
    final_year = 74        # 2125
    
    modified_predictions = predictions.copy()
    
    # Calculate initial growth rate from first few years
    if predictions[0] > 0 and predictions[4] > 0:
        initial_growth_rate = (predictions[4] - predictions[0]) / (4 * predictions[0])
    else:
        initial_growth_rate = 0.01  # Default 1% growth
    
    # Clamp initial growth rate to reasonable bounds
    initial_growth_rate = np.clip(initial_growth_rate, 0.0, 0.05)  # 0% to 5%
    
    # Phase 1: Parabolic decay of growth rate (2051-2085)
    # Growth rate decays parabolically from initial_growth_rate to ~0
    for i in range(1, growth_decay_end + 1):
        t = i / growth_decay_end  # Normalized time (0 to 1)
        # Parabolic decay: growth_rate(t) = initial * (1 - t^2)
        growth_rate = initial_growth_rate * (1 - t * t)
        
        # Apply growth rate to population
        modified_predictions[i] = modified_predictions[i-1] * (1 + growth_rate)
    
    # Phase 2: Plateau (2085-2100) - maintain population with very small fluctuations
    plateau_pop = modified_predictions[growth_decay_end]
    for i in range(growth_decay_end + 1, plateau_end + 1):
        # Very small random fluctuations around plateau (±0.1%)
        fluctuation = np.random.uniform(-0.001, 0.001)
        modified_predictions[i] = plateau_pop * (1 + fluctuation)
    
    # Phase 3: Light negative progression (2100-2125)
    # Small negative growth rate causing gradual decline
    plateau_pop = modified_predictions[plateau_end]
    negative_growth_rate = -0.002  # -0.2% per year (light decline)
    years_decline = final_year - plateau_end  # 25 years
    
    for i in range(plateau_end + 1, final_year + 1):
        years_since_plateau = i - plateau_end
        # Apply compound negative growth
        modified_predictions[i] = plateau_pop * ((1 + negative_growth_rate) ** years_since_plateau)
    
    return modified_predictions

# Function to predict future population using ML
def predict_population_ml(historical_values, years_ahead=75, birth_factor=1.0, conflict_risk=0.02):
    """
    Predict population using machine learning with birth rate and conflict factors
    """
    # Convert to numpy array and remove NaN
    values = np.array(historical_values)
    valid_mask = ~np.isnan(values)
    
    if np.sum(valid_mask) < 10:  # Need at least 10 data points
        # Fallback to simple extrapolation with constraints
        if np.sum(valid_mask) > 0:
            valid_vals = values[valid_mask]
            last_valid = valid_vals[-1]
            
            # Calculate growth rate more conservatively
            if len(valid_vals) > 5:
                # Use recent trend (last 5 years) for more accurate projection
                recent_growth = np.mean(np.diff(valid_vals[-5:])) / valid_vals[-5] if valid_vals[-5] > 0 else 0
            else:
                recent_growth = np.mean(np.diff(valid_vals)) / last_valid if len(valid_vals) > 1 and last_valid > 0 else 0.005
            
            # Constrain growth rate to reasonable bounds (more permissive)
            recent_growth = np.clip(recent_growth, -0.01, 0.05)  # Max 5% growth, max 1% decline
            
            predictions = []
            current = last_valid
            # Dynamic minimum: allow gradual decline but prevent sudden drops
            base_min_pop = last_valid * 0.3  # Start with 30% minimum
            
            for year_idx in range(years_ahead):
                # Apply growth with birth factor
                # Removed time decay to allow natural trends (was causing forced decline)
                growth = recent_growth * birth_factor
                current = current * (1 + growth)
                
                # CRITICAL: Cap extreme growth - never exceed 3% annual growth compounded
                max_growth_factor = 1.03 ** (year_idx + 1)
                max_allowed = last_valid * max_growth_factor
                current = min(current, max_allowed)
                
                # Also cap at 5x the starting value as absolute maximum
                absolute_max = last_valid * 5
                current = min(current, absolute_max)
                
                # Minimum population floor - only prevent unrealistic sudden drops
                # Don't force decline - keep minimum stable to allow natural trends
                min_pop = base_min_pop  # Keep at 30% of starting value (don't force decline)
                
                # Apply conflict impact (less severe, more realistic)
                if np.random.random() < conflict_risk * 0.5:  # Reduced frequency
                    conflict_impact = 1 - np.random.uniform(0.01, 0.10)  # 1-10% reduction (more realistic)
                    current = current * conflict_impact
                
                # Ensure minimum population
                current = max(min_pop, current)
                predictions.append(current)
            return predictions
        else:
            # If no valid data, return zeros
            return [0] * years_ahead
    
    valid_values = values[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    # Prepare features for ML model
    # Use recent trend, growth rate, and position in time series
    X_train = []
    y_train = []
    
    # Create features from sliding windows
    window_size = min(10, len(valid_values) // 2)
    for i in range(window_size, len(valid_values)):
        window = valid_values[i-window_size:i]
        
        # Features: recent values, trend, growth rate
        features = [
            window[-1],  # Latest value
            np.mean(window),  # Mean
            np.std(window) if len(window) > 1 else 0,  # Std dev
            (window[-1] - window[0]) / len(window) if len(window) > 1 else 0,  # Trend
            np.mean(np.diff(window)) if len(window) > 1 else 0,  # Growth rate
            valid_indices[i] / len(values),  # Position in time series
        ]
        
        X_train.append(features)
        y_train.append(valid_values[i])
    
    if len(X_train) < 5:
        # Fallback to exponential smoothing with constraints
        predictions = []
        current = valid_values[-1]
        base_min_pop = current * 0.3  # Start with 30% minimum
        
        # Calculate conservative growth rate
        if len(valid_values) > 5:
            recent_growth = np.mean(np.diff(valid_values[-5:])) / valid_values[-5] if valid_values[-5] > 0 else 0
        else:
            recent_growth = np.mean(np.diff(valid_values)) / valid_values[-1] if len(valid_values) > 1 and valid_values[-1] > 0 else 0.005
        recent_growth = np.clip(recent_growth, -0.01, 0.05)
        
        for year_idx in range(years_ahead):
            # Removed time decay to allow natural trends
            growth = recent_growth * birth_factor
            current = current * (1 + growth)
            
            # CRITICAL: Cap extreme growth - never exceed 3% annual growth compounded
            max_growth_factor = 1.03 ** (year_idx + 1)
            max_allowed = valid_values[-1] * max_growth_factor
            current = min(current, max_allowed)
            
            # Also cap at 5x the starting value as absolute maximum
            absolute_max = valid_values[-1] * 5
            current = min(current, absolute_max)
            
            if np.random.random() < conflict_risk * 0.5:
                current = current * (1 - np.random.uniform(0.01, 0.10))
            
            # Minimum population floor - only prevent unrealistic sudden drops
            # Don't force decline - keep minimum stable
            min_pop = base_min_pop  # Keep at 30% of starting value (don't force decline)
            current = max(min_pop, current)
            predictions.append(current)
        return predictions
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Train Ridge regression model (regularized linear regression)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Predict future values
    predictions = []
    current_window = list(valid_values[-window_size:])
    current_index = len(values)
    
    for year_ahead in range(years_ahead):
        # Prepare features for prediction
        window = np.array(current_window[-window_size:])
        features = np.array([[
            window[-1],
            np.mean(window),
            np.std(window) if len(window) > 1 else 0,
            (window[-1] - window[0]) / len(window) if len(window) > 1 else 0,
            np.mean(np.diff(window)) if len(window) > 1 else 0,
            (current_index + year_ahead) / (len(values) + years_ahead),
        ]])
        
        # Predict next value
        predicted = model.predict(features)[0]
        
        # Ensure prediction is reasonable (not negative or too extreme)
        if predicted < 0:
            # Use trend-based prediction instead
            if len(current_window) > 1:
                trend = (current_window[-1] - current_window[-2]) / current_window[-2] if current_window[-2] > 0 else 0
                trend = np.clip(trend, -0.02, 0.03)  # Constrain trend
                predicted = current_window[-1] * (1 + trend)
            else:
                predicted = current_window[-1]
        
        # CRITICAL: Cap extreme predictions to prevent unrealistic growth
        # Maximum growth: 3% per year compounded over projection period
        max_growth_factor = 1.03 ** (year_ahead + 1)  # Allow up to 3% annual growth
        max_predicted = valid_values[-1] * max_growth_factor
        
        # Also check against recent trend to prevent sudden jumps
        if len(current_window) > 1:
            recent_trend = (current_window[-1] - current_window[-2]) / current_window[-2] if current_window[-2] > 0 else 0
            recent_trend = np.clip(recent_trend, -0.02, 0.03)  # Constrain to -2% to +3%
            trend_based = current_window[-1] * (1 + recent_trend)
            max_predicted = min(max_predicted, trend_based * 2)  # Don't exceed 2x trend-based
        
        # Apply upper bound
        predicted = min(predicted, max_predicted)
        
        # Apply birth rate factor
        # Removed time factor to allow natural trends (was causing forced decline)
        predicted = predicted * birth_factor
        
        # Apply conflict impact (less severe, more realistic)
        if np.random.random() < conflict_risk * 0.5:  # Reduced frequency
            conflict_impact = 1 - np.random.uniform(0.01, 0.10)  # 1-10% reduction
            predicted = predicted * conflict_impact
        
        # Ensure minimum population - only prevent unrealistic sudden drops
        # Don't force decline - keep minimum stable to allow natural trends
        base_min = valid_values[-1] * 0.3
        min_pop = base_min  # Keep at 30% of starting value (don't force decline)
        predicted = max(min_pop, predicted)
        
        # Final sanity check: ensure prediction is within reasonable bounds
        # Never exceed 5x the last known value, even with all factors
        absolute_max = valid_values[-1] * 5
        predicted = min(predicted, absolute_max)
        
        predictions.append(predicted)
        
        # Update window for next prediction
        current_window.append(predicted)
        if len(current_window) > window_size * 2:
            current_window = current_window[-window_size:]
    
    return predictions

# Set random seed for reproducibility
np.random.seed(42)

# Process each city
print("\nPredicting populations for 2051-2125...")
new_columns = {}

for idx, row in df.iterrows():
    if idx % 100 == 0:
        print(f"Processing city {idx+1}/{len(df)}...")
    
    # Get historical population values
    historical = [row[col] for col in year_cols]
    
    # Get country factors
    location = row['Location']
    if pd.isna(location) or location not in country_factors:
        birth_factor = 1.0
        conflict_risk = 0.02
    else:
        birth_factor = country_factors[location]['birth_rate_factor']
        conflict_risk = country_factors[location]['conflict_risk']
    
    # Predict future populations
    predictions = predict_population_ml(historical, years_ahead=75, 
                                        birth_factor=birth_factor, 
                                        conflict_risk=conflict_risk)
    
    # Apply growth decay, plateau, and light negative progression for Asian and African cities
    if is_asian_or_african(location):
        # Apply parabolic decay of growth rate, then plateau, then light decline
        predictions = apply_growth_decay_with_plateau(predictions)
    
    # Store predictions
    for i, year in enumerate(range(2051, 2126)):
        col_name = f'Pop_{year}'
        if col_name not in new_columns:
            new_columns[col_name] = []
        new_columns[col_name].append(round(predictions[i], 0))

# Add new columns to dataframe
print("\nAdding predicted columns to dataframe...")
for col_name, values in new_columns.items():
    df[col_name] = values

# Save to metro_after.csv
print("\nSaving to metro_after.csv...")
df.to_csv('metro_after.csv', index=False)

print(f"\nDone! Added {len(new_columns)} new columns (Pop_2051 through Pop_2125)")
print(f"Total columns: {len(df.columns)}")
print(f"Total rows: {len(df)}")
print("\nSample of new predictions:")
sample_cols = ['City_Name', 'Location', 'Pop_2050', 'Pop_2060', 'Pop_2100', 'Pop_2125']
if all(col in df.columns for col in sample_cols):
    print(df[sample_cols].head(10).to_string())

