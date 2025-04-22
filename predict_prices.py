import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from itertools import product

# Generate synthetic data (unchanged)
np.random.seed(42)
n_samples = 500
years = np.random.randint(2010, 2025, n_samples)
locations = np.random.choice(['London', 'Manchester', 'Birmingham', 'Leeds', 'Bristol'], n_samples)
size_sqft = np.random.randint(500, 2500, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)


# Define city-specific base prices
city_base_prices = {
    'London': 10000,
    'Manchester': 8000,
    'Birmingham': 7500,
    'Leeds': 7000,
    'Bristol': 8500
}

# Calculate base price based on city
base_price = np.array([city_base_prices[loc] for loc in locations]) + (size_sqft * 150) + (bedrooms * 10000) + (bathrooms * 5000)

location_multiplier = {
    'London': 2.0,
    'Manchester': 1.3, 
    'Birmingham': 1.2,
    'Leeds': 1.1,
    'Bristol': 1.4
}
location_factor = np.array([location_multiplier[loc] for loc in locations])
year_factor = (years - 2010) * 0.04 + 1
price = base_price * location_factor * year_factor + np.random.normal(0, 20000, n_samples)

df = pd.DataFrame({
    'year': years,
    'location': locations,
    'size_sqft': size_sqft,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'price': price.astype(int)
})

# Prepare features and target
X = df[['year', 'location', 'size_sqft', 'bathrooms', 'bedrooms']]
y = df['price']

# Enhanced preprocessing with interaction terms
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(drop='first'), ['location'])
], remainder='passthrough')

# Updated pipeline with interaction features
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('interactions', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('regressor', LinearRegression())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate model
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Improved Model R² Score: {r2:.4f}")
print(f"Improved Mean Squared Error: {mse:.2f}")

# Corrected future data generation
years = list(range(2026, 2031))  # Next 5 years
locations = ['London', 'Manchester', 'Birmingham', 'Leeds', 'Bristol']
combinations = list(product(years, locations))

future_df = pd.DataFrame(combinations, columns=['year', 'location'])
future_df['size_sqft'] = 750 * (1.02 ** (future_df['year'] - 2025))  # Compound growth
future_df['bedrooms'] = 2 + ((future_df['year'] - 2025) % 2)  # Alternating pattern
future_df['bathrooms'] = 1 + ((future_df['year'] - 2025) % 3)  # Alternating pattern

# Make predictions
future_predictions = pipeline.predict(future_df)

# Format and display results
future_df['predicted_price'] = future_predictions
pivot_results = future_df.pivot(index='year', columns='location', values='predicted_price')

print("\nPredicted Prices (All Locations) in the next 5 Years:")
print(pivot_results.round(2))