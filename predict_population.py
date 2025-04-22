import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from itertools import product
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Generate synthetic data
np.random.seed(42)
n_samples = 500
years = np.random.randint(2010, 2025, n_samples)
locations = np.random.choice(['London', 'Bristol', 'Birmingham', 'Manchester', 'Paris', 'Amsterdam', 'Brussels'], n_samples)

city_base_population = {
    'London': 4500000,
    'Bristol': 100000,
    'Birmingham': 250000,
    'Manchester': 150000,
    'Paris': 1000000,
    'Amsterdam': 475000,
    'Brussels': 200000
}

location_growth_rate = {
    'London': 1.04,
    'Bristol': 1.02,
    'Birmingham': 1.03,
    'Manchester': 1.035,
    'Paris': 1.05,
    'Amsterdam': 1.045,
    'Brussels': 1.03
}

base_population = np.array([city_base_population[loc] for loc in locations])
years_since_2010 = years - 2010
population = base_population * np.array([location_growth_rate[loc] ** years_since_2010[i] 
                                         for i, loc in enumerate(locations)])
population += np.random.normal(0, base_population * 0.01, n_samples)  # Add 1% noise

df = pd.DataFrame({
    'Year': years,
    'Location': locations,
    'Population': population.astype(int)
})

X = df[['Year', 'Location']]
y = df['Population']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(drop='first'), ['Location']),
    ('year_passthrough', 'passthrough', ['Year'])
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('features', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate model
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Improved Model R² Score: {r2:.4f}")
print(f"Improved Mean Squared Error: {mse:.2f}")

# Future predictions
years = list(range(2026, 2031))  # Next 5 years
locations = ['London', 'Bristol', 'Birmingham', 'Manchester', 'Paris', 'Amsterdam', 'Brussels']
combinations = list(product(years, locations))

future_df = pd.DataFrame(combinations, columns=['Year', 'Location'])
future_df_transformed = pipeline.named_steps['preprocessor'].transform(future_df)

# Future predictions using the entire pipeline (preprocessing and regression)
future_predictions = pipeline.predict(future_df)

# Assign predictions to the DataFrame
future_df['Predicted_Population'] = future_predictions

# Plotting
plt.figure(figsize=(12, 6))
for city in ['London', 'Paris', 'Manchester']:
    city_data = future_df[future_df['Location'] == city]
    plt.plot(city_data['Year'], city_data['Predicted_Population'], label=city)

plt.title('Population Growth Projections')
plt.xlabel('Year')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.savefig("output.png")
plt.show()


# Format and display results
pivot_results = future_df.pivot(index='Year', columns='Location', values='Predicted_Population')

print("\nPredicted Population (All Locations) in the next 5 Years:")
print(pivot_results.round(2))