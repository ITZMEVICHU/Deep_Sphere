import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.DataFrame({
    'earnings': [40000, 50000, 60000, 75000, 80000, 90000, 100000, 120000, 140000, 160000],
    'earning_potential': [60000, 75000, 80000, 90000, 95000, 110000, 120000, 140000, 160000, 180000],
    'spending_limit': [500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800]
})

# Define features (X) and target variable (y)
X = data[['earnings', 'earning_potential']]
y = data['spending_limit']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Now, you can use this model for predictions.
# For example, to predict the spending limit for new data:
new_data = pd.DataFrame({'earnings': [80000], 'earning_potential': [95000]})
predicted_spending_limit = model.predict(new_data)
print(f"Predicted Spending Limit: {predicted_spending_limit[0]}")



import pickle

pkl_file_path = 'model.pkl'

# Serialize and save the dictionary to a PKL file
with open(pkl_file_path, 'wb') as pkl_file:
    pickle.dump(data, pkl_file)

print(f"PKL file '{pkl_file_path}' has been created.")

