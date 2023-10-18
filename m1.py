import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load your dataset
data = pd.read_csv("data.csv")

# Define your features (X) and target variable (y)
X = data[['earnings', 'earning_potential']]
y = data['spending_limit']

# Initialize and train a Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model as a .pkl file
joblib.dump(model, 'model.pkl')
