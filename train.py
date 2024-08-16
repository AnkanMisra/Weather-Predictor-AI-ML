import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
train_df = pd.read_csv('Feng Data/train_feature_engineered.csv')

# Define features and target
X = train_df.drop(columns=['TAVG'])
y = train_df['TAVG']

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model/final_model.pkl')

# Evaluate on the validation set
predictions = model.predict(X_valid)
mse = mean_squared_error(y_valid, predictions)
print(f'Validation MSE: {mse}')