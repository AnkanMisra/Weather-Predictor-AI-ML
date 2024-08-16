import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Load the preprocessed test dataset
test_df = pd.read_csv('Test Data/test_preprocessed.csv')

# Load the trained model
model = joblib.load('model/final_model.pkl')

# Ensure the test data has the same features as the training data
# Load the training data to get the list of features
train_df = pd.read_csv('Feng Data/train_feature_engineered.csv')

# Get the features used in training
train_features = train_df.drop(columns=['TAVG']).columns

# Align the test data with the training features
X_test = test_df[train_features]

# Make predictions on the test set
predictions = model.predict(X_test)

# Check if 'TAVG' is in test data for evaluation
if 'TAVG' in test_df.columns:
    mse = mean_squared_error(test_df['TAVG'], predictions)
    r2 = r2_score(test_df['TAVG'], predictions)
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'R-squared (R2): {r2}')
else:
    print("Target variable 'TAVG' not found in the test dataset. Skipping evaluation metrics.")

# Save the predictions to a CSV file
output_df = pd.DataFrame({'Predicted': predictions})
output_df.to_csv('Final Result Data/test_predictions.csv', index=False)

print("Evaluation complete. Predictions saved to 'Final Result Data/test_predictions.csv'.")