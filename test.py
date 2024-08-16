import os
import pandas as pd

# Ensure 'Test Data' folder exists
os.makedirs('Test Data', exist_ok=True)

# Update the paths to point to the correct directory
train_df = pd.read_csv('Data Raw/train.csv')
test_df = pd.read_csv('Data Raw/test.csv')
sample_submission_df = pd.read_csv('Data Raw/sample_submission.csv')

# Step 2: Display the first few rows of the training data
print("First few rows of the training data:")
print(train_df.head())

# Step 3: Handle missing values
# Impute missing values for temperature-related columns with the median
for column in ['TMAX_A', 'TMIN_A', 'TAVG_A', 'TAVG_B', 'TMAX_C', 'TMIN_C', 'TAVG_C']:
    train_df[column] = train_df[column].fillna(train_df[column].median())

# Impute missing values for precipitation-related columns with zero
train_df['PRCP_A'] = train_df['PRCP_A'].fillna(0)
train_df['PRCP_C'] = train_df['PRCP_C'].fillna(0)

# Step 4: Verify if all missing values are handled
missing_values_after_imputation = train_df.isnull().sum()
remaining_missing_values = missing_values_after_imputation[missing_values_after_imputation > 0]

if remaining_missing_values.empty:
    print("All missing values have been handled.")
else:
    print("Remaining missing values after imputation:")
    print(remaining_missing_values)

# Step 5: Basic Data Exploration (Optional)
print("\nBasic Statistics of the Training Data:")
print(train_df.describe())

# Step 6: Save the preprocessed data to the 'Test Data' folder
train_df.to_csv('Test Data/train_preprocessed.csv', index=False)
test_df.to_csv('Test Data/test_preprocessed.csv', index=False)

print("Preprocessing complete. Preprocessed data saved to 'Test Data/train_preprocessed.csv' and 'Test Data/test_preprocessed.csv'.")