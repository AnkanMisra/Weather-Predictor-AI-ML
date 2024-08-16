import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset
train_df = pd.read_csv('Test Data/train_preprocessed.csv')

# Handling Missing Values
# Fill missing values with median for numerical features
for col in train_df.select_dtypes(include=[np.number]).columns:
    train_df[col] = train_df[col].fillna(train_df[col].median())

# Removing Irrelevant Features
train_df.drop(columns=['Unnamed: 0'], inplace=True)

# Convert 'DATE' to datetime
train_df['DATE'] = pd.to_datetime(train_df['DATE'], dayfirst=True)

# Drop the 'DATE' column as it's not relevant for correlation and model building
train_df.drop(columns=['DATE'], inplace=True)

# Feature Engineering
train_df['Temp_Range_A'] = train_df['TMAX_A'] - train_df['TMIN_A']
train_df['Avg_Temp_BC'] = train_df[['TAVG_B', 'TAVG_C']].mean(axis=1)

# Outlier Handling (Example of logarithmic transformation)
train_df['PRCP_A_log'] = np.log1p(train_df['PRCP_A'])

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = train_df.select_dtypes(include=[np.number]).columns
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])

# Correlation Analysis
corr_matrix = train_df.corr()

# Drop highly correlated features (Example: Any correlation > 0.9)
high_corr_features = [column for column in corr_matrix.columns if any(corr_matrix[column] > 0.9) and column != 'TAVG']
train_df.drop(columns=high_corr_features, inplace=True)

# Save the processed data to the 'Feng Data' folder
train_df.to_csv('Feng Data/train_feature_engineered.csv', index=False)