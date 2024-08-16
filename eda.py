import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid")

# Load the dataset
train_df = pd.read_csv('Test Data/train_preprocessed.csv')

# 1. Basic Descriptive Statistics
print("Summary Statistics:")
print(train_df.describe())

# 2. Visualizing the Data

# Histograms with improved styling
train_df.hist(bins=20, figsize=(14, 10), color='skyblue', edgecolor='black', grid=False)
plt.suptitle('Distribution of Features', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Boxplots with better aesthetics
plt.figure(figsize=(12, 8))
sns.boxplot(data=train_df, palette="Set3", linewidth=1.5)
plt.xticks(rotation=90, fontsize=12)
plt.title('Boxplots of Features', fontsize=16)
plt.show()

# 3. Time Series Analysis
# Assuming DATE is parsed as datetime
train_df['DATE'] = pd.to_datetime(train_df['DATE'], dayfirst=True)

plt.figure(figsize=(12, 8))
plt.plot(train_df['DATE'], train_df['TAVG'], color='teal', linewidth=2)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Average Temperature (City D)', fontsize=14)
plt.title('Average Temperature Over Time', fontsize=16)
plt.grid(True)
plt.show()

# 4. Correlation Analysis

# Correlation Matrix
corr_matrix = train_df.corr()
print("Correlation Matrix:")
print(corr_matrix)

# Heatmap of the Correlation Matrix with better color map
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap', fontsize=16)
plt.show()

# 5. Scatter Plots

# Scatter plot for Elevation vs. Average Temperature
plt.figure(figsize=(10, 8))
sns.scatterplot(x='ELEVATION_A', y='TAVG', data=train_df, color='dodgerblue', s=100, edgecolor='black')
plt.xlabel('Elevation A', fontsize=14)
plt.ylabel('Average Temperature (City D)', fontsize=14)
plt.title('Elevation A vs Average Temperature (City D)', fontsize=16)
plt.grid(True)
plt.show()

# Pair Plot with enhanced aesthetics
sns.pairplot(train_df[['TAVG_A', 'TMAX_A', 'TMIN_A', 'ELEVATION_A', 'TAVG']], 
             diag_kind='kde', plot_kws={'edgecolor':'k', 's':100}, 
             palette='husl')
plt.suptitle('Pairplot of Selected Features', y=1.02, fontsize=16)
plt.show()

# 6. Missing Data Visualization

plt.figure(figsize=(12, 8))
sns.heatmap(train_df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Data Heatmap', fontsize=16)
plt.show()

# 7. Outlier Detection using Boxplots

plt.figure(figsize=(10, 8))
sns.boxplot(x=train_df['TAVG'], color='lightcoral', linewidth=1.5)
plt.title('Boxplot of Average Temperature (City D)', fontsize=16)
plt.show()