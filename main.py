# import pandas as pd
# import os
#
# # Define the path to the dataset
# file_path = 'Walmart_DataSet.csv'  # Update this to your actual dataset filename
#
# # Check if the file exists
# if not os.path.exists(file_path):
#     raise FileNotFoundError(f"The dataset file '{file_path}' was not found.")
#
# # Load the dataset
# try:
#     df = pd.read_csv(file_path)
# except Exception as e:
#     raise RuntimeError(f"Failed to load the dataset: {e}")
#
# # Strip whitespace from column names
# df.columns = df.columns.str.strip()
#
# # Check if 'Date' column exists
# if 'Date' not in df.columns:
#     raise KeyError("The 'Date' column is missing from the dataset.")
#
# # Convert 'Date' column to datetime format
# # Convert 'Date' column to datetime format
# try:
#     df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
#     if df['Date'].isnull().any():
#         print("Warning: Some dates could not be parsed and were set to NaT.")
# except Exception as e:
#     raise ValueError(f"Failed to convert 'Date' column to datetime: {e}")
#
# # Basic preprocessing: handle missing values
# df.fillna(method='ffill', inplace=True)
#
# # Print basic info about the dataset
# print("Dataset loaded and preprocessed successfully.")
# print(df.info())
# print(df.head())

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
df = pd.read_csv("Walmart_DataSet.csv")

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

# Create output directory if it doesn't exist
output_dir = 'code'
os.makedirs(output_dir, exist_ok=True)

# 1. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), format='png')
plt.close()

# 2. Holiday Impact on Weekly Sales
plt.figure(figsize=(8, 6))
sns.boxplot(x='Holiday_Flag', y='Weekly_Sales', data=df)
plt.title('Holiday Impact on Weekly Sales')
plt.xlabel('Holiday Flag (0 = No, 1 = Yes)')
plt.ylabel('Weekly Sales')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'holiday_impact.png'), format='png')
plt.close()

# 3. Seasonal Decomposition of Weekly Sales
weekly_sales = df.groupby('Date')['Weekly_Sales'].sum().sort_index()
weekly_sales = weekly_sales.asfreq('W', method='pad')  # Ensure weekly frequency

decomposition = seasonal_decompose(weekly_sales, model='additive', period=52)
fig = decomposition.plot()
fig.set_size_inches(10, 8)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'seasonal_decomposition.png'), format='png')
plt.close()

# 4. Weekly Sales Trend Over Time
plt.figure(figsize=(12, 6))
df_sorted = df.sort_values('Date')
sns.lineplot(x='Date', y='Weekly_Sales', data=df_sorted)
plt.title('Weekly Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'weekly_sales_trend.png'), format='png')
plt.close()

