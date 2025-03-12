import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Load all datasets into a single DataFrame
print("start")
file_names = [f'house{i}.csv' for i in range(1, 16)]
print("step-1 Done")
dataframes = [pd.read_csv(file) for file in file_names]
print("step-2 Done")
# Combine all datasets
combined_data = pd.concat(dataframes, ignore_index=True)
print("step-3 Done")
# Verify consolidation
# print(combined_data.head())
# print(combined_data.sort)
print("step-4 Done")

# Normalize numerical columns (optional)

# Define columns to normalize
normalized_columns = ['use', 'gen', 'grid', 'DHI', 'DNI', 'GHI']
scaler = MinMaxScaler()
print("step-5 Done")

# Apply normalization to the specified columns
combined_data[normalized_columns] = scaler.fit_transform(combined_data[normalized_columns])
print("step-6 Done")

# Verify normalization
print(combined_data[normalized_columns].head(100))
print("step-7 Done")



# Visualize energy consumption trends
# plt.figure(figsize=(12, 6))
# plt.plot(combined_data['localminute'], combined_data['use'], label='Energy Consumption')
# plt.plot(combined_data['localminute'], combined_data['gen'], label='Energy Generation')
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Energy (kWh)')
# plt.title('Energy Trends')
# plt.show()
