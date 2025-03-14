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
# print(combined_data.shape)
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
# print(combined_data[normalized_columns].head(100))
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
print("step-visualization Done")

# combined_data['localminute'] = pd.to_datetime(combined_data['localminute'])
# print(combined_data['Hour'].head(1))
# print(combined_data['Month'].head(1))
# print(combined_data['Day'].head(1))

# Normalize numerical columns (optional)
# scaler = MinMaxScaler()
# normalized_columns = ['use', 'gen', 'grid', 'DHI', 'DNI', 'GHI']
# combined_data[normalized_columns] = scaler.fit_transform(combined_data[normalized_columns])


# Define features and target variables for prediction
X = combined_data[['Hour', 'Day', 'Month', 'Temperature', 'DHI', 'DNI', 'GHI']]
y_use = combined_data['use']
y_gen = combined_data['gen']

print('selection of features and target variables done')

# Split data into training and testing sets
X_train_use, X_test_use, y_train_use, y_test_use = train_test_split(X, y_use, test_size=0.2, random_state=42)
X_train_gen, X_test_gen, y_train_gen, y_test_gen = train_test_split(X, y_gen, test_size=0.2, random_state=42)
print('creatng testing and training sets done')
# Train models (Random Forest example)
model_use = RandomForestRegressor(random_state=42)
model_gen = RandomForestRegressor(random_state=42)
print('creating models done')
model_use.fit(X_train_use, y_train_use)
model_gen.fit(X_train_gen, y_train_gen)
print('training models done')