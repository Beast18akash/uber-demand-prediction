"""
Uber NYC Data Analysis and Prediction
This script analyzes historical Uber pickup data and builds a simple
Random Forest Regression model to predict pickup demand using temporal features.
"""

import os
import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from scripts.clean_data import clean_uber_data, prepare_ml_dataset
from scripts.visualization import (
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_heatmap_day_hour,
    plot_monthly_pickups,
    plot_pickups_by_day,
    plot_pickups_by_dayofweek,
    plot_pickups_by_hour,
)


# Paths
project_dir = Path(__file__).resolve().parent
output_dir = project_dir / 'output'
output_dir.mkdir(exist_ok=True)

data_path = project_dir / 'data'
csv_files = sorted(data_path.glob('uber-raw-data-*.csv'))

# Load and clean data
raw_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
df = clean_uber_data(csv_files)
duplicate_count = len(raw_df) - len(df)

print('Cleaned dataset rows:', len(df))
print('Duplicate rows removed:', duplicate_count)
print('Date range:', df['Date/Time'].min(), 'to', df['Date/Time'].max())
print('Columns:', list(df.columns))
print('Missing values:\n', df.isnull().sum())

# Exploratory plots
plot_pickups_by_hour(df, save_path=str(output_dir / 'pickups_by_hour.png'))
plot_pickups_by_day(df, save_path=str(output_dir / 'pickups_by_day.png'))
plot_pickups_by_dayofweek(df, save_path=str(output_dir / 'pickups_by_dayofweek.png'))
plot_monthly_pickups(df, save_path=str(output_dir / 'pickups_by_month.png'))
plot_heatmap_day_hour(df, save_path=str(output_dir / 'heatmap_day_hour.png'))

# Descriptive statistics
print('\nDescriptive statistics for pickup count:')
ml_df = prepare_ml_dataset(df)
print(ml_df['Pickup_Count'].describe().to_string())
print('\nCorrelation matrix:')
print(ml_df[['Hour', 'Day', 'DayOfWeek', 'Pickup_Count']].corr().round(3).to_string())

# Prepare training data
X = ml_df[['Hour', 'Day', 'DayOfWeek', 'Month']]
y = ml_df['Pickup_Count']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print('\nModel Evaluation:')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R^2: {r2:.4f}')

# Actual vs predicted plot
plot_actual_vs_predicted(y_test, predictions, save_path=str(output_dir / 'actual_vs_predicted.png'))
plot_feature_importance(
    feature_names=['Hour', 'Day', 'DayOfWeek', 'Month'],
    importances=model.feature_importances_,
    save_path=str(output_dir / 'feature_importance.png'),
)

print('\nModel feature importance:')
for name, score in zip(X.columns, model.feature_importances_):
    print(f'{name}: {score:.4f}')

# Interactive prediction section
print('\nPrediction Input:')

while True:
    try:
        hour = int(input('Enter Hour (0-23): '))
        if not 0 <= hour <= 23:
            raise ValueError('Hour must be between 0 and 23.')

        day = int(input('Enter Day (1-31): '))
        if not 1 <= day <= 31:
            raise ValueError('Day must be between 1 and 31.')

        day_of_week = int(input('Enter DayOfWeek (0-6): '))
        if not 0 <= day_of_week <= 6:
            raise ValueError('DayOfWeek must be between 0 and 6.')

        month = int(input('Enter Month (1-12): '))
        if not 1 <= month <= 12:
            raise ValueError('Month must be between 1 and 12.')

        break
    except ValueError as error:
        print(f'Error: {error} Please enter valid values.')
    except EOFError:
        print('\nInput ended before valid values were entered. Please run the script again.')
        raise SystemExit
    except KeyboardInterrupt:
        print('\nInput cancelled by user.')
        raise SystemExit

user_input = pd.DataFrame({
    'Hour': [hour],
    'Day': [day],
    'DayOfWeek': [day_of_week],
    'Month': [month]
})

predicted_pickup = model.predict(user_input)[0]
print('\nPrediction Result:')
print(f'Hour = {hour}')
print(f'Day = {day}')
print(f'DayOfWeek = {day_of_week}')
print(f'Month = {month}')
print(f'Predicted Pickup_Count = {predicted_pickup:.2f}')

print('\nAnalysis complete. Outputs saved in the output folder.')
