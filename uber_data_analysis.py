"""
Uber NYC Data Analysis
This script loads, cleans, and visualizes Uber pickup data for business insights.
"""

import os
import glob
from scripts.clean_data import clean_uber_data
from scripts.visualization import plot_pickups_by_hour, plot_pickups_by_dayofweek, plot_heatmap_day_hour

# Path to data files
data_path = os.path.join(os.path.dirname(__file__), 'data')
csv_files = glob.glob(os.path.join(data_path, 'uber-raw-data-*.csv'))

# Load and clean data
df = clean_uber_data(csv_files)

# Basic info
print('Total records:', len(df))
print(df.head())


# Visualizations (save plots in the output folder)
output_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(output_dir, exist_ok=True)
plot_pickups_by_hour(df, save_path=os.path.join(output_dir, 'pickups_by_hour.png'))
plot_pickups_by_dayofweek(df, save_path=os.path.join(output_dir, 'pickups_by_dayofweek.png'))
plot_heatmap_day_hour(df, save_path=os.path.join(output_dir, 'heatmap_day_hour.png'))

print('Analysis complete. Visualizations saved as PNG files.')
