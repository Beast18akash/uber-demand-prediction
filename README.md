
# Uber Pickup Demand Analysis and Prediction

## Problem Statement
Uber pickup demand changes across time. Temporal factors such as hour, day, day of the week, and month help explain when demand is high or low. This project analyzes historical Uber NYC pickup data and builds a machine learning model to predict pickup demand using these temporal features.

## Objective
The main objective is to study historical Uber pickup behavior using descriptive statistics and correlation analysis, then prepare a simple machine learning dataset and predict aggregated pickup demand with a Random Forest Regressor.

## Dataset
This project uses the Uber NYC monthly CSV datasets in the data folder:
- Files used: uber-raw-data-apr14.csv, uber-raw-data-jun14.csv, uber-raw-data-jul14.csv, uber-raw-data-aug14.csv, and uber-raw-data-sep14.csv
- Source: Uber NYC trip records from April through September 2014
- Raw records: 3,881,892
- Variables: 4
- Date range: 2014-04-01 to 2014-09-30
- Columns: Date/Time, Lat, Lon, Base
- Important attributes: Date/Time, Lat, Lon, Base

The raw data is a large set of individual trip pickup events. To make the regression problem simple and practical, the raw records are grouped by temporal features to form daily demand observations.

## Technologies Used
- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook

## Methodology
1. Load and combine the Uber CSV data.
2. Remove duplicate records.
3. Convert Date/Time into datetime format.
4. Extract time features such as Hour, Day, DayOfWeek, and Month.
5. Perform descriptive analysis and correlation analysis.
6. Aggregate data into pickup demand counts.
7. Train a Random Forest Regressor.
8. Evaluate the model using MAE, MSE, RMSE, and R².

## Data Preprocessing
The preprocessing steps include:
- Loading the CSV file from the dataset folder
- Combining the dataset into one DataFrame
- Converting Date/Time to datetime format
- Extracting Hour, Day, DayOfWeek, and Month
- Checking missing values
- Removing duplicate rows
- Validating the final dataset structure

The final cleaned dataset contains 3,809,386 records after removing 72,506 duplicates. The raw files contain no missing values in the four original columns.

## Exploratory Data Analysis
The project visualizes:
- Pickup count by hour
- Pickup count by day of month
- Pickup count by day of week
- Pickup count by month
- Heatmap of day vs hour

These plots help identify busy hours, weekly patterns, and time-based demand variation.

## Statistical Analysis
Descriptive statistics were computed on the aggregated demand dataset.

Pickup_Count summary statistics:
- Mean: 1044.53
- Median: 1003.00
- Standard deviation: 641.21
- Minimum: 45.00
- Maximum: 3347.00
- 25th percentile: 517.00
- 75th percentile: 1450.00

The correlation matrix shows a positive relationship between Hour and Pickup_Count (0.630). Day (0.048) and DayOfWeek (-0.020) have weaker correlations. Correlation does not demonstrate causation.

## Machine Learning Model
The project uses a single main model:
- Random Forest Regression
- Library: RandomForestRegressor from scikit-learn

The model predicts aggregated pickup demand using these features:
- Hour
- Day
- DayOfWeek
- Month

The target variable is:
- Pickup_Count

Random Forest Regression is suitable because the target is numerical and the model can learn nonlinear relationships in temporal demand patterns without requiring complicated mathematical assumptions.

## Evaluation Metrics
The model was evaluated on the test set using:
- MAE: 84.06
- MSE: 16528.62
- RMSE: 128.56
- R²: 0.9607

## Results
The model explains about 96% of the variation in the test set. These results are specific to the current dataset and random train-test split.

The most important feature was Hour, followed by DayOfWeek, Month, and Day. Feature importance describes how the model used these inputs; it does not imply causation.

## Ethical Considerations
- Privacy: location-based transportation data should be handled responsibly.
- Bias: historical data may reflect limited time periods and locations.
- Fairness: the model may not represent all neighborhoods equally.
- Transparency: predictions are estimates and should not be treated as absolute truth.
- Responsible use: this model is intended for analysis and planning, not blind decision-making.

## Limitations
- The dataset is historical and covers selected months from April through September 2014.
- There are no weather or traffic variables.
- Holidays, events, and surge pricing are not included.
- Only a small set of temporal features is used.
- Data quality depends on the original records and duplicate handling.

## Future Scope
Possible improvements include:
- Weather data
- Traffic conditions
- Holiday indicators
- Event calendars
- More recent data
- Additional location-based features
- Testing other regression models
- Hyperparameter tuning

## How to Run
1. Open the project folder.
2. Create and activate a virtual environment if needed.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the main script:
   ```
   python uber_data_analysis.py
   ```
5. Open the notebook in the notebook folder to view the step-by-step analysis.

## Project Structure
```text
uber_data_analysis/
├── data/
│   ├── uber-raw-data-apr14.csv
│   ├── uber-raw-data-jun14.csv
│   ├── uber-raw-data-jul14.csv
│   ├── uber-raw-data-aug14.csv
│   └── uber-raw-data-sep14.csv
├── notebook/
│   └── uber_data_analysis.ipynb
├── output/
│   ├── pickups_by_hour.png
│   ├── pickups_by_day.png
│   ├── pickups_by_dayofweek.png
│   ├── pickups_by_month.png
│   ├── heatmap_day_hour.png
│   ├── actual_vs_predicted.png
│   └── feature_importance.png
├── scripts/
│   ├── clean_data.py
│   └── visualization.py
├── README.md
├── requirements.txt
├── uber_data_analysis.ipynb
├── uber_data_analysis.py
```

## Final Note
This project is a simple and student-friendly Experiment 10 implementation focused on Uber pickup demand analysis, statistical understanding, and a single Random Forest Regression model for prediction.
