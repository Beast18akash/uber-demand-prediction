import pandas as pd


def clean_uber_data(csv_files):
    """
    Load the Uber CSV files, combine them, remove duplicate records,
    and extract simple temporal features used in the analysis.
    """
    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    df = df.drop_duplicates().copy()

    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    df['Hour'] = df['Date/Time'].dt.hour
    df['Day'] = df['Date/Time'].dt.day
    df['DayOfWeek'] = df['Date/Time'].dt.dayofweek
    df['DayOfWeekName'] = df['Date/Time'].dt.day_name()
    df['Month'] = df['Date/Time'].dt.month

    return df


def prepare_ml_dataset(df):
    """
    Aggregate raw Uber pickup events into demand observations by time features.
    This creates the target variable Pickup_Count for regression.
    """
    demand_df = (
        df.groupby(['Hour', 'Day', 'DayOfWeek', 'Month'], as_index=False)
        .size()
        .rename(columns={'size': 'Pickup_Count'})
    )
    return demand_df
