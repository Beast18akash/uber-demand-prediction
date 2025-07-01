import pandas as pd

def clean_uber_data(csv_files):
    """
    Loads and cleans Uber CSV files, returning a combined DataFrame with extracted time features.
    """
    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    df['DayOfWeek'] = df['Date/Time'].dt.day_name()
    df['Hour'] = df['Date/Time'].dt.hour
    df['Day'] = df['Date/Time'].dt.day
    df['Month'] = df['Date/Time'].dt.month
    return df
