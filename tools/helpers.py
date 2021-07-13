import numpy as np
import pandas as pd

def percent_nan(df):
    """Return a Series of percent NaN values in each column of a DataFrame"""
    nulls = df.isnull().sum()
    length = len(df.index)
    return nulls / length if length != 0 else None


def display_percent_nan(df):
    """Display formatted percent-NaN-values for each column of a DataFrame"""
    series = percent_nan(df)
    for column in series.index:
        print(f"{column} : {100 * series.at[column]:.2f} % null")


def remove_outliers(data, col_names=None, criteria='normal'):
    """
    Remove outlier data points (rows) from a DataFrame according to a criteria
    Parameters:
        data: pd.DataFrame
        col_names: list or string of column names in data
        criteria: either 'normal' or 'iqr'
    """
    final = data
    cols = col_names

    if isinstance(col_names, str):
        cols = [col_names]

    for col in cols:
        # Remove all data points with zscore >= 3
        final = final[(np.abs(stats.zscore(final[col])) < 3)]
    return final