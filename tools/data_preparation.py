import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from tools.helpers import remove_outliers, mode_fill


def initial_clean(df):
    """Cast to numeric types, fill NaN values with the feature mode"""
    # Drop 'id' column
    final_df = df
    final_df.drop('id', axis=1, inplace=True)

    # Convert 'date' to pd.datetime
    final_df['date'] = pd.to_datetime(final_df['date'])

    # Convert 'sqft_basement' to float, coercing the '?' to NaN values
    final_df['sqft_basement'] = pd.to_numeric(final_df['sqft_basement'], errors='coerce')

    # Mode fill NaN values in 'waterfront', 'yr_renovated', and 'view'
    for col in ['waterfront', 'yr_renovated', 'view', 'sqft_basement']:
        mode_fill(final_df, col)

    return final_df
