import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from tools.helpers import remove_outliers, mode_fill


def initial_clean(df):
    """
    This function carries out the initial cleaning process for a DataFrame
    """
    # Drop 'id' column
    final_df = df.copy()
    final_df.drop('id', axis=1, inplace=True)

    # Convert 'date' to pd.datetime
    final_df['date'] = pd.to_datetime(final_df['date'])

    # Convert 'sqft_basement' to float, coercing the '?' to NaN values
    final_df['sqft_basement'] = pd.to_numeric(final_df['sqft_basement'], errors='coerce')

    # Mode fill NaN values in 'waterfront', 'yr_renovated', and 'view'
    for col in ['waterfront', 'yr_renovated', 'view', 'sqft_basement']:
        mode_fill(final_df, col)

    # Drop significant outliers from continuous features
    cols_w_outliers = [
        'price',
        'sqft_living',
        'sqft_above',
        'sqft_living15',
        'bedrooms',
        'bathrooms'
    ]

    final_df = remove_outliers(final_df, col_names=cols_w_outliers, criteria='normal')

    add_distance(final_df)

    return final_df


def add_distance(df):
    """Add column of distance from the center of the zipcode with the highest average price"""
    # Make DataFrame of zipcodes with their average prices and coordinates
    zip_avg_price = df.groupby('zipcode')[['price', 'lat', 'long']].mean().reset_index()

    # Store lat and long of zipcode with highest price
    lat, long = zip_avg_price.sort_values('price', ascending=False).iloc[0, 2:]
    df['lat_cent'] = lat
    df['long_cent'] = long
    eval_formula = """
    dist_from_center = ((lat - lat_cent)**2 + (long - long_cent)**2) ** 0.5
    """
    df.eval(eval_formula, inplace=True)
    df.drop(['lat_cent', 'long_cent'], axis=1, inplace=True)
    return None


if __name__ == '__main__':
    df = initial_clean(pd.read_csv('../data/kc_house_data.csv'))
    add_distance(df)
    print(df.info())
    print(df.describe())
