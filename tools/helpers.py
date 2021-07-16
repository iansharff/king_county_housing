"""
This module contains functions employed during the EDA process and for repeated processes during model testing. It is
used in the main Jupyter Notebook with the import statement 'import tools.helpers as th'.

CONTENTS:

Imports
I. Diagnostic Functions
II. Transformation Functions
"""

import numpy as np
import pandas as pd
import scipy.stats as stats

"""
I. DIAGNOSTIC FUNCTIONS

i. percent_nan
ii. display_percent_nan
iii. get_value_counts
iv. predictor_corrs
"""


def percent_nan(df):
    """Return a Series of percent NaN values in each column of a DataFrame"""
    nulls = df.isnull().sum()
    length = len(df.index)
    # Passed into the following function
    return nulls / length if length != 0 else None


def display_percent_nan(df):
    """Display formatted percent-NaN-values for each column of a DataFrame"""
    series = percent_nan(df)
    for column in series.index:
        print(f"{column} : {100 * series.at[column]:.2f} % null")


def get_value_counts(df, highest_two=False):
    """
    Display value counts for each column of a DataFrame

    Keyword arguments:
        highest_two -- bool, if True then only the two most frequent values are displayed
    """
    for col in df.columns:
        print(col, ':')
        counts = df[col].value_counts(dropna=False)
        if highest_two:
            print(counts.head(2), '\n')
        else:
            print(counts, '\n')


def predictor_corrs(X, cutoff=0.60):
    """
    Return a DataFrame of the highest pairwise correlations of features in X

    Keyword arguments:
        cutoff -- lower limit of the correlations displayed
    """

    # Find pairwise correlations and form a column of tuples of corresponding column names
    corrs = X.corr().abs().stack().reset_index().sort_values(0, ascending=False)
    corrs['pairs'] = list(zip(corrs.level_0, corrs.level_1))

    # Drop the redundant columns and set the index to the new 'pairs' column
    corrs.drop(['level_0', 'level_1'], axis=1, inplace=True)
    corrs.set_index('pairs', inplace=True)

    # Drop duplicates, only saving one of the two possible pairwise permutations.
    corrs.drop_duplicates(inplace=True)

    # Rename the column for clarity
    corrs.columns = ['r']
    high_cc = corrs[(corrs.r > cutoff) & (corrs.r < 1)]
    return high_cc


"""
II. TRANSFORMATION FUNCTIONS

i. log_transform
ii. exp_transform
iii. mode_fill
iv. normalize
v. remove_outliers
"""


def log_transform(data, col_names):
    """
    Transform selected columns inplace in a pd.DataFrame with natural logarithm

    Arguments:
        data -- pd.DataFrame
        col_names -- str/list, names of columns to transform
    """
    if isinstance(col_names, list):
        for col in col_names:
            data[col] = np.log(data[col])
    elif isinstance(col_names, str):
        data[col_names] = np.log(data[col_names])
    # Ensure that col_names is a string or list of strings
    else:
        raise TypeError("col_names must be a string or a list of strings")
    return None


def exp_transform(data, col_names):
    """
    Transform selected columns inplace in a pd.DataFrame with exponential function

    Arguments:
        data -- pd.DataFrame
        col_names -- str/list, name(s) of column(s) to transform
    """
    if isinstance(col_names, list):
        for col in col_names:
            data[col] = np.exp(data[col])
    elif isinstance(col_names, str):
        data[col_names] = np.exp(data[col_names])
    else:
        raise TypeError("col_names must be a string or a list of strings")
    return None


def mode_fill(df, column=None):
    """Fill column NaN values with the mode of the column"""
    if column:
        df[column].fillna(df[column].mode()[0], inplace=True)


def normalize(data, col_names):
    """
    Transform selected columns inplace in a pd.DataFrame with exponential function

    Arguments:
        data -- pd.DataFrame
        col_names -- str/list, name(s) of column(s) to transform
    """
    if isinstance(col_names, list):
        for col in col_names:
            data[col] = stats.zscore(data[col])
    elif isinstance(col_names, str):
        data[col_names] = stats.zscore(data[col_names])
    else:
        raise TypeError("col_names must be a list or a string")
    return None


def remove_outliers(data, col_names=None, criteria='normal'):
    """
    Remove outlier data points (rows) from a DataFrame according to a criteria

    Arguments:
        data: pd.DataFrame
        col_names: list or string of column names in data
        criteria: either 'normal' or 'iqr'
    """
    final = data
    # Ensure that cols is assigned a list
    cols = col_names if isinstance(col_names, list) else [col_names]

    if criteria == 'normal':
        for col in cols:
            # Remove all data points with zscore >= 3
            final = final[(np.abs(stats.zscore(final[col])) < 3)]
    if criteria == 'iqr':
        for col in cols:
            q1 = final[col].quantile(0.25)
            q3 = final[col].quantile(0.75)
            limit = 1.5 * (q3 - q1)
            mask = (final[col] > (q1 - limit)) | (final[col] < (q3 + limit))
            final = final[mask]
    return final


def MAPE(Y_actual, Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted) / Y_actual)) * 100
    return mape

if __name__ == '__main__':
    df = pd.read_csv('../data/kc_house_data.csv')
    test = remove_outliers(df, 'bedrooms', 'normal')
    print(test.describe().bedrooms)
    print(test.shape[0])
