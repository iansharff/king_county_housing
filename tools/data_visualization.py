import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tools.helpers import display_percent_nan


# DIAGNOSTIC VISUALIZATIONS
def null_heatmap(df, cmap='Greens_r'):
    fig, ax = plt.subplots(figsize=(15, 7))
    # fig = sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap=cmap, ax=ax)
    # return fig

    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap=cmap, ax=ax)
    plt.show()


def corr_heatmap(df):
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(15, 15))

    ax.set_title('Multicollinearity of Features')
    sns.heatmap(np.abs(corr), mask=mask, square=True, annot=True, ax=ax, robust=True, cmap='mako')
    plt.show()
    # return sns.heatmap(np.abs(corr), mask=mask, square=True, annot=True, ax=ax, robust=True, cmap='mako');


if __name__ == '__main__':
    pass
