import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

import scipy.stats as st
import statsmodels as sm

import folium
import plotly.express as px

data = pd.read_csv('data/kc_house_data.csv')
data.drop('id', axis=1, inplace=True)
data.reset_index(drop=True, inplace=True)
data['date'] = pd.to_datetime(data['date'])

by_zip = data.groupby('zipcode')
avg_prices_by_zip = (
    by_zip['price']
        .agg(np.mean)
        .rename('average_price')
        .sort_values()
)

with open('data/king_county_zipcodes.geojson') as f:
    zipcode_borders = json.load(f)
data_zips = data.zipcode.unique()


new_features = [feature for feature in zipcode_borders['features'] if feature['properties']['ZIP'] in data_zips]

zipcode_borders['features'] = new_features

fig = px.choropleth(
    avg_prices_by_zip,
    geojson=zipcode_borders,
    locations=avg_prices_by_zip.index,
    featureidkey='properties.ZIP',
    color='average_price',
    scope='usa',
    color_continuous_scale='Viridis',
    range_color=(avg_prices_by_zip.min(), avg_prices_by_zip.max()))
fig.update_geos(fitbounds='locations',)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()
