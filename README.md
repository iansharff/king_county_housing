# *King County Housing: Predicting House Prices with Multiple Linear Regression*

Ian Sharff, Samantha Baltodano, and Sanjit Varma

## Table of Contents
* [Overview](#overview)
* [Business Understanding](#business-understanding)
* [Data Understanding](#data-understanding)
* [Data Preparation](#data-preparation)
* [Model Training and Testing](#model-training-and-testing)
* [Analysis and Conclusions](#analysis-and-conclusions)
* [Contributors](#contributors)
* [Project Structure](#project-structure)

## Repository Links
* [Data](/data)
* [Python scripts](/tools)
* [Images](/images)

## Overview

In this project, we outline our process of creating an effective and reasonably accurate model to predict sales prices for houses in and around Seattle, Washington. Using data from house sales in 2014 and 2015 from a total of 70 different zip codes, we were able to design and train several models with acceptable accuracies given their complexity and inherent limitations. Specifically, we saught to use multiple linear regression models via the `statsmodels` and `sklearn` libraries in Python, and we briefly explored the Gradient Boost Regressor to understand how it could be used to improve the quality of future analyses.

## Business Understanding

We framed our business problem from the perspective of Zillow, a company that relies on its ability to provide relevant and accurate "Zestimates" for houses listed to rent and to buy. These estimates are very often accurate in their depiction of a property's value; however, like most predictive models, there are always cases when it <a href=https://onesouthrealty.com/zillow-sued-over-zestimates-and-we-all-rejoiced>misses the mark</a>. From this (hypothetical) standpoint, our goal is to find attributes of a home that should be incorporated into the Zestimate model to minimize the harmful consequences of innacurate predictions on homeowners. By no means is this a comprehensive solution; rather, we hope to highlight the features available to us in the data that were the most indicative of a property's sale price. In addition, we seek to find ways in which feature engineering can play a role in fine-tuning the accuracy of our models and the ramifications it could have on improving the Zillow Zestimate.

## Data Understanding

The data provided to us consist of information pertaining to over 20,000 house sales carried out between 2014 and 2015, located in the `data/kc_house_data.csv` file in this repository along with the `data/column_names.md` data dictionary summarizing the information contained in each of the 19 relevant features, not inluding the `id` column which can be discarded for our purposes. To

## Data Preparation

## Model Training and Testing

## Analysis and Conclusions

## Contributors
- Samantha Baltodano <br>
    Github: sbaltodano<br>
- Sanjit Varma <br>
    Github: sanjitva<br>
- Ian Sharff <br>
    Github: iansharff<br>

## Project Structure
```
├── P2_Project.ipynb
├── README.md
├── data
├── images
└── tools
    ├── __init__.py
    ├── data_preparation.py
    ├── data_visualization.py
    └── helpers.py
```
