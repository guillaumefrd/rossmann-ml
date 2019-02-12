"""
    This file is made to pre-process the data so that it can be fed to a machine learning algorithm.
    - data pre-processing
    - feature extraction
"""
import pandas as pd
import os
import numpy as np
import math
import pickle

from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest


def __load_data__(type):
    """
    Load data into Pandas dataframes.
    """
    if type == 'train':
        df = pd.read_csv(os.path.join('data', 'train.csv'))
    elif type == 'test':
        df = pd.read_csv(os.path.join('data', 'test.csv'))
    else:
        raise ValueError('dataset type must be "train" or "test".')

    # merge dataframes to get store information in both training and testing sets.
    df_store = pd.read_csv(os.path.join('data', 'store.csv'))
    df = df.merge(df_store, on='Store')

    print('Loaded tables in Pandas dataframes.')
    return df


def __handle_categorical__(df):
    """
    Handle categorical variables and encode them.
    :param df: Pandas dataframe containing categorical variables to deal with.
    """
    # define a dict where the keys are the element to encode, and the values are the targets
    encodings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
    # now we can encode the categorical features
    df['StateHoliday'].replace(encodings, inplace=True)
    df['StoreType'].replace(encodings, inplace=True)
    df['Assortment'].replace(encodings, inplace=True)

    print('Handled categorical features.')
    return df


def __preprocess_dates__(df):
    """
    Extract information from the date such as the month, day of the month...
    :param df: Pandas dataframe with a date column that we want to deal with.
    """

    def __get_months_since_competition__(year, competition_open_since_year, month, competition_open_since_month):
        if competition_open_since_year > 0 and competition_open_since_month > 0:
            return 12 * (year - competition_open_since_year) + (month - competition_open_since_month)
        else:
            return 0

    def __get_month_since_promo__(year, promo2_since_year, week_of_year, promo2_since_week):
        if promo2_since_week > 0 and promo2_since_year > 0:
            return 12 * (year - promo2_since_year) + (week_of_year - promo2_since_week) / 4.

    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfMonth'] = df['Date'].apply(lambda x: x.day)
    df['Month'] = df['Date'].apply(lambda x: x.month)
    df['Year'] = df['Date'].apply(lambda x: x.year)
    df['DayOfWeek'] = df['Date'].apply(lambda x: x.dayofweek)
    df['WeekOfYear'] = df['Date'].apply(lambda x: x.weekofyear)
    df.drop(['Date'], inplace=True, axis=1)

    # number of months since a competition store has opened
    df['MonthsSinceCompetition'] = df.apply(
        lambda row: __get_months_since_competition__(row['Year'], row['CompetitionOpenSinceYear'], row['Month'],
                                                 row['CompetitionOpenSinceMonth']), axis=1)
    df.drop(['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'], inplace=True, axis=1)

    # number of months since a promotion has started
    df['MonthsSincePromo'] = df.apply(
        lambda row: __get_month_since_promo__(row['Year'], row['Promo2SinceYear'], row['WeekOfYear'],
                                              row['Promo2SinceWeek']), axis=1)
    df['MonthsSincePromo'] = df['MonthsSincePromo'].apply(lambda x: x if x > 0 else 0)
    df.drop(['Promo2SinceYear', 'Promo2SinceWeek', 'PromoInterval'], inplace=True, axis=1)

    print("Added columns: 'DayOfMonth', 'Month', 'Year', 'DayOfWeek', 'WeekOfYear', 'MonthsSinceCompetition', "
          "'MonthsSincePromo'.")
    print("Removed columns: 'Date', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'Promo2SinceYear', "
          "'Promo2SinceWeek'")

    return df


def build_dataset(type):
    df = __load_data__(type)
    df = __handle_categorical__(df)
    df = __preprocess_dates__(df)
    print(df.info())
    return df


def save_dataset(df, filename):
    path = os.path.join('data', filename)
    df.to_csv(path, index=False)


class Preprocessor:
    def __init__(self):
        self.data_stats = {}
        pass

    def fit(self, data):
        """
        We suppose that the train and test data come from the same distribution, so we need to fit the preprocessing
        on the train data and later transform the test data with respect to the distribution of the train data.
        :param data: (Pandas dataframe) training set
        """

        print('Fitting data...')
        # save the mean of this column for transform
        self.data_stats['MonthsSinceCompetitionMean'] = math.floor(data['MonthsSinceCompetition'].mean())

        # save data_stats to pickle file.
        # this file will be necessary when preprocessing test data
        print('Saving data stats to pickle file...')
        data_stats_file = open('data_stats.pkl', 'wb')
        pickle.dump(self.data_stats, data_stats_file)
        data_stats_file.close()
        print('Fitting data done.')

    def transform(self, data):
        """
        Same operations applied as in the train transformation except when mean is used: should use the mean values
        used from the train set.
        :param data: dataset to transform
        """

        # if object has not been fit prior to transform call, load data stats from pickle file
        if not self.data_stats:
            print('Loading data stats from pickle file...')
            data_stats_file = open('data_stats.pkl', 'rb')
            self.data_stats = pickle.load(data_stats_file)
            data_stats_file.close()

        print('Transforming data...')
        mask1 = (data['MonthsSinceCompetition'].isnull())

        # fill missing values with mean
        data.loc[mask1, 'MonthsSinceCompetition'] = data.loc[mask1, 'MonthsSinceCompetition'].fillna(
            self.data_stats['MonthsSinceCompetitionMean'])

        data['CompetitionDistance'].fillna(0, inplace=True)

        # Fill with 0 promo2 variables
        data['MonthsSincePromo'].fillna(0, inplace=True)

        data['Open'].fillna(0, inplace=True)

        print('Null columns:')
        for column in data.columns.tolist():
            if data[column].isnull().any():
                print(column)
        print('0 columns:')
        for column in data.columns.tolist():
            if 0 in data[column].unique():
                print(column)

        # scale target ('Sales')
        if 'Sales' in data.columns.tolist():
            print('Scaling sales values...')
            data['Sales'] = data['Sales'].apply(lambda x: np.log(x) if x > 0 else 0)

        print('Transforming data done.')

        print(data.info())
        return data


def __get_most_important_features__(X_train, y_train, n_features):
    """
    Perform feature selection using XGBoost algorithm
    """
    model = XGBClassifier()
    model.fit(X_train, y_train)
    sorted_idx = np.argsort(model.feature_importances_)[::-1]
    sorted_idx = sorted_idx[:n_features]
    return X_train.columns[sorted_idx]


def __outlier_detection__(df):
    """
    Perform outlier detection using IsolationForest algorithm
    """

    n_samples = df.shape[0]
    outliers_fraction = 0.15
    n_outliers = int(outliers_fraction * n_samples)
    model = IsolationForest(behaviour='new',
                            contamination=outliers_fraction,
                            random_state=42)
    # contains the anormality score for each sample
    # the lower, the more abnormal.
    y_pred = model.fit(df).score_samples(df)
    outliers_indx = np.argsort(y_pred)[::-1]
    return y_pred, outliers_indx


# unit test
if __name__ == "__main__":
    data = build_dataset('train')
    preprocessor = Preprocessor()
    preprocessor.fit(data)
    data = preprocessor.transform(data)
