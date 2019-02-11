"""
    This file is made to pre-process the data so that it can be fed to a machine learning algorithm.
    - data pre-processing
    - feature extraction
"""
import pandas as pd
import os
import numpy as np
import math

from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self, folder_path):
        """
        Initialize class instance.
        :param folder_path: (string) folder containing the csv files.
        """
        self.df_store, self.df_train, self.df_valid = None, None, None
        self.folder_path = folder_path

    def load_data(self):
        """
        Load data into Pandas dataframes.
        """
        self.df_store = pd.read_csv(os.path.join(self.folder_path, 'store.csv'))
        self.df_train = pd.read_csv(os.path.join(self.folder_path, 'train.csv'))

        # merge dataframes to get store information in both training and testing sets.
        self.df_train = self.df_train.merge(self.df_store, on='Store')

        # split into training and validation set
        self.df_train, self.df_valid = train_test_split(self.df_train, test_size=0.1, random_state=42)

        print('Loaded tables in Pandas dataframes.\n')
        print('df_train shape: {} \ndf_valid shape: {}\n'.format(self.df_train.shape, self.df_valid.shape))

    @staticmethod
    def fit(df):
        """
        We suppose that the train and test data come from the same distribution, so we need to fit the preprocessing
        on the train data and later transform the test data with respect to the distribution of the train data.
        :param df: (Pandas dataframe) training set
        :return: fill_values: the values we used from the training set either to fill NaN of to generate new features.
        """
        fill_values = {}

        ###############
        # fill NaN values
        ###############
        df_new = df.copy()

        # Impute the columns 'CompetetionOpenSinceMonth' and 'CompetitionOpenSinceYear' with mean values in case
        # 'CompetitionDistance' is not missing => there is a competitor but the open date is unknown.
        mask1 = (df_new['CompetitionOpenSinceMonth'].isnull()) & (df_new['CompetitionDistance'].isnull() == False)

        # save the mean of this column for later usage on test set
        fill_values['CompetitionOpenSinceMonthMean'] = math.floor(df['CompetitionOpenSinceMonth'].mean())
        df_new.loc[mask1, 'CompetitionOpenSinceMonth'] = df_new.loc[mask1, 'CompetitionOpenSinceMonth'].fillna(
            fill_values['CompetitionOpenSinceMonthMean'])

        # save the mean of this column for later usage on test set
        fill_values['CompetitionOpenSinceYearMean'] = math.floor(df['CompetitionOpenSinceYear'].mean())
        df_new.loc[mask1, 'CompetitionOpenSinceYear'] = df_new.loc[mask1, 'CompetitionOpenSinceYear'].fillna(
            fill_values['CompetitionOpenSinceYearMean'])

        # Impute the columns 'CompetitionDistance', 'CompetetionOpenSinceMonth' and 'CompetitionOpenSinceYear'
        # with 0 values when 'CompetitionDistance' is missing => there is no competitor.
        mask2 = df_new['CompetitionDistance'].isnull()

        df_new['CompetitionDistance'].fillna(0, inplace=True)
        df_new.loc[mask2, 'CompetitionOpenSinceMonth'] = df_new.loc[mask2, 'CompetitionOpenSinceMonth'].fillna(0)
        df_new.loc[mask2, 'CompetitionOpenSinceYear'] = df_new.loc[mask2, 'CompetitionOpenSinceYear'].fillna(0)

        # Impute with 0 the columns related to the participation to Promo2 when the store didn't
        # participate to the promo2.
        df_new['Promo2SinceWeek'].fillna(0, inplace=True)
        df_new['Promo2SinceYear'].fillna(0, inplace=True)
        df_new['PromoInterval'].fillna(0, inplace=True)

        ###############
        # mean sales by store and by store type
        ###############

        def get_mean_cart(sales, customers):
            if sales > 0 and customers > 0:
                return sales / float(customers)
            else:
                return 0

        df_new['Sales'] = df_new['Sales'].apply(lambda x: np.log(x) if x > 0 else 0)
        df_new['MeanCart'] = df_new.apply(lambda row: get_mean_cart(row['Sales'], row['Customers']), axis=1)

        fill_values['MeanCart'] = df_new[['Store', 'MeanCart']]
        fill_values['MeanCart'].drop_duplicates(subset='Store', inplace=True)

        df_mean_cart_by_type = df_new.groupby('StoreType')['MeanCart'].aggregate(['mean'])
        df_mean_cart_by_type.reset_index(inplace=True)

        '''
        Computing df_mean_cart_by_type gives the following data frame

        StoreType     mean
            a       8.846277
            b       5.133097
            c       8.626227
            d       11.277862
        '''
        # we can now add a new column to our dataframe
        df['MeanCartByType'] = df.merge(df_mean_cart_by_type, on='StoreType')['mean']
        print("Added columns 'MeanCart', 'MeanCartByType'.")

        fill_values['MeanCartByType'] = df_mean_cart_by_type

        return df_new, fill_values

    @staticmethod
    def transform(df, fill_values):
        """
        Same operations applied as in the train transformation except when mean is used: should use the mean values
        used from the train set.
        :param df: (Pandas dataframe) test set.
        :param fill_values: value from the distribution of the train set.
        :return:
        """
        df_new = df.copy()

        mask1 = (df_new['CompetitionOpenSinceMonth'].isnull()) & (df_new['CompetitionDistance'].isnull() == False)

        df_new.loc[mask1, 'CompetitionOpenSinceMonth'] = df_new.loc[mask1, 'CompetitionOpenSinceMonth'].fillna(
            fill_values['CompetitionOpenSinceMonthMean'])
        df_new.loc[mask1, 'CompetitionOpenSinceYear'] = df_new.loc[mask1, 'CompetitionOpenSinceYear'].fillna(
            fill_values['CompetitionOpenSinceYearMean'])

        mask2 = df_new['CompetitionDistance'].isnull()

        df_new['CompetitionDistance'].fillna(0, inplace=True)
        df_new.loc[mask2, 'CompetitionOpenSinceMonth'] = df_new.loc[mask2, 'CompetitionOpenSinceMonth'].fillna(0)
        df_new.loc[mask2, 'CompetitionOpenSinceYear'] = df_new.loc[mask2, 'CompetitionOpenSinceYear'].fillna(0)

        # Fill with 0 promo2 variables
        df_new['Promo2SinceWeek'].fillna(0, inplace=True)
        df_new['Promo2SinceYear'].fillna(0, inplace=True)
        df_new['PromoInterval'].fillna(0, inplace=True)

        df_new['Sales'] = df_new['Sales'].apply(lambda x: np.log(x) if x > 0 else 0)

        df_new = df_new.merge(fill_values['MeanCart'], on='Store')
        df_new = df_new.merge(fill_values['MeanCartByType'], on='StoreType')
        df_new.rename(columns={'mean': 'MeanCartByType'}, inplace=True)

        return df_new

    @staticmethod
    def preprocess_dates(df_list):
        """
        Extract information from the date such as the month, day of the month...
        :param df_list: list of Pandas dataframes with a date column that we want to deal with.
        """

        def get_months_since_competition(year, competition_open_since_year, month, competition_open_since_month):
            if competition_open_since_year > 0 and competition_open_since_month > 0:
                return 12 * (year - competition_open_since_year) + (month - competition_open_since_month)
            else:
                return 0

        def get_month_since_promo(year, promo2_since_year, week_of_year, promo2_since_week):
            if promo2_since_week > 0 and promo2_since_year > 0:
                return 12 * (year - promo2_since_year) + (week_of_year - promo2_since_week) / 4.

        for df in df_list:
            df['Date'] = pd.to_datetime(df['Date'])
            df['DayOfMonth'] = df['Date'].apply(lambda x: x.day)
            df['Month'] = df['Date'].apply(lambda x: x.month)
            df['Year'] = df['Date'].apply(lambda x: x.year)
            df['DayOfWeek'] = df['Date'].apply(lambda x: x.dayofweek)
            df['WeekOfYear'] = df['Date'].apply(lambda x: x.weekofyear)
            df.drop(['Date'], inplace=True, axis=1)

            # number of months since a competition store has opened
            df['MonthsSinceCompetition'] = df.apply(
                lambda row: get_months_since_competition(row['Year'], row['CompetitionOpenSinceYear'], row['Month'],
                                                         row['CompetitionOpenSinceMonth']), axis=1)
            df.drop(['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'], inplace=True, axis=1)

            # number of months since a promotion has started
            df['MonthsSincePromo'] = df.apply(
                lambda row: get_month_since_promo(row['Year'], row['Promo2SinceYear'], row['WeekOfYear'],
                                                  row['Promo2SinceWeek']), axis=1)
            df['MonthsSincePromo'] = df['MonthsSincePromo'].apply(lambda x: x if x > 0 else 0)
            df.drop(['Promo2SinceYear', 'Promo2SinceWeek', 'PromoInterval'], inplace=True, axis=1)

        print("Added columns: 'DayOfMonth', 'Month', 'Year', 'DayOfWeek', 'WeekOfYear', 'MonthsSinceCompetition', "
              "'MonthsSincePromo'.")
        print("Removed columns: 'Date', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'Promo2SinceYear', "
              "'Promo2SinceWeek'")

    @staticmethod
    def handle_categorical(df_list):
        """
        Handle categorical variables and encode them.
        :param df_list: list of Pandas dataframes containing categorical variables to deal with.
        """
        for df in df_list:
            # define a dict where the keys are the element to encode, and the values are the targets
            encodings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
            # now we can encode the categorical features
            df['StateHoliday'].replace(encodings, inplace=True)
            df['StoreType'].replace(encodings, inplace=True)
            df['Assortment'].replace(encodings, inplace=True)

        print('Dealed with categorical features.')

    @staticmethod
    def get_most_important_features(X_train, y_train, n_features):
        """
        Perform feature selection using XGBoost algorithm
        """
        model = XGBClassifier()
        model.fit(X_train, y_train)
        sorted_idx = np.argsort(model.feature_importances_)[::-1]
        sorted_idx = sorted_idx[:n_features]
        return X_train.columns[sorted_idx]

    @staticmethod
    def outlier_detection(df):
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

    def run(self):
        """
            Run the whole pipeline of pre-processing.
        """
        self.load_data()
        df_list = [self.df_train, self.df_valid]
        self.handle_categorical(df_list)
        self.df_train, fill_values = self.fit(self.df_train)
        self.df_valid = self.transform(self.df_valid, fill_values)
        self.preprocess_dates(df_list)

        print(self.df_train.info())
        print(self.df_valid.info())


if __name__ == "__main__":
    preprocessor = Preprocessor('data')
    preprocessor.run()
