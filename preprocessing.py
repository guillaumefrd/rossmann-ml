"""
    This file is made to pre-process the data so that it can be fed to a machine learning algorithm.
    - data pre-processing
    - feature extraction
"""

from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
import pandas as pd
import os
import numpy as np


class Preprocessor:
    def __init__(self, folder_path):
        """
        Initialize class instance.
        :param folder_path: (string) folder containing the csv files.
        """
        self.df_store, self.df_train, self.df_test = None, None, None
        self.folder_path = folder_path

    def load_data(self):
        """
            Load data into Pandas dataframes.
        """
        self.df_store = pd.read_csv(os.path.join(self.folder_path, 'store.csv'))
        self.df_train = pd.read_csv(os.path.join(self.folder_path, 'train.csv'))
        self.df_test = pd.read_csv(os.path.join(self.folder_path, 'test.csv'))

        # merge dataframes to get store information in both training and testing sets.
        self.df_train = self.df_train.merge(self.df_store, on='Store')
        self.df_test = self.df_test.merge(self.df_store, on='Store')

        print('Loaded tables in Pandas dataframes.\n')
        print('df_store shape: {} \ndf_train shape: {} \ndf_test shape: {}\n'.format(self.df_store.shape,
                                                                                     self.df_train.shape,
                                                                                     self.df_test.shape))

    @staticmethod
    def preprocess_date(df):
        """
            Extract information from the date such as the month, day of the month...
            :param df: (Pandas dataframe) dataframe with a date column that we want to deal with.
        """
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfMonth'] = df['Date'].apply(lambda x: x.day)
        df['Month'] = df['Date'].apply(lambda x: x.month)
        df['Year'] = df['Date'].apply(lambda x: x.year)
        df['DayOfWeek'] = df['Date'].apply(lambda x: x.dayofweek)

    @staticmethod
    def add_avg_basket(df):
        """
        Add average customer's basket value according for each store individually, and for each store type.
        :param df: (Pandas dataframe)
        """
        df['mean_cart'] = df['Sales'] / df['Customers']
        # get the average basket value by store type
        df_mean_cart_by_type = df.groupby('StoreType')['mean_cart'].aggregate(['mean'])
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
        df['mean_cart_by_type'] = df['StoreType'].apply(
            lambda x: df_mean_cart_by_type[df_mean_cart_by_type['StoreType'] == x]['mean'].values[0])

    @staticmethod
    def handle_categorical(df):
        """
        Handle categorical variables and encode them.
        :param df: (Pandas dataframe) dataframe containing categorical variables to deal with.
        """
        # define a dict where the keys are the element to encode, and the values are the targets
        encodings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
        # now we can encode the categorical features
        df['StateHoliday'].replace(encodings, inplace=True)
        df['StoreType'].replace(encodings, inplace=True)
        df['Assortment'].replace(encodings, inplace=True)

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
        self.preprocess_date(self.df_train)
        self.preprocess_date(self.df_test)
        print("Columns 'DayOfMonth', 'DayOfWeek', 'Year' and 'Month' have been added to both train and test sets.")

    # TODO: fill nan, normalization (careful with outliers)
    #  Average customer basket value according to store type, and for each store,
    #  number of month since the competition store is open


preprocessor = Preprocessor('data')
preprocessor.run()
