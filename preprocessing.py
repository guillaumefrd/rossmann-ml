"""
    This file is made to pre-process the data so that it can be fed to a machine learning algorithm.
    - data pre-processing
    - feature extraction
"""
import pandas as pd
import os
import numpy as np

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
    def preprocess_dates(df_list):
        """
            Extract information from the date such as the month, day of the month...
            :param df_list: list of Pandas dataframes with a date column that we want to deal with.
        """
        for df in df_list:
            df['Date'] = pd.to_datetime(df['Date'])
            df['DayOfMonth'] = df['Date'].apply(lambda x: x.day)
            df['Month'] = df['Date'].apply(lambda x: x.month)
            df['Year'] = df['Date'].apply(lambda x: x.year)
            df['DayOfWeek'] = df['Date'].apply(lambda x: x.dayofweek)
            df['WeekOfYear'] = df['Date'].apply(lambda x: x.weekofyear)
            df.drop(['Date'], inplace=True, axis=1)

            # number of months since a competition store has opened
            df['MonthsSinceCompetition'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + \
                                    (df['Month'] - df['CompetitionOpenSinceMonth'])
            df.drop(['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'], inplace=True, axis=1)

            # number of months since a promotion has started
            df['MonthsSincePromo'] = 12 * (df['Year'] - df['Promo2SinceYear']) + \
                              (df['WeekOfYear'] - df['Promo2SinceWeek']) / 4.0
            df['MonthsSincePromo'] = df['MonthsSincePromo'].apply(lambda x: x if x > 0 else 0)
            df.drop(['Promo2SinceYear', 'Promo2SinceWeek'], inplace=True, axis=1)

        print("Added columns: 'DayOfMonth', 'Month', 'Year', 'DayOfWeek', 'WeekOfYear', 'MonthsSinceCompetition', "
              "'MonthsSincePromo'.")
        print("Removed columns: 'Date', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'Promo2SinceYear', "
              "'Promo2SinceWeek'")

    @staticmethod
    def add_avg_basket(df_list):
        """
        Add average customer's basket value according for each store individually, and for each store type.
        :param df_list: list of Pandas dataframes
        """
        for df in df_list:
            df['MeanCart'] = df['Sales'] / df['Customers']
            # get the average basket value by store type
            df_mean_cart_by_type = df.groupby('StoreType')['MeanCart'].aggregate(['mean'])
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
        self.preprocess_dates(df_list)
        self.add_avg_basket(df_list)
        self.handle_categorical(df_list)
        print(self.df_train.columns)
        print(self.df_train.info())


preprocessor = Preprocessor('data')
preprocessor.run()
