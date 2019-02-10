from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import linear_model

class Model(BaseEstimator, ClassifierMixin):
    """scikit-learn estimator for the Rossmann's stores problem
    
    Parameters
    ----------
    alpha : float
        The regularization parameter for ridge and lasso regression
    max_iter : int
        The number of iterations / epochs to do on the data.
    solver : 'xgb' | 'lasso' | 'ridge' | 'linear'
  
    """
    def __init__(self, max_iter=2000, solver='xgb', alpha=0.1):
        self.max_iter = max_iter
        self.alpha = alpha
        self.solver = solver
        self.model  = None
        assert self.solver in ['xgb', 'lasso', 'ridge', 'linear'] 
        

    def fit(self, X, y):
        """Fit method
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The features.
        y : ndarray, shape (n_samples,)
            The target. 
        """
        
        if self.solver == 'xgb':
            self.model = XGBRegressor(objective="reg:linear")
            self.model.fit(X, y, eval_metric='rmse')
            
        elif self.solver == 'lasso':
            self.model = linear_model.Lasso(alpha=self.alpha, max_iter=self.max_iter)
            self.model.fit(X, y)
            
        elif self.solver == 'ridge':
            self.model = linear_model.Ridge(alpha=self.alpha, max_iter=self.max_iter)
            self.model.fit(X, y)
            
        elif self.solver == 'linear':
            self.model = linear_model.LinearRegression().fit(X, y)
            
        return self
    

    def predict(self, X):
        """Predict method
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The features.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            The predicted target.
        """
        return self.model.predict(X)

   