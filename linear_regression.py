import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston


b = load_boston()
df = pd.DataFrame(data=b.data, columns=b.feature_names)
df['y'] = b.target

def calculate_normal_equation(X, y, add_intercept=True):
    if (isinstance(X, pd.DataFrame))|(isinstance(X, pd.Series)):
        X = X.values
    if (isinstance(y, pd.DataFrame))|(isinstance(y, pd.Series)):
        y = y.values
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    if add_intercept:
        X = np.pad(X, (1,0), 'constant', constant_values=(1,))[1:]
        
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


class OLS:
    
    def __init__(self, df, X_cols, y_col):
        self.df = df
        self.feature_names = X_cols.astype('>U9')
        self.target_name = y_col
        
    def fit_model(self, add_intercept:bool=True):
        params = calculate_normal_equation(self.df[self.feature_names],
                                           self.df[self.target_name],
                                           add_intercept)
        self.params = params
        if add_intercept:
            names = np.pad(self.feature_names, (1,0), 'constant', constant_values=('intercept',))
        self.params_dict = dict(zip(names, params))
        
    def predict(self, test_df):
        assert 'params' in dir(self), 'OLS instance not fitted yet'
        assert set(self.feature_names).issubset(test_df.columns), 'not all features are present in dataframe'
        assert self.target_name in test_df.columns, 'target variable not in dataframe'
        
        X = test_df[self.feature_names].values
        if 'intercept' in self.params_dict.keys():
            X = np.pad(X, (1,0), 'constant', constant_values=(1,))[1:]
        predictions = (X*self.params).sum(axis=1)
        return predictions
        
        
        