import numpy as np
import pandas as pd
import statsmodels.api as sm
from src.econometrics import Econometrics 

metrics = Econometrics()

class Aux:

    def shift_RHS(lhs, rhs, num_periods=12, freq='M'):
        rhs   = rhs.shift(num_periods, freq=freq)
        # We now take the inresection so we have a good dataframe without nans.
        new_index = lhs.index.intersection(rhs.index)
        lhs = lhs.loc[new_index]
        rhs = rhs.loc[new_index]
        return lhs, rhs

    def format_index(df):
        if "Date" in df.columns: 
            df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d")
            df = df.set_index("Date")
            df.index.name = None
            df.index = df.index.date
        else: df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        return df
    
    def take_same_index(df1, df2):
        new_index = df1.index.intersection(df2.index)
        df1 = df1.loc[new_index]
        df2 = df2.loc[new_index]
        return df1, df2
    
    def OLS(y,X, correct=1):
        y, X = metrics.y_and_X_as_numpy_arrays(y,X)
        ## Now do everything
        if correct == 1:
            alpha_and_beta, se_beta, R2v, R2adjv, Sigma, F = metrics.OLS_gmm_corrected_se(y,X,12,0)
        else: 
            alpha_and_beta = metrics.OLS_beta(y,X)
            se_beta = None
            R2v = None
            R2adjv = None
            Sigma = None
            F = None
            t_stat = None
        #alpha_and_beta = metrics.OLS_beta(y,X)
        alpha = alpha_and_beta.T["beta0"].values
        # We want alpha to be at least 1D array. Sometimes the second
        # dimension is ommited as if it were a list.
        try: alpha = alpha.reshape(alpha.shape[0],alpha.shape[1])
        except IndexError: alpha = alpha.reshape(alpha.shape[0],1)
        # Take the beta
        beta  = alpha_and_beta.T[["beta" + str(n) for n in range(1,len(alpha_and_beta))]].values    
        # Do the same as in alpha fot the beta:
        try: beta = beta.reshape(beta.shape[0],beta.shape[1])
        except IndexError: beta = beta.reshape(beta.shape[0],1)
        if correct == 1:
            t_stat = np.divide(beta, se_beta.values[1:].T)
            t_stat = np.ravel(t_stat)
        else: t_stat = None
        return alpha, beta, se_beta, t_stat, R2v, R2adjv, Sigma, F

    def is_float(a):
        try:
            if  (a.shape[0] == 1) and (a.shape[1] == 1): a = float(np.round(a,2))
        except (TypeError, IndexError):
            if len(a) == 1: a = float(np.round(a,2))
        return a
            