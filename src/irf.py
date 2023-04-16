import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.affine_model import AffineModel 


class IRF:

    def __init__(self) -> None:

        # Construct the Affine Model and all its variabels and parameters:
        self._am = AffineModel()

        # Remember, K is the number of factors:
        self._num_factors = self._am.K

        # Number of years:
        self._T = 10

        # FORWARD RATES DYNAMICS COEFFICIENTS:
        if type(self._am.Bf) is pd.DataFrame: 
            self._Af  = self._am.Af.values
            self._Bf = self._am.Bf.values
        else: 
            self._Af = self._am.Af
            self._Bf = self._am.Bf
        if type(self._am.delta1) is pd.DataFrame: self._delta1 = self._am.delta1.values
        else: self._delta1 = self._am.delta1

        # X DYNAMICS COEFFICIENTS:
        # Coefficient of the AR(1) model for X fitted using OLS:
        self._phi = pd.read_csv("data/results/phi.csv", index_col=0)

        # Coefficient of the AR(1) model for X under risk neutrality:
        self._phistar = self._am.phistar

        # REAL oefficient of the AR(1) model for X.
        self._real_phi = self._am.real_phi

    def irf_X(self, i, real_or_risk_neutral = "real"):
        """
        IRF for X.
        """
        # Take the appropriate phi:
        _phi = self.appropriate_phi(real_or_risk_neutral)
        # Create the factor X time series:
        X_t = np.zeros((self._T,self._num_factors))  
        # Shock at time=1:
        X_t[1,i] = 1
        # Take the dynamics:
        for t in range(2,self._T):
            X_t[t,:] = X_t[t-1,:]@_phi.T 
        # Divide by 2 for scaling reasons:
        X_t[:,0] /= 2 
        return X_t, _phi
    
    def irf_f_Ey1_and_x(self, i, real_or_risk_neutral = "real"):
        # Take the appropriate phi:
        _phi = self.appropriate_phi(real_or_risk_neutral)
        # Create the factor X time series:
        _X_t = np.zeros((self._T,self._num_factors))  
        # Create the Forward Rate time series:
        _F_t = np.zeros((self._T,1))  
        # Shock at time=1 to the factor X::
        _X_t[1,i] = 1
        # Inherited shock at time=1 to F:
        _F_t[1] = self._Af[1] + _X_t[1,:]@self._Bf[1,:].T
        for t in range(2,self._T):
            # We need the complete series of the factor X:
            _X_t[t,:]   = _X_t[t-1,:]@_phi.T 
            # Note that we just multiply by the factor X:
            _F_t[t]     = self._Af[t-1] + _X_t[1,:]@self._Bf[t-1,:].T
        _Ey1 = _X_t @ self._delta1 
        return _X_t, _F_t, _Ey1, _phi

    def appropriate_phi(self, real_or_risk_neutral):
        if "real" in real_or_risk_neutral: _phi = self._real_phi
        elif "neutral" in real_or_risk_neutral: _phi = self._phistar
        else:
            raise ValueError("ϕ must be either 'real' or 'risk_neutral'")
        return _phi