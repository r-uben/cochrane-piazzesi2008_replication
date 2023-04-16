import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.aux import Aux as aux
from src.cleaners.cleaner_f import CleanerF
from src.cleaners.cleaner_rx import CleanerRX
from src.econometrics import Econometrics
from src.files_headers import FilesHeaders as fh


class AffineModel:

    def __init__(self, 
                 level_or_spread = "spread", 
                 gsw_or_fb = "gsw", 
                 tex_path = "/Users/ruben/Dropbox/library/replications/cochrane-piazzesi2008/tex/"):

        # Keep the inputs:
        self.level_or_spread = level_or_spread
        self.gsw_or_fb = gsw_or_fb

        # Data:
        self._rx = pd.read_csv("data/gsw_rx.csv", 
                             index_col=[0], 
                             parse_dates=True, 
                             date_parser=lambda x : pd.to_datetime(x).to_period('M')).copy()
        self._f = pd.read_csv("data/fb_f.csv", 
                            index_col=0, 
                            parse_dates=True, 
                            date_parser=lambda x: pd.to_datetime(x).to_period('M')).copy()

        # Affine term structure model related variables:
        self._N = 20    # How far out to calculate the term structure
        self._K = 4     # This will be the number of factors (X = (x,level,slope, curvature)) 
        self._A = np.zeros((self._N,1))
        self._B = np.zeros((self._N, self._K))
        self._Af = self._A
        self._Bf = self._B
        self._delta0    = None
        self._delta1    = None
        self._phistar   = None
        self._mustar    = None 

        # Class with econometric functions:
        self._metrics    = Econometrics()

        # Forward rates to calculate the return forecasting factor::
        self._raw_clean  = CleanerF("fb")
        self._raw_clean.load_and_clean()
        self._raw_f      = self._raw_clean.f_spread
        self._f_levels   = self._raw_clean.f_levels
        self._the_idx    = self._raw_clean.the_idx
        # Return Forecasting Factor dictionary:
        self._x = None

        # Vector of factors:
        self._X = None

        # Expected returns: We are always using GSW (2006) data:
        self._raw_clean  = CleanerRX()
        self._raw_rx     = self._raw_clean.rx

        # Path for the tex file:
        self.tex_path = tex_path

        # Forecast errors from regressing rx on the forward rates:
        self._eps       = None

        # Matrix of eigenvectors and eigenvalues for rx_GSW:
        self._Q         = None
        self._Lambda    = None

        # OLS loadings of forward rates (GSW) on the vector X of factors:
        self._a     = None
        self._b     = None 

        # Covariances of returns with factor shocks, and fitted values:
        self._C     = None

        # AR(1) for the factors, X:
        self._mu    = None
        self._phi   = None
        self._v     = None
        self._V     = None
        self._std_v = None

        # Market price of risk dataset:
        self._lamb  = None

        # Assuming lambda = lambda0 + lambda1 * X:
        self._lamb0 = None
        self._lamb1 = None

        # Fitted highest-eigenvalue eigenvector by regressing qr on cov(rx, v)
        self._qr_fitted  = None

        # Same but taking only thelevel:
        self._qr_fitted_level = None

        # Aux variables:
        self._gamma0 = None
    
    ## ----------------------------------------------------------------
    ## FUNCTION TO CREATE THE RETURN FORECASTING
    def create_x(self):
        _x_dict = {}
        for unit in ["level", "spread"]:
            # Save the excess returns as a private variable:
            _rx = self.raw_rx
            # Load and clean the data:
            FFB = CleanerF("fb")
            FFB.load_and_clean()
            # Save the forward rates as a private variable:
            if unit == "level": _f = FFB.f_levels
            if unit == "spread": _f = FFB.f_spread
            # Save the no shifted forward rate as a private variable but adding a constant:
            _f_no_shift = sm.add_constant(_f)
            # Shifft:
            _rx, _f = aux.shift_RHS(_rx, _f)

            # Run the regression:   
            _output = aux.OLS(_rx, _f)
            _alpha = _output[0]
            _beta  = _output[1]
            _alpha_and_beta = np.concatenate((_alpha, _beta), axis=1)

            # Calculate Expected returns:
            _Erx = _f_no_shift @ _alpha_and_beta.T
            _Erx.columns = _rx.columns 
            _Erx.index = _f_no_shift.index
            _Erx.to_csv("data/Erx_" + unit + ".csv")

            # Save the errors:
            _eps = _rx.values - sm.add_constant(_f) @ _alpha_and_beta.T
            _eps = pd.DataFrame(_eps.values, 
                                index=_f.index, 
                                columns=_rx.columns)
            _eps.to_csv("data/other/eps_rx_" + unit + ".csv")
            if self.level_or_spread == unit: self._eps = _eps
            # Calculate the covariance matrix  
            _cov_rx = np.cov(_Erx.values, rowvar=False)
            # We decompose (eigenvalues & eigenvectors, ordering them)
            self._Q, self._Lambda = self._metrics.eig_decompose(_cov_rx)
            self._Q.index = [col.replace("rx","") for col in _rx.columns]
            self._Q.columns = [col.replace("rx","") for col in _rx.columns]
            self._Q.to_csv("data/other/Q_cov_Erx_" + unit + ".csv")
            self._Lambda.index   = self._Q.index
            self._Lambda.columns = self._Q.columns
            self._Lambda.to_csv("data/other/lamb_cov_Erx_" + unit + ".csv")
            
            # Same with f:
            _cov_f = np.cov(_f, rowvar=False)
            _Qf, _Lambdaf = self._metrics.eig_decompose(_cov_f)
            _Qf.index  = [col.replace("f","") for col in _f.columns]
            _Qf.columns = [col.replace("f","") for col in _f.columns]
             # Express results in terms of regression on levels of f
            if unit == 'spread':
                _beta = np.concatenate((
                -np.sum(_beta, axis=1, keepdims=True),
                _beta
            ), axis=1)
                
            # The return forcasting factor:
            _qr = self._Q[self._Q.columns[0]].values.reshape(-1,1)
            _gamma = (_beta.T @ _qr).reshape(-1,1)
            _fvals = self.f_levels.values
            _x_vec = _qr.T@_alpha + _fvals @ _gamma
            # This is to avoid ValueError when constructing the DataFrame
            _x_dict[unit] =  pd.DataFrame(_x_vec.reshape(-1), index=FFB.the_idx)
            if unit == self.level_or_spread: 
                self._x = _x_dict[unit]
                self._x.columns = ["x"]

            ## Check whether the return forecasting factor is correct:
            _x = _x_dict[unit].shift(12, freq='M')
            _y, _X = aux.take_same_index(_rx, _x)
            self._gamma0, beta, _, _, _, _, _, _ = aux.OLS(_y,_X,0)
            for i, rxi in enumerate(_rx.columns):
                _new_X = self._gamma0[i] + _qr[i]*_X
                alpha, beta, _, _, _, _, _, _ = aux.OLS(_y[rxi],_new_X, 0)
                #print(float(alpha), float(beta))

        _x = pd.concat(_x_dict.values(), axis=1)
        _x.columns = ["level", "spread"]
        _x =_x.dropna()
        _x.to_csv("data/cleaned/x.csv")
        return _x
    
    ## FUNCTION TO CREATE THE LEVEL, SLOPE AND CURVATURE FACTORS
    def create_X(self):
        '''
            Inputs:
                - None

            Outputs:
                - X: the observable factors (Cochrane-Piazzesi 2008)

            Description:
                - This function creates the return forcasting factor, and the level, slope and curvature factors.
                - The level, slope and curvature factors are obtained from the Gürkaynak, Sack and Wright 2006 data.
                - The return forcasting factor (Cochrane-Piazzesi 2008) is obtained before. We just load it.
                - All factors are demeaned.
        '''
        # Get the return forecasting factor (Cochrane-Piazzesi 2008):
        _x = self.x
        # Get the forward rates (Gürkaynak, Sack and Wright 2006):
        _f = self.get_f("gsw")
        # Add a constant to _x to get the RHS part of the model:
        _X = sm.add_constant(_x)
        # Get the subdataframes with coincindent indexes:
        _f, _X = aux.take_same_index(_f, _X)
        # Run the regression:
        _c, _d, _, _, _, _, _, _ = aux.OLS(_f, _X, 0)
        # Concatenate the coefficients:
        _c_d = np.concatenate((_c, _d), axis=1)
        # Take the errors:
        _e = _f - _X.values @ _c_d.T
        # Take the covariance matrix:
        _cov_E = np.cov(_e, rowvar=False)
        # Take the eigenvalues:
        _Q_e, _ = self._metrics.eig_decompose(_cov_E)
        _Q_e.index = ["e" + str(i) for i in range(1,len(_Q_e.columns)+1)]
        _Q_e.columns = _Q_e.index
        # Take the level, slope and curvature factors:
        _c     = (_c @ np.ones((1, len(_f)))).T
        _level = (_c + _e ) @ _Q_e["e1"].values.reshape(-1,1)
        _slope = (_c + _e ) @ _Q_e["e2"].values.reshape(-1,1)
        _curvature = (_c + _e ) @ _Q_e["e3"].values.reshape(-1,1)
        # Construct the factor:
        _X = pd.concat((_X["x"], _level, _slope, _curvature), axis=1)
        _X.columns = ["x", "level", "slope", "curvature"]
        # Demean the factors: the reason why they are demeaning is that, for the model X_{t+1} = μ + ϕ Xt + v_{t+1},
        # model factor mean is E[X] = (I-ϕ)^{-1}μ, so it's a very sensitive function of μ. Also, long-term means
        # are important to long-term forward rate decompositions. Hence, by de-meaning the factors before starting, we make sure
        # the model produces the sample means of the factors, or at least that (1-ϕ)Ε[X] = μ (= 0)
        _X = _X - _X.mean(axis=0)
        return _X
    
    ## FUNCTION TO TAKE THE LOADINGS OF FORWARD RATES ON FACTORS IN STATISTICAL MODEL:
    ## This is our benchmark for cross-sectional fitting.x = pd.concat(x.values(), axis=1)
    def f_on_X_OLS(self):
        # Take my factors and include a constant:
        _X = sm.add_constant(self.X)
        # Take the forward rates:
        _f = self.get_f("gsw")
        # Same index:
        _f, _X = aux.take_same_index(_f, _X)
        # Get the results from the OLS regression:
        _output = aux.OLS(_f, _X, 0)
        # Get the loadings:
        _a = _output[0]
        _b = _output[1]
        # Concatenate the loadings:
        _loadings = np.concatenate((_a, _b), axis=1)
        return _a, _b, _X @ _loadings.T

    ## FUNCTION TO GET THE LOADINGS OF THE UNCONSTRAINED VAR ESTIMATE OF FACTOR DYNAMICS:
    ## MODEL: X_{t+1} = mu + phi Xt + v_{t+1}; V = E[vv']
    def X_VAR_dynamics(self):
        # We take a coppy of self.X to lag it 12 months (1 year):
        _lagged_X = self.X.copy()
        _lagged_X = _lagged_X.shift(periods=12).dropna()
        # Add a constant:
        _lagged_X = sm.add_constant(_lagged_X)
        # We take X as a private variable:
        _X = self.X
        # Take same index to avoid problems with the dimensions of X:
        _X, _lagged_X = aux.take_same_index(_X, _lagged_X)
        # Get the results from the OLS regression:
        _output = aux.OLS(_X, _lagged_X, 0)
        # Get the loadings:
        _mu = _output[0]
        _phi = _output[1]
        # Concatenate the loadings:
        _loadings = np.concatenate((_mu, _phi), axis=1)
        # Take the errors:
        _v = _X.values - _lagged_X.values @ _loadings.T
        # Take the matrix of VAR errors:
        _V = np.cov(_v, rowvar=False)
        _V = pd.DataFrame(_V, columns=_X.columns, index=_X.columns)
        # Save _v as a dataframe:
        _v = pd.DataFrame(_v, index=_X.index, columns=_X.columns)
        return _mu, _phi, _v, _V 

    ## FUNCTION TO CREATE AND GET THE MARKET PRICES OF RISK:
    def create_lambdas(self):
        # Get the excess returns:
        _eps = self.eps
        # Get the errors:
        _v = self.v
        # Take the same index:
        _eps, _v = aux.take_same_index(_eps, _v)
        # Take the std of the errors:
        self._std_v = np.std(_v)
        # Covariance matrix:
        self._C = _eps.T @ _v / len(_eps)
        self._C.index = [idx.replace("rx","") for idx in self._C.index]
        # Market Prices of Risk:
        self._lamb = np.linalg.inv(self.C.T@ self.C) @ self.C.T @ self.qr
        self._lamb.index = _v.columns
        self._lamb.columns = ["lamb1"]
        self._lamb.loc[:, "lamb0"] = (np.linalg.inv(self.C.T@ self.C) @ self.C.T @ (self.gamma0 + 1/2 * np.var(_eps).values.reshape(-1,1))).values
        # Fit qr using the market prices of risk
        self._qr_fitted = self._C @ self._lamb["lamb1"]
        # Now repeat the same but using only the level:
        _C_level = self.C.loc[:,"level"].to_frame()
        self._lamb.loc["level", "lamb_1_level"] = (np.linalg.inv(_C_level.T @ _C_level) @ _C_level.T @ self.qr).values
        self._lamb.loc["level", "lamb_0_level"] = (np.linalg.inv(_C_level.T @ _C_level) @ _C_level.T @ (self.gamma0 + 1/2 * np.var(_eps).values.reshape(-1,1))).values
        # Fit qr using the level market prices of risk:
        self._qr_fitted_level = self._C["level"] * self._lamb.loc["level", "lamb_1_level"]
        self.lamb.to_csv("data/other/lamb.csv")
        # Now take the lambda0 and lambda1 matrices:
        lam0 = self.lamb.loc["level", "lamb_0_level"]
        lam1 = self.lamb.loc["level", "lamb_1_level"]
        self._lamb0 = np.array([0,lam0, 0, 0]).reshape(-1,1)
        self._lamb0 = pd.DataFrame(self._lamb0, index=self.lamb.index, columns = ["lamb0"])
        self._lamb1 = np.array([0, lam1, 0, 0]).reshape(-1,1)
        self._lamb1 = np.concatenate([self.lamb1, np.zeros((4,3))], axis=1)
        self._lamb1 = pd.DataFrame(self._lamb1, index=self.lamb.index, columns = self.lamb.index)
    ## FUNCTION TO CALCULATE THE AFFINE TERM STRUCTURE
    def affine_term_structure(self):
        self._A[0]      = -self.delta0
        self._B[0,:]    = -self.delta1.T
        _delta1 = self.delta1
        _delta0 = self.delta0
        _mustar = self.mustar.values
        _phistar = self.phistar.values
        _V = self.V.values
        for n in range(1, self.N):
            _Anminus1   = self._A[n-1]
            _Bnminus1   = self._B[n-1,:].reshape(-1,1)
            self._B[n,:] = -_delta1.T + _Bnminus1.T@_phistar
            self._A[n]   = -_delta0 + float(_Anminus1) + float(_Bnminus1.T@_mustar) + 1/2*float(_Bnminus1.T@_V@_Bnminus1)

        self._A     = pd.DataFrame(self._A, 
                                    columns=["const"], 
                                    index=[n for n in range(1,self.N+1)]
                                )
        self._B     = pd.DataFrame(self._B, 
                                    columns=["x", "level", "slope", "curvature"], 
                                    index=[n for n in range(1,self.N+1)]
                                )

        self._Af    = pd.DataFrame(np.concatenate([np.array([self.delta0]).reshape(-1,1), 
                                    self.A.iloc[0:self.N-1].values- self._A.iloc[1:self.N].values], 
                                    axis=0),
                                    columns = self.A.columns,
                                    index = self.A.index
                                )
        self._Bf    = pd.DataFrame(np.concatenate([self.delta1.T, 
                                    self.B.iloc[0:self.N-1].values - self.B.iloc[1:self.N].values], 
                                    axis=0),
                                    columns = self.B.columns,
                                    index = self.B.index
                                    )
        self.Af.to_csv("data/results/Af.csv")
        self.Bf.to_csv("data/results/Bf.csv")

    ## ----------------------------------------------------------------
    ## GETTERS:
    def get_X(self):
        _X = self.create_X()
        return _X
    
    def get_a(self):
        _a, _b, _f_fitted = self.f_on_X_OLS()
        _a = pd.DataFrame(_a, 
                            columns=["const"],
                            index=[n for n in range(1,_b.shape[0]+1)])
        return _a

    def get_b(self):
        _a, _b, _f_fitted = self.f_on_X_OLS()
        _b = pd.DataFrame(_b, 
                            columns=["x", "level", "slope", "curvature"],
                            index=[n for n in range(1,_b.shape[0]+1)])
        return _b

    def get_x(self): 
        # Take the data and convert:
        _x = self.create_x()
        _x = _x[self.level_or_spread].to_frame()
        _x.columns = ["x"]
        return _x
    
    def get_f(self, gsw_or_fb = "gsw"):
        FGSW = CleanerF(gsw_or_fb)
        FGSW.load_and_clean()
        return FGSW.f_levels

    def get_eps(self):
        eps = pd.read_csv("data/other/eps_rx_" + self.level_or_spread + ".csv", 
                        index_col=0,
                        parse_dates=True, 
                        date_parser=lambda x: pd.to_datetime(x).to_period('M')
                        )
        return eps
    
    def get_affine_model_vars(self):
        if (self._mu is None) or (self._phi is None) or (self._v is None)  or (self._V is None): 
            self._mu, self._phi, self._v, self._V = self.X_VAR_dynamics()
        if self._delta0 is None: self._delta0 = float(self.a.iloc[0])
        if self._delta1 is None: self._delta1 = self.b.iloc[0].values.reshape(-1,1)

    ## ----------------------------------------------------------------
    ## FACTORS AND PARAMETERS:
    @property
    def eps(self):
        if self._eps is None: self.create_x()
        self._eps = self.get_eps()
        return self._eps
    
    @property
    def x(self):
        if self._x is None: 
            self._x = self.get_x()
        return self._x
    
    @property
    def X(self):
        if self._X is None: self._X = self.get_X()
        return self._X
    
    @property
    def f_fitted_OLS_benchmark(self):
        _a, _b, _f_fitted = self.f_on_X_OLS()
        return _f_fitted

    @property
    def a(self):
        if self._a is None: self._a = self.get_a()
        self.save_result(self._a, "a")
        return self._a

    @property
    def b(self):
        if self._b is None: self._b = self.get_b()
        self.save_result(self._b, "b")
        return self._b

    @property
    def mu(self):
        if self._mu is None: self.get_affine_model_vars()
        self.save_result(self._mu, "mu")
        return self._mu

    @property
    def phi(self):
        if self._phi is None: self.get_affine_model_vars()
        self.save_result(self._phi, "phi")
        return self._phi
    
    @property
    def v(self):
        if self._v is None: self.get_affine_model_vars()
        return self._v

    @property
    def V(self):
        if self._V is None: self.get_affine_model_vars()
        self.save_result(self._V, "V")
        return self._V
    
    ## ------------------------------------------------------------------
    ## AFFINE TERM STRUCTURE VARIABLES AND PARAMETERS:
    @property
    def delta0(self):
        if self._delta0 is None: self.get_affine_model_vars()
        self.save_result(self._delta0, "delta0")
        return self._delta0
    
    @property
    def delta1(self):
        if self._delta1 is None: self.get_affine_model_vars()
        self.save_result(self._delta1, "delta1")
        return self._delta1
    
    @property
    def phistar(self):
        output = self.phi - self.V @ self.lamb1
        self.save_result(output, "phistar")
        return output

    @property
    def real_phi(self):

        output = self.phistar + self.V @ self.lamb1
        print(output)
        self.save_result(output, "real_phi")
        return output
    
    @property
    def mustar(self):
        output = self.mu - self.V @ self.lamb0
        self.save_result(output, "mustar")
        return output
    
    @property
    def A(self):
        mask = self.array_mask(self._A) and (self._A.all() == 0)
        if mask: self.affine_term_structure()
        self.save_result(self._A, "A")
        return self._A
    
    @property
    def B(self):
        mask = self.array_mask(self._B) and (self._B.all() == 0)
        if mask: self.affine_term_structure()
        self.save_result(self._B, "B")
        return self._B
    
    @property
    def Af(self):
        mask = self.array_mask(self._Af) and (self._Af.all() == 0)
        if mask: self.affine_term_structure()
        print(type(self._Af))
        self.save_result(self._Af, "Af")
        return self._Af
    
    @property
    def Bf(self):
        mask = self.array_mask(self._Bf) and (self._Bf.all() == 0)
        if mask: self.affine_term_structure()
        self.save_result(self._Bf, "Bf")
        return self._Bf
    
    ## –-----------------------------------------------------------------
    ## DATA:
    @property 
    def raw_rx(self):
        return self._rx
    
    @property
    def raw_f(self):
        return self._f
    
    @property
    def f_levels(self):
        return self._f_levels

    ## EIGENVECTORS AND EIGENVALUES:
    
    @property
    def Q(self):
        '''
            Matrix with the eigenvectors of the covariance matrix of the expected excess returns on its columns.
        '''
        if self._Q is None: self.create_x()
        self._Q = pd.read_csv("data/other/Q_cov_Erx_" + self.level_or_spread + ".csv", index_col=0)
        return self._Q
    
    @property
    def qr(self):
        '''
            Eigenvector with the highest associated eigevalue.
        '''
        return self.Q[self._Q.columns[0]].values.reshape(-1,1)
    
    @property
    def Lambda(self):
        '''
            Diagonal matrix with the eigenvalues of the covariance matrix of the expected excess returns.
        '''
        if self._Lambda is None: self.create_x()
        self._Lambda = pd.read_csv("data/other/Lambda_cov_Erx_" + self.level_or_spread + ".csv", index_col=0)
        return self._Lambda

    ## ----------------------------------------------------------------
    ## MARKET PRICES OF RISK:
    @property
    def lamb(self):
        if self._lamb is None: self.create_lambdas()
        return self._lamb

    @property
    def lamb0(self):
        if self._lamb is None: self.create_lambdas()
        return self._lamb0
    
    @property
    def lamb1(self):
        if self._lamb is None: self.create_lambdas()
        
        return self._lamb1
    
    @property
    def qr_fitted(self):
        if self._qr_fitted is None: self.create_lambdas
        return self._qr_fitted
    
    @property
    def qr_fitted_level(self):
        if self._qr_fitted_level is None: self.create_lambdas()
        return self._qr_fitted_level

    @property
    def C(self):
        if self._C is None: self.create_lambdas()
        return self._C
    
    @property
    def std_v(self):
        if self._std_v is None: self.create_lambdas_()
        return self._std_v
    
    ## ----------------------------------------------------------------
    ## AUXILIAR VARIABLES:
    @property
    def gamma0(self):
        if self._gamma0 is None: self.create_x()
        return self._gamma0

    @property
    def N(self):
        return self._N

    @property
    def K(self):
        return self._K

    ## ----------------------------------------------------------------
    ## AUXILIAR FUNCTIONS
    def save_result(self, result, name):

        def set_col_and_idx(result, name):
            factor_names = ["x", "level", "slope", "curvature"]
            num_idx  = result.shape[0]
            num_cols = result.shape[1]
            # COLUMNS
            if num_cols == 4: cols = factor_names
            else: cols = [name]
            # INDEXES
            if num_idx == 4: idx = factor_names
            else: idx = [n for n in range(1, num_idx+1)]
            return cols, idx

        if self.float_mask(result):
            result = pd.DataFrame([result], columns=[name])

        elif self.array_mask(result):
            cols, idx = set_col_and_idx(result, name)
            result      = pd.DataFrame(result, columns = cols, index = idx)

        elif self.dataframe_mask(result):
            cols, idx       = set_col_and_idx(result, name)
            result.columns  = cols
            result.index    = idx

        else:
            raise ValueError("Result must be a float, array or dataframe")

        result.to_csv("data/results/" + name + ".csv")

    def array_mask(self, array):
        return (type(array) == np.ndarray) or (type(array) == np.array)

    def dataframe_mask(self, result):
        return (type(result) == pd.DataFrame)

    def float_mask(self, result):
        return (type(result) == float)
