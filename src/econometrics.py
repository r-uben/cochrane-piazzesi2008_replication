import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

class Econometrics():

    

    def OLS_beta(self, 
                 y: np.array, 
                 X: np.array, 
                 starting_index = 1,
                 name_var = None) -> pd.DataFrame:
        '''
            This function computes the OLS betas, i.e., b in y = Xb

            Input: 
                1. y: TxN np.array (lhs in the equation).
                2. X: TxK np.array (rhs in the equation).

                NOTE: If N>1, this runs N regressions of the left hand columns on all the (same) 
                right hand variables.
            
            Output:
                beta: KxN pd.DataFrame.
            
            Raises:
                ValueError: If the determinant of the matrix is zero.
        '''
        # y is a matrix TxN and X is a matrix TxK.
        # We are running the regression y = Xb + u. 
        # Hence b = (X'X)^(-1) X' y.

        try:
            beta = np.linalg.inv(X.T@ X) @ X.T @ y # , _, _, _ = np.linalg.lstsq(X, y, rcond=None) #
            beta = pd.DataFrame(beta)
            beta.index = ["beta"+str(n) for n in range(len(beta.index))]
            if name_var is not None: beta.columns = [name_var+str(n) for n in range(starting_index,len(beta.columns)+starting_index)]
        except:
            if X.T@X.shape[0] == 0:
                raise ValueError(self.empty_input)
            if np.linalg.det(X.T@X) < 1e-5:
                raise ValueError(self.singular_matrix)
        return beta
    
    def OLS_gmm_corrected_se(self,
                             y: np.array, 
                             X: np.array, 
                             lags: int,
                             weight = 0,
                             starting_index = 1,
                             name_var = None):
        '''
            This function computes the OLS GMM corrected standard errors.

            Input:
                1. y: TxN np.array (lhs in the equation).
                2. X: TxK np.array (rhs in the equation).
                3. lags: integer for number of lags to include in GMM corrected standard errors.
                4. weights: integer taking three possible values:
                    1 for Newey-Text weighting.
                    0 for even weighting
                    2 for modified even weighting. Imposes homoskedasticity and MA(h) structure on HH to get better small sample performance
            
            Output:
                betav: regression coefficients K x N pd.DataFrame of coefficients
                se_betav: K x N pd.DataFrame of standard errors of parameters. 
                  (Note this will be negative if variance comes out negative) 
                v: variance covariance matrix of estimated parameters. If there are many y variables, the vcv are stacked vertically
                R2v:    unadjusted R2
                R2vadj: adjusted R2
                F: [Chi squared statistic    degrees of freedom    pvalue] for all coeffs jointly zero. (Nx3 pd.DataFrame)
                NOTE: program checks whether first is a constant and ignores that one for test
            
                Raises: 
                    ValueError: If the determinant of the matrix is zero.
                    ValueError: If the inputs are None
        '''
        if (y is None) or (X is None): 
            raise ValueError(self.empty_input)
        T       = y.shape[0]
        N       = y.shape[1]
        K       = X.shape[1]
        if X.shape[0] != T: raise ValueError("Number of rows of y and X do not coincide")
        if  np.linalg.det(X.T@X) != 0:
            Exxprim = np.linalg.inv(1/T*(X.T@X))
            # Calculate the betas
            beta   = self.OLS_beta(y,X,name_var)
            betav  = beta.values
            # Calculate the errors. Note that we must take .values on "beta", because it is a DataFrame
            epsv    = y - X @ betav
            # Calculate the Mean Square Error
            mse     = np.var(epsv, axis=0)
            # Calculate the variance of y
            var_yv  = np.var(y,axis=0)
            # Calculate the R2 and the adjusted one
            R2v     = self.r2(mse, var_yv)
            R2v     = self.is_float(R2v)
            R2adjv  = self.r2adj(R2v, T, K)
            R2adjv  = self.is_float(R2adjv)
            # Compute the GMM standard errors:
            se_betav = np.zeros((K,N))
            F        = np.zeros((N,3))
            
            for n in range(N):
                # Take the error and reshape it so that its number of columns is one and not None
                eps = epsv[:,n].reshape(-1,1)
                # 
                if (weight == 0) | (weight == 1):
                    
                    inner = (X * (eps @ self.ones(K).T)).T @ (X * (eps @ self.ones(K).T)) / T

                    for j in range(1,lags):
                        inneradd    = (X[:T-j,:] * (eps[:T-j,:] @ self.ones(K).T)).T @ (X[j:,:] * (eps[j:,:] @ self.ones(K).T)) / T
                        inner       += (1-weight*j/(lags+1))*(inneradd + inneradd.T)

                elif weight == 2:
                    inner = X.T @ X / T
                    for j in range(1,lags):
                        innerad = X[:(T-1)-lags,:].T @ X[j:T,:]
                        inner += (1-j/(lags))*(inneradd + inneradd.T)
                    inner = inner * np.std(eps)**2
                else:
                    continue

                var_beta    = 1/T*Exxprim@inner@Exxprim
                var_beta    = np.round(var_beta,3)

                if n == 0: Sigma = var_beta
                else: Sigma = np.hstack((Sigma, var_beta))
                se_beta         = np.diag(var_beta)
                se_beta         = np.sign(se_beta) * np.sqrt(np.abs(se_beta))
                se_betav[:,n]   = se_beta 

                # F-test:
                if np.all(X[:,0] == 1):
                    if np.linalg.det(var_beta[1:,1:])!= 0:
                        chi2val = betav[1:, n].T @ np.linalg.inv(var_beta[1:,1:]) @ betav[1:, n]
                        dof = betav[1:,0].shape[0]
                        pval = 1-chi2.cdf(chi2val,dof)
                        F[n,:] = np.array([chi2val, dof, pval])
                    else:
                        raise ValueError(self.singular_matrix)
                else:
                    if np.linalg.det(var_beta) != 0:
                        chi2val = betav[:, n].T @ np.linalg.inv(var_beta) @ betav[:, n]
                        dof = betav[:,0].shape[0]
                        pval = 1-chi2.cdf(chi2val,dof)
                        F[n,:] = np.array([chi2val, dof, pval])
                    else:
                        raise ValueError(self.singular_matrix)

            # Save se_betav as a DataFrame
            se_beta = pd.DataFrame(se_betav)
            se_beta.index = ["se"+str(n) for n in range(len(se_beta))]
            if name_var is not None: se_betav.columns = [name_var+str(n) for n in range(starting_index,len(se_beta.columns)+starting_index)]
        else:
            raise ValueError(self.singular_matrix)
        return beta, se_beta, R2v, R2adjv, Sigma, F
    
    def eig_decompose(self, A):

        eigenvalues, eigenvectors = np.linalg.eigh(A, UPLO='U')

        # Reorder eigenvectors and eigenvalues in descending order of eigenvalues
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]

        # Get Q and Lambda matrices
        Q = eigenvectors
        Lambda = np.diag(eigenvalues)
        Q = pd.DataFrame(data=Q)
        Lambda = pd.DataFrame(data=Lambda)
        return Q, Lambda
    
    def ones(self,n):
        return np.ones((n,1))
    
    def r2(self,mse,var):
        return (1-mse/var).reshape(-1,1)
    
    def r2adj(self,R2,T,K):
        return 1 - R2*(T-1)/(T-K)
    
    def t_stat(self,beta_hat,se_beta,beta=0):
        return (beta_hat - beta) / se_beta

    def is_float(self,a):
        if  (type(a) is float):
            a = np.round(a,2)
        else:
            if (a.shape[0] == 1) and (a.shape[1] == 1): 
                a = float(np.round(a,2))
        return a


    def y_and_X_as_numpy_arrays(self,y,X):
        ## We may have a pd.Series as an input. First convert it to a DataFrame
        if type(y) is pd.Series: y = y.to_frame()
        if type(X) is pd.Series: X = X.to_frame()
        ## If it's a pd.DataFrame, then convert it to a np.array. In the case of X, add first a constant if not already added
        if type(X) is pd.DataFrame:
            if np.all(X.loc[:,X.columns[0]] != 1): X = sm.add_constant(X)
            X = X.values
        if type(y) is pd.DataFrame: 
            y = y.values.reshape(len(y.index), len(y.columns))
        ## Analog thing for arrays.
        if ((type(X) is np.ndarray) or (type(X) is np.array)) and (np.all(X[:,0] != 1)): 
                X = pd.DataFrame(X)
                X = sm.add_constant(X)
                X = X.values 
        ## Manage Nx1 vectors:
        if len(y.shape) == 1: 
            y = y.reshape(len(y), 1)
        if len(y.shape) == 1:
            X = X.reshape(len(X), 1)
        return y, X
    
            


    ## ERROR MESSAGES:
    @property
    def singular_matrix(self):
        return "Determinant equal to zero. Cannot invert the matrix."
    
    @property
    def empty_input(self):
        return "You give me empty inputs."