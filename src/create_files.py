import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from src.files_headers import FilesHeaders as fh
from src.aux import Aux as aux

class CreateFiles(object):

    def __init__(self, init_year, end_year, dataset='gsw') -> None:
        self.init_year = init_year
        self.end_year = end_year
        self.dataset = dataset
        if dataset == 'gsw': self.max_n = 15
        if dataset == 'fb': self.max_n = 5
        self.lag = 12
        self.freq = 'M'

        self.use_FED_params = False

    @property
    def y(self):
        '''
            This function calculates the yields of the Treasury Bonds.
        '''
        
        def y_gsw():
            '''
                This function calculates the yields of the Treasury Bonds using GSW (FED) data.
            '''
            mycols = [fh.y_header(n) for n in range(1,30+1)]
            y = self.gsw[mycols]
            # Cochrane divides by 100;
            y /= 100
            return y 
        
        if self.dataset == 'gsw': y = y_gsw()
        else: y = None
        y = self.take_my_window(y)
        return y 
    
    @property
    def p(self): 

        def p_gsw():
            # Firs take the yields, because GSW dataset does not contain prices.
            y = self.y
            # Dictionary to save it in a Dataframe
            p = {}
            for n in range(self.max_n+1): 
                if n == 0: p[fh.p_header(n)] = 0
                else: p[fh.p_header(n)] = -n*y[fh.y_header(n)]
            p = pd.DataFrame(p, index=y.index)
            p = self.take_my_window(p)
            p = p.dropna()
            p.to_csv("data/gsw_p.csv")
            return p

        def p_fb():
            # beg = 140 is equivalent to 1964
            # beg = 234 is equivalent to 1971:11, date for merging with 15 year maturity fed data 
            beg = 234
            P = pd.read_csv("data/fb.txt", sep=4*" ", engine="python")
            # Create the date range (see Cochrane's code for this election)
            dates = pd.date_range('1952-06-01', periods=len(P), freq='M')
            P = P.iloc[beg:]
            dates = dates[beg:]
            # New Index column with pretty date format
            P.index = dates
            # Drop the ugly one
            P = P.drop('qdate',axis=1)
            # log prices
            p = np.log(P/100)
            p.columns = [fh.p_header(n) for n in range(1,self.max_n+1)]
            # That't not necessary I think but just in case I ensure myself that the 
            # year range is correct
            p.index = p.index.to_period('M')            
            p = self.take_my_window(p)
            # Now simply save it with the "fb" prefix to indicate the dataset (Fama & Bliss, 1987)
            p.to_csv("data/fb_p.csv")
            return p

        if self.dataset == 'gsw': p = p_gsw()
        elif self.dataset == 'fb': p = p_fb()
            
        else: print("Give me a good dataset.")
        return p


    @property
    def f(self):
        '''
            This function saves and returns the forward rates (daily frequency if GSW, monthly if FB).
        '''

        def f_gsw():
            '''
            This dataset contains data OF ZERO-COUPON TREASURY YIELDS from the FED and it's used by Gürkaynak, Sack and Wright (2006). 
            The variables contained in the columns are:
                – SVENYXX: Zero-coupon yield (continuously compounded). All integers 1-30.
                – SVENPYXX: Par yield Coupon-Equivalent. All integers 1-30.
                – SVENFXX: Instantaneous forward rate (continuously compounded). All integers 1-30.
                – SVEN1FXX: One-year forward rate Coupon-Equivalent. Integers 1, 4, and 9.
                – Parameters BETA0, BETA1, BETA2, TAU1, TAU2
            '''
            
            if self.use_FED_params:
                # In the paper they say that beta3 and tau2 are only appearing starting in 1980, and jump in at unfortunately large values.
                # This fact means that there can only be one hump in the yield curve before that date. That's the reason they use to 
                # not use these parameters although they describe de data perfectly (sic). Instead, they use the yields given.
                def add_f_n(params, n):
                    beta0 = params["beta0"]
                    beta1 = params["beta1"]
                    beta2 = params["beta2"]
                    beta3 = params["beta3"]
                    tau1  = params["tau1"]
                    tau2  = params["tau2"]
                    params["f"+str(n)] = beta0 + beta1*np.exp(-n/tau1) + beta2*(n/tau1)*np.exp(-n/tau1)+beta3*n/tau2*np.exp(-n/tau2)
                    return params
                # We are only interested in the columns containing beta and tau (and the date one)
                mycols = ["Date"] + self.param_cols
                # I guess there would be some problems if I just take the mean of the parameters (which are given daily
                # in the original file). Hence I reload it again and use it daily and afterwards aggregate monthly.
                params = pd.read_csv("data/gsw.csv", usecols=mycols)
                # We want the date column to be the index and in DateTime format:
                params = aux.format_index(params)
                # Now that we have only betas and the taus in the columns, we lowerise the letters
                params.columns = [param.lower for param in self.param_cols]
                # We construct the dataset to be used with the forward rates using n=1,...,30.
                for n in range(1,30+1): params = add_f_n(params,n)
                f = params[[fh.f_header(n) for n in range(1,30+1)]]
                # Now monthly
                f = f.resample('M').mean()
                f = f.to_period('M')

                # Plot the parameters:
                fig = plt.figure()
                ax = fig.gca()
                for param in params.columns:
                    params[param].plot(ax=ax, label=param)
                ax.grid(linewidth=0.4)
                ax.legend()
                fig.savefig("fig/FED_params.jpg", format='jpg')
            
            else:
                f = {} 
                p = self.p
                for n in range(1,self.max_n+1):
                    if n == 1: f[fh.f_header(n)] = self.y[fh.y_header(1)]
                    else: f[fh.f_header(n)] = p[fh.p_header(n-1)] -  p[fh.p_header(n)]
                f = pd.DataFrame(f, index=self.p.index)
            f = self.take_my_window(f)
            f = f.dropna()
            f.to_csv("data/gsw_f.csv")
            return f
        
        def f_fb():
            p = pd.read_csv("data/fb_p.csv", index_col=0)
            p.index = pd.to_datetime(p.index)
            f = {}
            for n in range(1,5+1):
                if n == 1: f[fh.f_header(n)] = -p[fh.p_header(n)]
                else: f[fh.f_header(n)] = p[fh.p_header(n-1)]-p[fh.p_header(n)]
            f = pd.DataFrame(f)
            f = aux.format_index(f)
            f.index = f.index.to_period('M')
            f = self.take_my_window(f)
            f = f.dropna()
            f.to_csv("data/fb_f.csv")
            return f

        if self.dataset == 'gsw': f = f_gsw()
        elif self.dataset == 'fb': f = f_fb()
        else: 
            f = None
            print("This dataset is not available.")
        return f
    

    @property
    def r(self):
        p = self.p
        r = {}
        for n in range(1,self.max_n+1): 
            if n == 1: r[fh.r_header(n)] =  - p[fh.p_header(n)].shift(self.lag, freq=self.freq)
            else: r[fh.r_header(n)] = p[fh.p_header(n-1)] - p[fh.p_header(n)].shift(self.lag, freq=self.freq)
        r = pd.DataFrame(r, index=p.index)
        r = self.take_my_window(r)
        r = r.dropna()
        r.to_csv("data/" + self.dataset + "_r.csv")
        return r

    @property
    def rx(self):
        # First: load the two files that we need to compute the excess returns:
        #       1. File of prices.
        #       2. File of returns
        p = self.p
        r = self.r
        # Second: get the spot rate, that is, one-year yield. This can be computed 
        # by means of prices. Note that when calling it from the dataframe p,
        # what we get is a 'pandas.core.series.Series', not a 'pandas.core.frame.DataFrame'.
        # We do not convert it to frame to prevent errors in the new file.
        y1 = - p[fh.p_header(1)]
        # We lag yearly
        y1_shifted = y1.shift(12, freq='M').iloc[1:]
        # Third: Create the output by first creating a dictionary that will be 
        # eventually converted to a 'pandas.core.frame.DataFrame'.
        rx = {}
        for n in range(2,self.max_n+1): rx[fh.rx_header(n)] = r[fh.r_header(n)] - y1_shifted
        rx = pd.DataFrame(rx, columns=self.cols("rx"), index=r.index)
        rx = self.take_my_window(rx)
        rx = rx.dropna()
        # Fourth: Save it
        rx.to_csv("data/" + self.dataset + "_rx.csv")
        return rx

    @property
    def gsw(self):
        '''
            This function takes the GSW data and cleans it monthly (taking the monthly mean in every column)
        '''
        gsw = pd.read_csv("data/gsw.csv", index_col=0, parse_dates=True, date_parser=lambda x: pd.to_datetime(x))
        gsw = gsw.resample('M').mean()
        gsw = gsw.to_period('M')
        return gsw
    
    @property
    def beta_cols(self):
        beta_cols = ["BETA" + str(i) for i in range(3+1)]
        return beta_cols
    
    @property
    def tau_cols(self):
        tau_cols  = ["TAU" + str(i) for i in range(1,2+1)]
        return tau_cols
    
    @property
    def param_cols(self):
        return self.beta_cols + self.tau_cols
    
    def cols(self, var):
        return [var+str(n) for n in range(2,self.max_n+1)]
    
    def take_my_window(self,df):
        df = df[(df.index.year >= self.init_year) & (df.index.year <= self.end_year)]
        return df