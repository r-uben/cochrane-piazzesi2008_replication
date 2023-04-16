
import __init__
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from src.aux import Aux as aux
from src.econometrics import Econometrics
from src.get_files import GetFiles
from src.files_headers import FilesHeaders as fh

from sklearn.linear_model import LinearRegression

# Global
metrics = Econometrics()
tex_path = "/Users/ruben/Dropbox/library/replications/cochrane-piazzesi2008/tex/"

## FUNCTIONS FOR CONSTRUCTING THE RETURN-FORECASTING FACTOR

def take_min_n_if_level_or_spread(var):
    if var == 'spread': min_n = 2
    elif var == 'level': min_n = 1
    else: 
        raise ValueError("Possible models are in levels or spreads.")
    return min_n

def f_as_three_month_window_MA(f):
    f = f.rolling(3).mean()
    f = f.dropna()
    return f

def f_as_spread(f, order='level'):
    # Take the spot rate
    y1 = f[fh.f_header(n=1)]

    # We follow the specification in eq 21, so we only use spread information (p.15)
    if order == 'spread': f = f.sub(y1,axis=0)[[fh.f_header(n) for n in range(2,len(f.columns)+1)]]
    return f

def clean_f(f, unit='level'):
    min_n = take_min_n_if_level_or_spread(unit)
    f = f_as_three_month_window_MA(f)
    f_levels = f
    # Spread or level option
    if unit == 'spread': 
        # Calculate the spot rate
        y1 = f[fh.f_header(n=1)]
        # Substract it
        f = f.sub(y1,axis=0)[[fh.f_header(n) for n in range(min_n,len(f.columns)+1)]]
    return f, f_levels

# FUNCTIONS TO CONSTRUCT THE LEVEL, SLOPE AND CURVATURE FACTORS

def z(i, Q, f):
    idx = f.index
    if type(Q) == pd.DataFrame: Q = Q.values    
    if type(f) == pd.DataFrame: f = f.values
    if i==0: name = "level"
    elif i==1: name = "slope"
    elif i==2: name = "curvature"
    else: 
        name = "f" + str(i)
        i-=1
    z = f @ Q[:,i]
    return pd.DataFrame(z.reshape(-1,1), index=idx, columns=[name])

def f_level(Q, f):
    return z(0,Q,f)

def f_slope(Q,f):
    return z(1,Q,f)

def f_curvature(Q,f):
    return z(2,Q,f)

# FUNCTIONS TO CONSTRUCT THE DATASETS AND DO SOME ANALYSIS OF THE FACTOR MODELS

def get_my_files(dataset):
    init_year = 1971
    end_year = 2006
    get = GetFiles(init_year, end_year, dataset)
    if dataset == 'gsw':
        get.f()
        get.p()
    elif dataset == 'fb':
        get.p()
        get.f()
    else: print('Give me a correct dataset')
    get.r()
    get.rx()

def get_rxbar(dataset):
    rx = pd.read_csv("data/" + dataset + "_rx.csv", index_col=0, parse_dates=True, date_parser=lambda x: pd.to_datetime(x).to_period('M'))
    rxbar = rx.mean(axis=1).to_frame()
    rxbar.columns = ["rxbar"]
    return rxbar

def get_R2(X,Y):
    X = X.dropna()
    Y = Y.dropna()
    X,Y = aux.take_same_index(X,Y)
    model = LinearRegression()
    model.fit(X, Y)
    r2 = model.score(X, Y)
    return r2.round(3)

def get_coeffs(X, Y):
    model = LinearRegression()
    model.fit(X, Y)
    return model.coef_

def R2_f1tof5(df):
    Y = df["rxbar"] 
    X = df[[fh.f_header(n) for n in range(1,5+1)]]
    r2 = get_R2(X,Y)
    return r2

def R2_f1f3f5(df):
    Y = df["rxbar"]
    X = df[[fh.f_header(n) for n in range(1,5+1) if n%2!=0]]
    r2 = get_R2(X,Y)
    return r2

def R2_f1tof15(df):
    Y = df["rxbar"]
    X = df[[fh.f_header(n) for n in range(1,15+1)]]
    r2 = get_R2(X,Y)
    return r2

def get_df_for_OLS(lhs, rhs):
    rxbar = get_rxbar(lhs)
    rxbar.index = pd.to_datetime(rxbar.index.to_timestamp()).to_period('M')
    # We get also the forward rate dataframe
    if 'y1' in rhs: rhs = 'fb' 
    f = pd.read_csv("data/" + rhs + "_f.csv", index_col=0)
    f.index = pd.to_datetime(f.index).to_period('M')
    # We are only interested in the first 15. We shift them because we are gonna run predictive regressions
    # Also, we drop the first row because it contains NaNs.
    if rhs == 'gsw': max_n = 15
    elif rhs == 'fb': max_n = 5
    else: print('We do not have this dataset.')
    # We have aggregated the data so that everything is monthly
    lag = 12 
    freq = 'M'
    f = f[[fh.f_header(n) for n in range(1,max_n+1)]].shift(lag, freq=freq).dropna()
    # We merge them to have just one dataframe. But first we have to be sure that the lengths are the same:
    if (lhs=='gsw') and (rhs=="fb_minus_y1"):
        p = pd.read_csv("data/fb_p.csv", index_col=0)
        p.index = pd.to_datetime(p.index).to_period('M')
        y1 = - p[fh.p_header(1)]
        y1_shifted = y1.shift(12, freq='M').iloc[1:]
        f = f - y1_shifted
    df =  pd.merge(rxbar, f, how='outer', left_index=True, right_index=True)
    # We drop NaNs to avoid problems.
    df.index = pd.to_datetime(df.index.to_timestamp()).to_period('M')
    return df, rxbar, f

def get_MA_3M_df(rxbar, f):
    MA_3M_df=  pd.merge(rxbar, f, how='outer', left_index=True, right_index=True)
    MA_3M_df = MA_3M_df.dropna()
    MA_3M_df.index = pd.to_datetime(MA_3M_df.index.to_timestamp())
    MA_3M_df = MA_3M_df[(MA_3M_df.index.year >= 1971) & (MA_3M_df.index.year <= 2006)]
    return MA_3M_df

def get_se(se_beta:pd.DataFrame, i:int):
    return float(np.round(se_beta.loc["se" + str(i),0],2))

## PAPER MAIN STUFF:

def one_year_return_forecasts_mixing_FB_and_GSW_data():
    
    gsw_on_gsw_df, gsw_on_gsw_rxbar, gsw_on_gsw_f   = get_df_for_OLS(lhs="gsw", rhs="gsw")
    gsw_on_fb_df, gsw_on_fb_rxbar, gsw_on_fb_f      = get_df_for_OLS(lhs="gsw", rhs="fb")
    fb_on_fb_df, fb_on_fb_rxbar, fb_on_fb_f         = get_df_for_OLS(lhs="fb", rhs="fb")
    gsw_on_fb_minus_spot_df, gsw_on_fb_minus_spot_rxbar,  gsw_on_fb_minus_spot_f  = get_df_for_OLS(lhs="gsw", rhs="fb_minus_y1")

    # We do the regressions and take the R2:
    table_code  = '\\begin{tabular}{r|cc|c|c}\n'
    table_code += '                     &   $rx_{t+1}^{GSW}$       & on            & $rx_{t+1}^{FB}$ on    & $rx_{t+1}^{FB}$ on \\\\\n'
    table_code += '                     &   $f_t^{GSW}$            & $f_t^{FB}$    & $f_t^{FB}$            & $f_t^{FB} - y_t^{(1)}$\\\\\n'
    table_code += '\\hline\n'
    table_code += ' f1-f5               &   {}                  & {}          & {}                 & {}            \\\\\n'.format(R2_f1tof5(gsw_on_gsw_df),R2_f1tof5(gsw_on_fb_df), R2_f1tof5(fb_on_fb_df), R2_f1tof5(gsw_on_fb_minus_spot_df)) 
    table_code += ' f1-f3-f5            &   {}                  & {}          & {}                  & {}            \\\\\n'.format(R2_f1f3f5(gsw_on_gsw_df), R2_f1f3f5(gsw_on_fb_df), R2_f1f3f5(fb_on_fb_df), R2_f1f3f5(gsw_on_fb_minus_spot_df)) 
    table_code += ' f1-f15              &   {}                  &               &                       &               \\\\\n'.format(R2_f1tof15(gsw_on_gsw_df)) 

    gsw_on_gsw_f = gsw_on_gsw_f.dropna().rolling(window=90).mean()
    gsw_on_fb_f = gsw_on_fb_f.dropna().rolling(window=3).mean()
    fb_on_fb_f = fb_on_fb_f.dropna().rolling(window=3).mean()
    gsw_on_fb_minus_spot_f = gsw_on_fb_minus_spot_f.dropna().rolling(3).mean()

    MA_3M_gsw_on_gsw_df =  get_MA_3M_df(gsw_on_gsw_rxbar, gsw_on_gsw_f)
    MA_3M_gsw_on_fb_df =  get_MA_3M_df(gsw_on_fb_rxbar, gsw_on_fb_f)
    MA_3M_fb_on_fb_df =  get_MA_3M_df(fb_on_fb_rxbar, fb_on_fb_f)
    MA_3M_gsw_on_fb_minus_spot_df =  get_MA_3M_df(gsw_on_fb_minus_spot_rxbar, gsw_on_fb_minus_spot_f)
    # We drop NaNs to avoid problems.
    table_code += ' 3 mo. MA f1-f5      &   {}                  & {}         & {}                 & {}            \\\\\n'.format(R2_f1tof5(MA_3M_gsw_on_gsw_df), R2_f1tof5(MA_3M_gsw_on_fb_df), R2_f1tof5(MA_3M_fb_on_fb_df), R2_f1tof5(MA_3M_gsw_on_fb_minus_spot_df)) 
    table_code += ' 3 mo. MA f1-f15     &   {}                  &               &                       &               \\\\\n'.format(R2_f1tof15(MA_3M_gsw_on_gsw_df)) 
    table_code += '\\hline\n'
    table_code += '\\end{tabular}'
    with open(tex_path + "tab/table1.txt", "w") as file: file.write(table_code)

def return_forecasting_factor_x():
    # Load the excess returns and the forward rates
    raw_rx = pd.read_csv("data/gsw_rx.csv", index_col=[0], parse_dates=True, date_parser=lambda x : pd.to_datetime(x).to_period('M')).copy()
    raw_f = pd.read_csv("data/fb_f.csv", index_col=0, parse_dates=True, date_parser=lambda x: pd.to_datetime(x).to_period('M')).copy()
    fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    ax3  = {'level': ax3_1, 'spread': ax3_2}
    fig4 = plt.figure(figsize=(13, 5), dpi=200)
    ax4  = plt.gca()
    x = {}
    count=1
    for unit in ["level", "spread"]:
        # Reload rx
        rx = raw_rx
        # Calculate the three month MA of the five FB forward rates and again put beautiful index
        f, f_levels = clean_f(raw_f,unit)
        the_idx = f.index
        # Lag it: (The orden here is very important, otherwise you get bad results because of double lagging or something similar)
        f_no_shift = f # Save the no shifted f for later
        f   = f.shift(12,freq='M')
        # We now take the inresection so we have a good dataframe without nans.
        rx, f = aux.take_same_index(rx,f)
        # Create a new dataframe for the dependent variable, rx_(t+1)
        y = rx
        # Create a new dataframe for the independent variable, f_t (shifted 12 months, i.e., 1 year)
        X = f
        # We create a "big X", i.e., without the dates lost from shifting, to being able to create the 
        # adequate E[rx]:
        bigX = f_no_shift
        # Run the regression    
        y, X = metrics.y_and_X_as_numpy_arrays(y,X)
        _, bigX = metrics.y_and_X_as_numpy_arrays(y,bigX)
        alpha_and_beta = metrics.OLS_beta(y,X)
        alpha = alpha_and_beta.T["beta0"].values.reshape(-1,1)
        beta  = alpha_and_beta.T[["beta" + str(n) for n in range(1,len(alpha_and_beta))]].values
        # Calculate Expected returns:
        Erx = bigX @ alpha_and_beta.values
        # Calculate the covariance matrix  
        cov_rx = np.cov(Erx, rowvar=False)
        # We decompose (eigenvalues & eigenvectors, ordering them)
        Q, Lambda = metrics.eig_decompose(cov_rx)
        Q.index = [col.replace("rx","") for col in rx.columns]
        Q.columns = [col.replace("rx","") for col in rx.columns]
        Q.to_csv("data/Q_cov_Erx_" + unit + ".csv")
        Lambda.index = Q.index
        Lambda.columns= Q.columns
        Lambda.to_csv("data/Lambda.csv")
        
        # Same with f:
        cov_f = np.cov(f, rowvar=False)
        Qf, Lambdaf = metrics.eig_decompose(cov_f)
        Qf.index  = [col.replace("f","") for col in f.columns]
        Qf.columns = [col.replace("f","") for col in f.columns]
        Qf.to_csv("data/Q_cov_f_" + unit + ".csv")

        # Expected returns in a DataFrame to be saved:
        Erx = pd.DataFrame(Erx)
        Erx.columns = rx.columns
        Erx.index = f_no_shift.index
        Erx.to_csv("data/Erx_" + unit + ".csv")

        # Express results in terms of regression on levels of f
        if unit == 'spread':
            beta = np.concatenate((
            -np.sum(beta, axis=1, keepdims=True),
            beta
        ), axis=1)
        # The return forcasting factor:
        qr = Q[Q.columns[0]].values.reshape(-1,1)
        gamma = (beta.T@qr).reshape(-1,1)
        fvals = f_levels.values
        x_vec = qr.T@alpha + fvals @ gamma
        # This is to avoid ValueError when constructing the DataFrame
        x[unit] =  pd.DataFrame(x_vec.reshape(-1), index=the_idx)

        L = np.diag(Lambda)
        total_var = sum(np.diag(Lambda))
        # Plot
        min_n = take_min_n_if_level_or_spread(unit)
        for n in range(2, min_n+4):
            sigma = np.sqrt(L[n-min_n])
            fraq_sigma2= sigma**2 / total_var
            mylabel = r"$\sigma=$ {:.2f}, $\%\sigma^2 = {:.2f}$".format(sigma*100, fraq_sigma2*100)
            Q[str(n)].plot(ax=ax3[unit], label=mylabel, linewidth=0.8)
        ax3[unit].grid(linewidth=0.4)
        ax3[unit].legend()
        ax3[unit].set_xlabel("Maturity, years")
        ax3[unit].set_ylabel("Loading (" + unit + ")")
        count+=1

         ## CHECK WHETHER THE RETURN-FORECASTING FACTOR IS CORRECT:
        _x = x[unit].shift(12, freq='M')
        y, X = aux.take_same_index(rx, _x)
        gamma0, beta, _, _, _, _, _, _ = aux.OLS(y,X,0)
        for i, rxi in enumerate(rx.columns):
            print((float(qr[i])*X.values.reshape(-1,1)).shape)
            new_X = float(gamma0[i]) + float(qr[i]*X
            alpha, beta, _, _, _, _, _, _ = aux.OLS(y[rxi],new_X, 0)
            #print(float(alpha), float(beta))
            
    # Save the file containing the returnsforecasting factor constructed both using levels and spreads
    x = pd.concat(x.values(), axis=1)
    x.columns = ["level", "spread"]
    x.plot(ax=ax4)
    # Plot the grid
    ax4.grid(linewidth=0.4)
    ax4.legend()
    x = x.dropna()
    x.to_csv("data/x.csv")
    fig3.subplots_adjust(wspace=0.3)
    fig3.savefig("fig/eps/Figure3.eps", format="eps")
    fig3.savefig("fig/Figure3.jpg", format="jpg")
    fig4.subplots_adjust(wspace=0.3)
    fig4.savefig("fig/eps/Figure4.eps", format="eps")
    fig4.savefig("fig/Figure4.jpg", format="jpg")
    # return x

def return_forecasting_factor_against_conventional_factors():
    # Load the files
    rx0 = pd.read_csv("data/gsw_rx.csv", index_col=[0], parse_dates=True,  date_parser=lambda x: pd.to_datetime(x).to_period('M')).copy()
    f0 = pd.read_csv("data/fb_f.csv", index_col=0, parse_dates=True, date_parser=lambda x: pd.to_datetime(x).to_period('M')).copy()
    
    # We are gonna do it both in levels and in spreads
    #for unit in ["level", "spread"]:
    unit = "level"
    rxbar = get_rxbar("gsw")
    # Same with forward rates but computing also the rolling average (three months window)
    f = f0
    f, _ = clean_f(f, unit)
    # We have previously computed the matrix of covariances of forward rates and its
    # eigenvector decomposition, so we just download it
    Q = pd.read_csv("data/Q_cov_f_" + unit + ".csv", index_col=0)
    # We load the return-forecasting factor:
    x = pd.read_csv("data/x.csv", index_col=0, date_parser=lambda x: pd.to_datetime(x).to_period('M'))
    x = x.loc[:,unit].to_frame()
    x.columns = ["x"]
    level = f_level(Q, f)
    slope = f_slope(Q, f)
    curvature = f_curvature(Q, f)
    f4 = z(4,Q,f)
    # Find intersection of indexes to make things work
    rxbar, f = aux.take_same_index(rxbar, f)
    x, f = aux.take_same_index(x,f)
    if unit == "level": f5 = z(5,Q,f)
    else: f5 = None
    y = rxbar
    X = pd.concat([x, level, slope, curvature, f4, f5], axis=1)
    X = X.shift(12,freq='M')
    y, X = aux.take_same_index(y,X)
    df = pd.concat([y, X], axis=1)
    df = df.dropna()
    # Now we perform the regressions and save them in the table
    table_code = '\\begin{tabular}{rcccccccc}\n'
    table_code += '\\hline\n'
    _ , beta, se_beta, t_stat, R2, R2adj, Sigma, F = aux.OLS(df["rxbar"], df["x"]) 
    beta = aux.is_float(beta)
    se1 = get_se(se_beta,1)
    t_stat = aux.is_float(t_stat)
    table_code += ' & $x_t$ & level & slope & curve & $z_4$ & $z_5$ & $R^2$ \\\\ \\hline\n'
    table_code += ' & {} & & & & & & {} \\\\ \n'.format(float(beta), R2)
    table_code += ' & ({}) & & & & & &  \\\\ \n'.format(t_stat)
    table_code += ' & [{}] & & & & & &  \\\\ \n'.format(se1)
    _ , beta, se_beta, t_stat, R2, R2adj, Sigma, F = aux.OLS(df["rxbar"], df["level"])
    beta = aux.is_float(beta)
    se_level = get_se(se_beta,1)
    t_stat = aux.is_float(t_stat)
    table_code += ' &  & {} & & & & & {} \\\\ \n'.format(beta, R2)
    table_code += ' & & ({}) & & & & & \\\\ \n'.format(t_stat)
    table_code += ' & & [{}] & & & & & \\\\ \n'.format(se_level)
    _ , beta, se_beta, t_stat, R2, R2adj, Sigma, F = aux.OLS(df["rxbar"], df[["level", "slope"]])
    beta = np.round(beta[0],2)
    se_level = get_se(se_beta,1)
    se_slope = get_se(se_beta,2)
    t_level  = np.round(float(t_stat[0]),2)
    t_slope  = np.round(float(t_stat[1]),2)
    table_code += ' & & {} & {} & & & & {} \\\\ \n'.format(beta[0], beta[1], R2)
    table_code += ' & & ({}) & ({}) & & & & \\\\ \n'.format(t_level,t_slope)
    table_code += ' & & [{}] & [{}] & & & & \\\\ \n'.format(se_level,se_slope)
    _ , beta, se_beta, t_stat, R2, R2adj, Sigma, F = aux.OLS(df["rxbar"], df[["level", "slope", "curvature"]])
    beta = np.round(beta[0],2)
    se_level = get_se(se_beta,1)
    se_slope = get_se(se_beta,2)
    se_curve = get_se(se_beta,3)
    t_level  = np.round(float(t_stat[0]),2)
    t_slope  = np.round(float(t_stat[1]),2)
    t_curve  = np.round(float(t_stat[2]),2)
    table_code += ' & & {} & {} & {} & &  & {} \\\\ \n'.format(beta[0], beta[1], beta[2], R2)
    table_code += ' & & ({}) & ({}) & & & & \\\\ \n'.format(se_level, se_slope, se_curve)
    table_code += ' & & [{}] & [{}] & [{}] & & & \\\\ \n'.format(t_level, t_slope, t_curve)
    _ , beta, se_beta, t_stat, R2, R2adj, Sigma, F = aux.OLS(df["rxbar"], df[["level", "slope", "curvature", "f4"]])
    beta = np.round(beta[0],2)
    se_level = get_se(se_beta,1)
    se_slope = get_se(se_beta,2)
    se_curve = get_se(se_beta,3)
    se_f4    = get_se(se_beta,4)
    t_level  = np.round(float(t_stat[0]),2)
    t_slope  = np.round(float(t_stat[1]),2)
    t_curve  = np.round(float(t_stat[2]),2)
    t_f4     = np.round(float(t_stat[3]),2)
    table_code += ' & & {} & {} & {} & {}& & {} \\\\ \n'.format(beta[0], beta[1], beta[2], beta[3], R2)
    table_code += ' & & ({}) & & & & & \\\\ \n'.format(se_level, se_slope, se_curve, se_f4)
    table_code += ' & & [{}] & [{}] & [{}] & [{}] & & \\\\ \n'.format(t_level, t_slope, t_curve, t_f4)
    _ , beta, se_beta, t_stat, R2, R2adj, Sigma, F = aux.OLS(df["rxbar"], df[["level", "slope", "curvature", "f4", "f5"]])
    beta = np.round(beta[0],2)
    se_level = get_se(se_beta,1)
    se_slope = get_se(se_beta,2)
    se_curve = get_se(se_beta,3)
    se_f4    = get_se(se_beta,4)
    se_f5    = get_se(se_beta,5)
    t_level  = np.round(float(t_stat[0]),2)
    t_slope  = np.round(float(t_stat[1]),2)
    t_curve  = np.round(float(t_stat[2]),2)
    t_f4     = np.round(float(t_stat[3]),2)
    t_f5     = np.round(float(t_stat[4]),2)
    table_code += ' & & {} s& {} & {} & {}& {} & {}\\\\ \n'.format(beta[0], beta[1], beta[2], beta[3], beta[4], R2)
    table_code += ' & & ({}) & ({}) & ({}) & ({})& ({}) & \\\\ \n'.format(se_level, se_slope, se_curve, se_f4, se_f5)
    table_code += ' & & [{}] & [{}] & [{}] & [{}] & [{}] & \\\\ \n'.format(t_level, t_slope, t_curve, t_f4, t_f5)
    table_code += '\\hline\n'
    table_code += '\\end{tabular}'
    with open(tex_path + "tab/table2.txt", "w") as file: file.write(table_code)

def main():

    ## CLEANING THE DATA
    #get_my_files("gsw")
    #get_my_files("fb")

    ## SECTION 4.2.: ONE-YEAR RETURN FORECASTS MIXING FB AND GSW DATA. TABLE 1
    one_year_return_forecasts_mixing_FB_and_GSW_data()

    ## SECTION 4.3.: THE RETURN-FORECASTING FACTOR X. FIGURES 3 AND 4
    return_forecasting_factor_x()

    ## SECTION 4.4.: THE RETURN-FORECAST FACTOR IS NOT SUBSUMED IN LEVEL, SLOPE AND CURVATURE. TABLE 2
    return_forecasting_factor_against_conventional_factors()

    ## SECTION 4.5.: CONSTRUCTING FACTORS

if __name__ == "__main__":
    main()
