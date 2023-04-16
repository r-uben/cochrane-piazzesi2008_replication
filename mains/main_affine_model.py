
import __init__
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from src.affine_model import AffineModel
from src.figures import Figures 
from src.irf import IRF
from src.aux import Aux as aux
from src.econometrics import Econometrics
from src.get_files import GetFiles
from src.files_headers import FilesHeaders as fh

from sklearn.linear_model import LinearRegression

def main():

    ## AFFINE MODEL
    # am = AffineModel()
    # am.create_lambdas()
    # am.affine_term_structure()
    # print(am.phi)

    ## FIGURES
    plot = Figures()
    # plot.Figure3()
    # plot.Figure4()
    # plot.Figure5()
    # plot.Figure6()    
    # plot.Figure7()
    # plot.Figure8()
    plot.Figure9()
    plot.Figure10()

    ## TABLES

if __name__ == '__main__':
    main()
