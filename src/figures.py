import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.affine_model import AffineModel
from src.cleaners.cleaner_f import CleanerF
from src.cleaners.cleaner_rx import CleanerRX
from src.files_headers import FilesHeaders as fh
from src.irf import IRF


class Figures(object):

    def __init__(self):
        self._am = AffineModel()
        self._irf = IRF()

        # Remember, K is the number of factors:
        self._num_factors = self._am.K

    def save_my_fig(self, fig, num_fig):
        fig.savefig("tex/fig/eps/Figure" + str(num_fig) + ".eps", format="eps")
        fig.savefig("tex/fig/Figure" + str(num_fig) + ".jpg", format="jpg")

    def Figure3(self):
        FFB = CleanerF()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        ax  = {'level': ax1, 'spread': ax2}
        for unit in ["level", "spread"]:
            if unit == "level": min_n = FFB.min_n_in_levels
            elif unit == "spread": min_n = FFB.min_n_with_spread
            else: break

            L = np.diag(self._am.Lambda)
            total_var = sum(np.diag(self._am.Lambda))
            # Plot
            for n in range(2, min_n+4):
                sigma = np.sqrt(L[n-min_n])
                fraq_sigma2= sigma**2 / total_var
                mylabel = r"$\sigma=$ {:.2f}, $\%\sigma^2 = {:.2f}$".format(sigma*100, fraq_sigma2*100)
                self._am.Q[str(n)].plot(ax=ax[unit], label=mylabel, linewidth=2)
            ax[unit].grid(linewidth=0.4)
            ax[unit].legend()
            ax[unit].set_xlabel("Maturity, years")
            ax[unit].set_ylabel("Loading (" + unit + ")")
        fig.subplots_adjust(wspace=0.3)
        self.save_my_fig(fig,3)

    def Figure4(self):
        fig = plt.figure(figsize=(13, 5), dpi=300)
        ax  = plt.gca()
        x = pd.read_csv("data/cleaned/x.csv", 
                        index_col=0,
                        parse_dates=True,
                        date_parser=lambda x: pd.to_datetime(x).to_period('M'))
        x.plot(ax=ax)
        # Plot the grid
        ax.grid(linewidth=0.4)
        ax.legend()
        fig.subplots_adjust(wspace=0.3)
        self.save_my_fig(fig,4)

    def Figure5(self):
        fig, axs = plt.subplots(nrows=2,ncols=2, dpi=300, figsize=(10, 7))
        self.plot_f_on_risk_neutral_loadings(axs)
        self.plot_f_on_X_OLS_loadings(axs)
        fig.set_tight_layout(True)
        self.save_my_fig(fig,5)

    def Figure6(self):
        fig = plt.figure(figsize=(7, 5), dpi=300)
        ax  = plt.gca()
        cov_lines = self._am.C.div(self._am.std_v, axis=1)
        scaling = [50, -6, 15, -20]
        for i, scale in enumerate(scaling): cov_lines.loc[:, cov_lines.columns[i]] = cov_lines.loc[:, cov_lines.columns[i]]*scale
        cov_lines.columns = [r"$cov\left(r, x\right)$", 
                             r"$cov\left(r, level\right)$", 
                             r"$cov\left(r, slope\right)$", 
                             r"$cov\left(r, curve\right)$"]
        cov_lines.plot(ax=ax, linewidth=2, marker='o', markersize=4)
        ax.plot(self._am.qr, linewidth=2, marker='o', markersize=4, label=r"$q_r$")
        ax.plot(self._am.qr_fitted, linewidth=2, marker='o', markersize=4, label=r"$q_r$ (fitted using the complete $\lambda_1$)")
        ax.plot(self._am.qr_fitted_level, linewidth=2, marker='o', markersize=4, label=r"$q_r$ (fitted using only $\lambda_{1,level}$)")
        ax.grid(linewidth=0.4)
        ax.legend(loc='upper left', frameon=True, framealpha=1)
        self.save_my_fig(fig,6)

    def Figure7(self):
        self.plot_irf_for_X("risk neutral", 7)

    def Figure8(self):
        self.plot_irf_for_X("real", 8)

    def Figure9(self):
        self.plot_irf_for_f_Ey1_and_x("risk neutral", 9)

    def Figure10(self):
        self.plot_irf_for_f_Ey1_and_x("real", 10)

    #s# FUNCTION TO PLOT THE LOADINGS OF FORWARD RATES ON FACTORS IN STATISTICAL MODEL (Fiugre 5 in the paper):
    def plot_f_on_X_OLS_loadings(self, axs):
        axs[0,0].plot(self._am.b["x"], marker='o', linestyle=' ', markersize=4)
        axs[0,1].plot(self._am.b["level"], marker='o', linestyle=' ', markersize=4)
        axs[1,0].plot(self._am.b["slope"], marker='o', linestyle=' ', markersize=4)
        axs[1,1].plot(self._am.b["curvature"], marker='o', linestyle=' ', markersize=4)
        for j in [0,1]:
            for i in [0,1]:
                ax = axs[i,j]
                ax.set_xlabel("Maturity")
                ax.grid(linewidth=0.5)
        
    def plot_f_on_risk_neutral_loadings(self, axs):
        axs[0,0].set_title(r"Loading $B_f$ on $x$")
        axs[0,0].plot(self._am.Bf.loc[:,"x"], linestyle='--')
        axs[0,1].set_title(r"Level")
        axs[0,1].plot(self._am.Bf.loc[:,"level"], linestyle='--')
        axs[1,0].set_title(r"Slope")
        axs[1,0].plot(self._am.Bf.loc[:,"slope"], linestyle='--')
        axs[1,1].set_title(r"Curvature")
        axs[1,1].plot(self._am.Bf.loc[:,"curvature"], linestyle='--')
        for j in [0,1]:
            for i in [0,1]:
                ax = axs[i,j]
                ax.set_xlabel("Maturity")
                ax.grid(linewidth=0.5)

    def plot_irf_for_X(self, phi, num_fig):
        if self._num_factors == 4: dist = (2,2)
        if self._num_factors == 3: dist = (1,3)
        if self._num_factors == 2: dist = (1,2)
        fig, axs = plt.subplots(nrows=dist[0], ncols=dist[1], dpi=200, figsize=(10,7))
        for i, ax in enumerate(axs.reshape(-1)):
            _X_t, _phi = self._irf.irf_X(i, phi)
            for k in range(self._num_factors):
                ax.plot(_X_t[:,k], label=_phi.columns[k], linewidth=0.8, marker='^', markersize=4)
            ax.set_title("IRF to {} shock".format(_phi.columns[i]))
            if (i==0): ax.legend(loc='upper right')
            ax.grid(linewidth=0.5)
            ax.set_xlabel("Year")
        fig.tight_layout()
        self.save_my_fig(fig, num_fig)

    def plot_irf_for_f_Ey1_and_x(self, phi, num_fig):
        if self._num_factors == 4: dist = (2,2)
        if self._num_factors == 3: dist = (1,3)
        if self._num_factors == 2: dist = (1,2)
        fig, axs = plt.subplots(nrows=dist[0], ncols=dist[1], dpi=200, figsize=(10,7))
        for i, ax in enumerate(axs.reshape(-1)):
            _X_t, _F_t, _Ey1, _phi = self._irf.irf_f_Ey1_and_x(i, phi)
            ax.plot(_X_t[:,0]/3, label="Er factor x", linewidth=0.8, marker='^', markersize=4)
            ax.plot(_F_t, label="Current forwards", linewidth=0.8, marker='^', markersize=4)
            ax.plot(_Ey1, label="Expected one year yield", linewidth=0.8, marker='^', markersize=4)
            ax.set_title("IRF to {} shock".format(_phi.columns[i]))
            if (i==0): ax.legend(loc='upper right')
            ax.grid(linewidth=0.5)
            ax.set_xlabel("Year")
        fig.tight_layout()
        self.save_my_fig(fig, num_fig)
