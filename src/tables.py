import pandas as pd


from src.affine_model import AffineModel    


class Tables:

    def __init__(self):
        self._am = AffineModel()

        self.tex_path = "/Users/ruben/Dropbox/library/replications/cochrane-piazzesi2008/tex/"
        self.factors = ["x", "level", "slope", "curvature"]

    def save_table(self, num, table_code):
        with open(self.tex_path + "tab/table" + str(num) + ".txt", "w") as file: file.write(table_code)

    def Table4(self):
        # Estimates of model dynamics, μ and ϕ in X_{t+1} = μ + ϕ X_t + v_{t+1}
        # under the OLS benchmark:
        _phi    = self._am.phi
        _mu     = self._am.mu     
        # Estimates of model dynamics, μ and ϕ in X_{t+1} = μ + ϕ X_t + v_{t+1}
        # under risk neutrality:
        _phi_star = self._am.phistar
        _mu_star  = self._am.mustar
        # Construct the table:
        table_code  = '\\begin{tabular}{r|ccccc}\n'
        table_code += '                     &   $100\\times\mu$ &  $x$   &  level   &   slope   &   curvature \\\\ \n'
        table_code += '\\hline\n'
        table_code += ' Risk Neutrality     &   $\mu^\\star$    & \\multicolumn{ 4 }{ c }{ $\phi^\star$}\\\\\n'
        table_code += '\\hline\n'
        for i, factor in enumerate(self.factors):
            _mu_factor  = round(_mu_star.iloc[i]*100, 2)
            _phi_x      = round(_phi_star.loc[factor, "x"], 2)
            _phi_level  = round(_phi_star.loc[factor, "level"], 2)
            _phi_slope  = round(_phi_star.loc[factor, "slope"], 2)
            _phi_curvature = round(_phi_star.loc[factor, "curvature"], 2)
            table_code += ' {} & {} & {} & {} & {} & {}\\\\ \n'.format(factor, float(_mu_factor), float(_phi_x), _phi_level, _phi_slope, _phi_curvature)

        table_code += ' aCTUAL     &   $\mu$    & \\multicolumn{ 4 }{ c }{ $\phi$}\\\\\n'
        table_code += '\\hline\n'
        for i, factor in enumerate(self.factors):
            _mu_factor  = round(_mu.iloc[i]*100, 2)
            _phi_x      = round(_phi.loc[factor, "x"], 2)
            _phi_level  = round(_phi.loc[factor, "level"], 2)
            _phi_slope  = round(_phi.loc[factor, "slope"], 2)
            _phi_curvature = round(_phi.loc[factor, "curvature"], 2)
            table_code += ' {} & {} & {} & {} & {} & {} \\\\ \n'.format(factor, float(_mu_factor), float(_phi_x), _phi_level, _phi_slope, _phi_curvature)
        self.save_table(4, table_code)

    def Table5(self):
        # Take the innovations of the factor X dynamics:
        _v = self._am.v
        # Compute the correlation matrix:
        _corr_v = _v.corr()
        # Construct the table:
        table_code = '\\begin{tabular}{r|cccc}\n'
        table_code +=' & {} & {} & {} & {} \\\\\n'
        table_code += '\\hline\n'
        for i, factor1 in enumerate(self.factors):
            table_code += ' {} & '.format(factor1)
            for j, factor2 in enumerate(self.factors):
                _corr_f1_f2 = _corr_v.loc[factor1, factor2]
                if i <= j: _corr_f1_f2 = round(_corr_f1_f2, 2)
                else: _corr_f1_f2 = ' '
                if j < len(self.factors)-1: table_code += ' {} &'.format(_corr_f1_f2)
                else: table_code +='{} \\\\\n'.format(_corr_f1_f2)
        self.save_table(5, table_code)

    def Table(self, num_table): 
        if num_table == 4: self.Table4()
        if num_table == 5: self.Table5()