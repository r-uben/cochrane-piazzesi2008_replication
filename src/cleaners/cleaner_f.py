import pandas as pd

from src.files_headers import FilesHeaders as fh

class CleanerF(object):

    def __init__(self, gsw_or_fb = "gsw"):
        self._gsw_or_fb = gsw_or_fb
    
        self._f_levels = None
        self._f_spread = None

        self._x = None

    ## UPDATE F AS A ROLLING WINDOW
    def MA_f(self, num_months=3):
        self._f_levels = self._f_levels.copy()
        self._f_levels = self._f_levels.rolling(num_months).mean()
        self._f_levels = self._f_levels.dropna()

    ## -----------------------------------------------------------------
    ## LOAD THE DATA. Remember that it depends on whether we are using GSW or FB.
    def load_f(self):
        # Condition on the GSW:
        if self.gsw_or_fb.lower() == "gsw":
            # Take the data and convert:
            self._f_levels  = pd.read_csv("data/gsw_f.csv", 
                            index_col=0,
                            parse_dates=True,
                            date_parser=lambda x: pd.to_datetime(x).to_period('M')
            )
        # Condition on the FB:
        elif self.gsw_or_fb.lower() == "fb":
            # Take the data and convert:
            self._f_levels  = pd.read_csv("data/fb_f.csv", 
                            index_col=0,
                            parse_dates=True,
                            date_parser=lambda x: pd.to_datetime(x).to_period('M')
            )
        # Error if the condition is not correct:
        else:
            raise ValueError("gsw_or_fb must be 'gsw' or 'fb'")
    
    ## FUNCTION TO TAKE THE FORWARD RATE DATA
    def clean_f(self):
        self.MA_f()
        # Substract it
        self._f_spread = self.f_levels.sub(self.y1,axis=0)[[fh.f_header(n) for n in range(self.min_n_with_spread,len(self.f_levels.columns)+1)]]
        self.f_levels.to_csv("data/cleaned/f_levels.csv", index=False)
        self.f_spread.to_csv("data/cleaned/f_spread.csv", index=False)
    
    ## ----------------------------------------------------------------
    ## MAIN FUNCTION: LOAD AND CLEAN THE FORWARD RATE DATA
    def load_and_clean(self):
        if self.f_levels is None: self.load_f()
        self.clean_f()  

    ## ----------------------------------------------------------------
    ## PROPERTIES:
    @property
    def min_n_with_spread(self):
        return 2
    
    @property
    def min_n_in_levels(self):
        return 1
    
    @property
    def f_spread(self):
        return self._f_spread

    @property
    def f_levels(self):
        return self._f_levels
    
    @property
    def y1(self):
        y1 = self.f_levels[fh.f_header(n=1)]
        return y1
    @property
    def the_idx(self):
        return self.f_levels.index
    
    @property
    def gsw_or_fb(self):
        return self._gsw_or_fb

    @property
    def level_or_spread(self):
        return self._level_or_spread
