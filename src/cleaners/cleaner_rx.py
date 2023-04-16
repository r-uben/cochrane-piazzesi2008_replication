import pandas as pd

class CleanerRX(object):
    
    def __init__(self, gsw_or_fb = "gsw"):
        self._gsw_or_fb = gsw_or_fb
    
    @property
    def rx(self):
        # Condition on the GSW:
        if self.gsw_or_fb.lower() == "gsw":
            # Take the data and convert:
            _rx  = pd.read_csv("data/gsw_rx.csv", 
                            index_col=0,
                            parse_dates=True,
                            date_parser=lambda x: pd.to_datetime(x).to_period('M')
            )
        # Condition on the FB:
        elif self.gsw_or_fb.lower() == "fb":
            # Take the data and convert:
            _rx  = pd.read_csv("data/fb_rx.csv", 
                            index_col=0,
                            parse_dates=True,
                            date_parser=lambda x: pd.to_datetime(x).to_period('M')
            )
        # Error if the condition is not correct:
        else:
            raise ValueError("gsw_or_fb must be 'gsw' or 'fb'")
        return _rx
    
    @property
    def gsw_or_fb(self):
        return self._gsw_or_fb
