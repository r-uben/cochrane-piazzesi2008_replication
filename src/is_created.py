import pandas as pd
from src.create_files import CreateFiles

class IsCreated:

    def __init__(self, init_year, end_year, dataset='gsw') -> None:
        self.init_year = init_year
        self.end_year = end_year
        self.dataset = dataset
    
    
    @property
    def p(self):
        create = CreateFiles(self.init_year, self.end_year, self.dataset)
        p = None
        while True:
            if p is not None: break
            try: p = pd.read_csv("data" + self.dataset + "_p.csv", index_col=0)
            except FileNotFoundError: p=create.p
        return p
    
    @property
    def f(self):
        create = CreateFiles(self.init_year, self.end_year, self.dataset)
        f = None
        while True:
            if f is not None: break
            try: f = pd.read_csv("data/" + self.dataset + "_f.csv", index_col=0)
            except FileNotFoundError: f=create.f
        return f
    
    @property
    def r(self):
        create = CreateFiles(self.init_year, self.end_year, self.dataset)
        r = None
        while True:
            if r is not None: break
            try: r = pd.read_csv("data/" + self.dataset + "_r.csv", index_col=0)
            except FileNotFoundError: r=create.r
        return r
    
    @property
    def rx(self):
        create = CreateFiles(self.init_year, self.end_year, self.dataset)
        rx = None
        while True:
            if rx is not None: break
            try: rx = pd.read_csv("data/" + self.dataset + "_rx.csv", index_col=0)
            except FileNotFoundError: rx=create.rx
        return rx

