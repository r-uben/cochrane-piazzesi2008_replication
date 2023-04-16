
from src.is_created import IsCreated


class GetFiles:

    def __init__(self, init_year, end_year, dataset) -> None:
        self.init_year = init_year
        self.end_year = end_year
        self.dataset = dataset
    
    def p(self): 
        is_created = IsCreated(self.init_year, self.end_year, self.dataset)
        p = is_created.p
        p.to_csv("data/" + self.dataset + "_p.csv")

    def f(self):
        is_created = IsCreated(self.init_year, self.end_year, self.dataset)
        f = is_created.f
        f.to_csv("data/" + self.dataset + "_f.csv")

    def r(self):
        is_created = IsCreated(self.init_year, self.end_year, self.dataset)
        r = is_created.r
        r.to_csv("data/" + self.dataset + "_r.csv")

    def rx(self):
        is_created = IsCreated(self.init_year, self.end_year, self.dataset)
        rx = is_created.rx
        rx.to_csv("data/" + self.dataset + "_rx.csv")